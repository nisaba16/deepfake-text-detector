"""
Browser model of a Jev-like decision model (Julia 1): the question is baked into the ONNX graph.

The official export (SupersonicLabs/Julia-1-ONNX) takes the whole decision sequence plus the [MASK] positions:

    input_ids, attention_mask [B, T]   marker_pos, marker_mask [B, 2]   qtype [B]   ->   logits [B, options]

The page only tokenizes the pasted text. So the wrapper graph takes the text tokens (tokenized with a
post-processor that appends <eos> only, see tokenizer_with_template) and prepends the constant question prefix,
'[CLS] noul question: ... [SEP] [MASK] <false option> [MASK] <true option> [SEP]', whose [MASK] positions are
constants too. Output: logits [B, 2] over [false, true] -> labels ["human", "AI-generated"] for a question
whose "true" means AI. No PyTorch export is involved, so transformers versions do not matter here.

Size: 144M parameters, 68% of them the 256k-token vocabulary. fp32 550 MiB; see PRECISIONS below.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..detectors.decision import (JULIA_CHECKPOINT, QTYPES, QUESTION_PRESETS, DecisionTokens,
                                  download_decision_checkpoint, encode_prefix, noul_options)
from .quantize import quantize_weights_int8
from .webfolder import tokenizer_with_template, write_web_model

JULIA_ONNX_REPO = "SupersonicLabs/Julia-1-ONNX"
INNER = "_decision_"   # prefix of the renamed inputs of the wrapped graph

# Weight precisions (textdet.export.quantize). Measured on MAGE texts against the fp32 model (log-odds of AI):
#   mixed  int8 embeddings (block 32) + fp16 matrices   196 MiB   mean |error| 0.07, rank correlation 0.997
#   int8   int8 embeddings and matrices (block 32)       158 MiB   mean |error| 0.25, rank correlation 0.97
#   fp32   the official weights                          550 MiB   exact
# (onnxruntime's quantize_dynamic, 140 MiB, is unusable: it also quantizes activations; correlation 0.02.)
PRECISIONS = {"mixed": dict(block=32, matmul="fp16"), "int8": dict(block=32, matmul="int8")}


def download_official_onnx(repo: str = JULIA_ONNX_REPO, revision: Optional[str] = None) -> Path:
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo, revision=revision, allow_patterns=["model.onnx", "model.onnx.data"]))
    return root / "model.onnx"


def wrap_decision_graph(src_onnx: Path, prefix: list, markers: list, qtype: int, mask_id: int, unk_id: int,
                        out_path: Path) -> Path:
    """Write out_path: the decision graph with the question built in. Its weights stay in the source's
    external data file, so out_path must be in the same directory as src_onnx (checked)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if out_path.parent.resolve() != src_onnx.parent.resolve():
        raise ValueError("The wrapper must sit next to the source graph (it shares its external weights)")
    model = onnx.load(str(src_onnx), load_external_data=False)
    g = model.graph
    expected = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
    if {i.name for i in g.input} != expected:
        raise ValueError(f"Unexpected decision graph inputs {[i.name for i in g.input]}")

    # 1. Rename the original inputs everywhere they are consumed
    for node in g.node:
        for k, name in enumerate(node.input):
            if name in expected:
                node.input[k] = INNER + name
    del g.input[:]

    # The graph's shape annotations name the full sequence length "tokens": drop them, they no longer hold
    del g.value_info[:]

    # 2. New inputs: the text tokens (ending with <eos>) and their mask
    g.input.extend([helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "text_tokens"]),
                    helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "text_tokens"])])

    lp, k = len(prefix), len(markers)
    consts = {
        "prefix": np.array([prefix], dtype=np.int64),
        "prefix_ones": np.ones((1, lp), dtype=np.int64),
        "markers": np.array([markers], dtype=np.int64),
        "marker_true": np.ones((1, k), dtype=bool),
        "qtype": np.array([qtype], dtype=np.int64),
        "lp": np.array([lp], dtype=np.int64),
        "k": np.array([k], dtype=np.int64),
        "zero": np.array([0], dtype=np.int64),
        "mask_id": np.array(mask_id, dtype=np.int64),
        "unk_id": np.array(unk_id, dtype=np.int64),
    }
    g.initializer.extend(numpy_helper.from_array(v, name=f"{INNER}c_{n}") for n, v in consts.items())
    c = lambda n: f"{INNER}c_{n}"
    t = lambda n: f"{INNER}t_{n}"
    nodes = [
        helper.make_node("Shape", ["input_ids"], [t("shape")]),
        helper.make_node("Gather", [t("shape"), c("zero")], [t("batch")], axis=0),          # [1]
        helper.make_node("Concat", [t("batch"), c("lp")], [t("prefix_shape")], axis=0),     # [B, Lp]
        helper.make_node("Concat", [t("batch"), c("k")], [t("marker_shape")], axis=0),      # [B, K]
        helper.make_node("Expand", [c("prefix"), t("prefix_shape")], [t("prefix_b")]),
        helper.make_node("Expand", [c("prefix_ones"), t("prefix_shape")], [t("prefix_mask_b")]),
        # A literal "<mask>" pasted in the text must not look like an option marker
        helper.make_node("Equal", ["input_ids", c("mask_id")], [t("is_mask")]),
        helper.make_node("Where", [t("is_mask"), c("unk_id"), "input_ids"], [t("text_ids")]),
        helper.make_node("Concat", [t("prefix_b"), t("text_ids")], [INNER + "input_ids"], axis=1),
        helper.make_node("Concat", [t("prefix_mask_b"), "attention_mask"], [INNER + "attention_mask"], axis=1),
        helper.make_node("Expand", [c("markers"), t("marker_shape")], [INNER + "marker_pos"]),
        helper.make_node("Expand", [c("marker_true"), t("marker_shape")], [INNER + "marker_mask"]),
        helper.make_node("Expand", [c("qtype"), t("batch")], [INNER + "qtype"]),
    ]
    for n in reversed(nodes):
        g.node.insert(0, n)
    onnx.save(model, str(out_path))
    return out_path


def consolidate(src: Path, dst: Path) -> Path:
    """One self-contained file (the page loads a single model file)."""
    import onnx
    model = onnx.load(str(src), load_external_data=True)
    onnx.save(model, str(dst), save_as_external_data=False)
    return dst


def export_decision_web(out_dir: Path, *, question: Tuple[Dict, bool] = QUESTION_PRESETS["ai_generated"],
                        checkpoint: str = JULIA_CHECKPOINT, onnx_path: Optional[Path] = None,
                        max_length: int = 1024, head_length: int = 256, precision: str = "mixed",
                        name: str = "Julia 1 · Jev-like (zero-shot)", description: Optional[str] = None,
                        register: bool = True) -> Path:
    """Export a Julia-format decision model with one question as a browser model folder."""
    from tokenizers import Tokenizer

    q, ai_is_true = question
    if q.get("type", "noul") != "noul":
        raise ValueError("Only noul (yes/no) questions make a detector")
    root = download_decision_checkpoint(checkpoint)
    tok_dir = root / "tokenizer"
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    special = DecisionTokens.from_tokenizer_dir(tok_dir)
    prefix, markers = encode_prefix(tokenizer, special, q["instructions"], noul_options(q), "noul", head_length)
    unk = tokenizer.token_to_id("<unk>")
    if unk is None:
        raise ValueError("The tokenizer has no <unk> token")

    src = Path(onnx_path) if onnx_path else download_official_onnx()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # onnx refuses external data behind a symlink (the Hugging Face cache) or a hard link: copy it
        for f in (src, src.with_name(src.name + ".data")):
            if f.exists():
                shutil.copyfile(f.resolve(), work / f.name)
        wrapper = wrap_decision_graph(work / src.name, prefix, markers, QTYPES["noul"], special.mask, unk,
                                      work / "wrapper.onnx")
        model_file = work / "model_web.onnx"
        if precision in PRECISIONS:
            quantize_weights_int8(wrapper, model_file, **PRECISIONS[precision])
        elif precision == "fp32":
            consolidate(wrapper, model_file)
        else:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)} or fp32")

        labels = ["human", "AI-generated"] if ai_is_true else ["AI-generated", "human"]
        tok_json = json.loads((tok_dir / "tokenizer.json").read_text())
        sep_token = tokenizer.id_to_token(special.sep)
        config = {
            "name": name,
            "description": description or (
                f"Zero-shot: Julia 1 (Supersonic Labs, mmBERT-small, 144M parameters, Apache-2.0) is asked "
                f"\"{q['instructions']}\". Not trained for this task: expect a bias towards answering yes. "
                f"{precision}, reads the first {max_length - len(prefix) - 1} tokens."),
            "model": "model.onnx",
            "labels": labels,
            "output": "logits",
            # The graph adds the question: the page tokenizes the text, up to max_length - prefix tokens
            "preprocess": [{"op": "tokenize", "max_length": max_length - len(prefix)}],
        }
        return write_web_model(out_dir, model_file, config, tokenizer_with_template(tok_json, ["$A", sep_token]),
                               tok_dir / "tokenizer_config.json", register=register, move=True)
