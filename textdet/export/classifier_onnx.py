"""
Browser model of an off-the-shelf Hugging Face sequence classifier (e.g. a RAID-trained detector).

    tokenizer.json / tokenizer_config.json   from the checkpoint (the page tokenizes with them)
    model.onnx                                input_ids, attention_mask (+ token_type_ids for BERT) -> logits
    config.json                               labels in output order; "sigmoid" for single-logit models
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from ..detectors.classifier import PRETRAINED
from .quantize import quantize_weights_int8
from .torch_onnx import export_text_module
from .webfolder import write_web_model


def export_classifier_web(checkpoint: str, out_dir: Path, *, ai_index: int = 1, max_length: int = 512,
                          precision: str = "int8", name: Optional[str] = None, description: Optional[str] = None,
                          register: bool = True) -> Path:
    """checkpoint: a repo id or a PRETRAINED short name (which also sets ai_index and max_length)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    spec = PRETRAINED.get(checkpoint)
    short = checkpoint if spec else None
    if spec:
        checkpoint, ai_index, max_length = spec.repo, spec.ai_index, min(max_length, spec.max_length)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, attn_implementation="eager").eval()
    uses_types = "token_type_ids" in tokenizer.model_input_names

    class Logits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kw = {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
            return self.m(input_ids=input_ids, attention_mask=attention_mask, **kw).logits

    n_out = model.config.num_labels
    if n_out == 1:
        # One logit: sigmoid gives P(labels[1]); ai_index -1 means the logit scores HUMAN
        labels, output = (["AI-generated", "human"] if ai_index == -1 else ["human", "AI-generated"]), "sigmoid"
    else:
        labels = [f"class {i}" for i in range(n_out)]
        labels[ai_index] = "AI-generated"
        if n_out == 2:
            labels[1 - ai_index] = "human"
        output = "logits"

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fp32 = export_text_module(Logits(model), tokenizer, work / "fp32.onnx", with_token_type_ids=uses_types)
        final = work / "model.onnx"
        if precision == "int8":
            quantize_weights_int8(fp32, final, block=32)
        elif precision == "fp16":
            quantize_weights_int8(fp32, final, block=32, matmul="fp16")
        elif precision == "fp32":
            fp32.rename(final)
        else:
            raise ValueError("precision must be int8, fp16 (int8 embeddings + fp16 matrices) or fp32")
        tokenizer.save_pretrained(work / "tok")
        params = sum(p.numel() for p in model.parameters())
        config = {
            "name": name or f"{short or checkpoint.split('/')[-1]} (off-the-shelf)",
            "description": description or (
                f"{checkpoint}, {params / 1e6:.0f}M parameters, used as published (no training by us)"
                + (f"; RAID leaderboard TPR@5%/1%FPR {spec.raid} (no attack | all attacks)" if spec else "")
                + f". {precision} weights, reads the first {max_length} tokens."),
            "model": "model.onnx",
            "labels": labels,
            "output": output,
            "preprocess": [{"op": "tokenize", "max_length": max_length}],
        }
        return write_web_model(out_dir, final, config, work / "tok" / "tokenizer.json",
                               work / "tok" / "tokenizer_config.json", register=register, move=True)
