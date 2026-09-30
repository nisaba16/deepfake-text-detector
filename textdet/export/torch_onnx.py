"""
PyTorch -> ONNX for text detectors, and the parity check against the Python detector.

torch.onnx.export (TorchScript tracing) needs transformers < 5: with transformers 5.4 it fails while tracing
the attention mask (IndexError), and the dynamo exporter needs onnxscript. Use a separate environment:

    uv venv --system-site-packages .venv-export && VIRTUAL_ENV=.venv-export uv pip install -r requirements-export.txt

Tracing pitfalls handled here: the example batch is padded (so the mask path is traced, not skipped), the
model is in eval mode, and attention is "eager" (plain ops that onnxruntime-web runs everywhere).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

EXAMPLE_TEXTS = ["A short text.", "A somewhat longer example text, so that the batch needs some padding tokens."]


def require_transformers_4():
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5:
        raise RuntimeError(f"torch.onnx.export fails with transformers {transformers.__version__} (attention-mask "
                           "tracing): run the export with transformers<5 (see requirements-export.txt)")


def export_text_module(module, tokenizer, path: Path, opset: int = 17, with_token_type_ids: bool = False) -> Path:
    """Export module(input_ids, attention_mask[, token_type_ids]) -> scores with dynamic batch and length."""
    import torch
    require_transformers_4()
    module = module.eval()
    enc = tokenizer(EXAMPLE_TEXTS, padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if with_token_type_ids else [])
    if with_token_type_ids and "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    args = tuple(enc[n] for n in names)
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=names, output_names=["logits"],
                          dynamic_axes={n: {0: "batch", 1: "tokens"} for n in names} | {"logits": {0: "batch"}},
                          opset_version=opset, dynamo=False)
    module.eval()  # export can leave the module in training mode (dropout on)
    return path


def parity(reference, candidate, texts: Sequence[str]) -> Dict[str, float]:
    """Compare two detectors (e.g. the PyTorch one and its browser folder) on the same texts."""
    from scipy.stats import spearmanr
    a = next(iter(reference.score(list(texts)).values()))
    b = next(iter(candidate.score(list(texts)).values()))
    d = np.abs(a.score - b.score)
    out = {"n": len(texts), "max_abs_log_odds_diff": float(d.max()), "mean_abs_log_odds_diff": float(d.mean()),
           "spearman": float(spearmanr(a.score, b.score)[0]) if len(texts) > 2 else float("nan"),
           "decision_agreement": float(np.mean((a.p_ai >= 0.5) == (b.p_ai >= 0.5)))}
    print("Parity: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()))
    return out
