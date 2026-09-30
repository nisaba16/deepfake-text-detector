"""
Browser model of a detector saved by scripts/train_and_save_detector.py (embedding features + classifier).

Everything after the tokenizer goes in one graph, the way the saved detector computes it:

    encoder (cut after the chosen layer) -> hidden_states[layer] over the valid tokens ([CLS]/[SEP] included)
    -> optional per-token L2 normalization -> pooling (mean | max | first | last | mean_std)
    -> StandardScaler -> PCA (when fitted) -> logistic regression       BinaryDetector(classifier_type="lr")
                                            -> RBF one-class SVM          OutlierDetections(detector_type="ocsvm")
    -> one logit = log-odds of AI ("output": "sigmoid", labels ["human", "AI-generated"])

For the one-class SVM, P(AI) = sigmoid(-decision_function) as OutlierDetections.predict does, so the logit is
-decision_function. Training ran the encoder in fp16; the graph runs fp32, so scores differ slightly (see the
parity printed by scripts/export_web_model.py).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ..detectors.saved import load_saved_detector
from .quantize import quantize_weights_int8
from .torch_onnx import export_text_module
from .webfolder import write_web_model

POOLINGS = ("mean", "max", "first", "last", "mean_std")


def _cut_encoder(encoder, layer: int) -> int:
    """Keep only the layers hidden_states[layer] needs; returns the index to read in the cut model.

    One extra layer is kept when possible: for decoders (Qwen) the LAST hidden state gets the final norm,
    intermediate ones do not.
    """
    for attr in ("encoder.layer", "layers", "h", "transformer.layer"):
        obj, parent, name = encoder, None, None
        for part in attr.split("."):
            parent, name, obj = obj, part, getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            n = len(obj)
            if layer < 0:
                layer += n + 1
            keep = min(n, layer + 1)
            setattr(parent, name, type(obj)(list(obj)[:keep]))
            if hasattr(encoder.config, "num_hidden_layers"):
                encoder.config.num_hidden_layers = keep
            return layer
    return layer


def export_saved_web(model_path: str, out_dir: Path, *, precision: str = "int8", max_length: Optional[int] = None,
                     name: Optional[str] = None, description: Optional[str] = None, register: bool = True) -> Path:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    detector, meta = load_saved_detector(model_path)
    if meta.get("analysis_type") != "embedding":
        raise ValueError("Only embedding detectors can be exported (TF-IDF, perplexity and PHD are not graphs)")
    if meta.get("use_specialized_extraction"):
        raise NotImplementedError("Specialized extraction is not supported yet: export the layer/pooling detectors")
    pooling, normalize = meta.get("pooling", "mean"), bool(meta.get("normalize", False))
    if pooling not in POOLINGS:
        raise ValueError(f"Pooling {pooling!r} cannot be exported (supported: {POOLINGS})")
    model_name = meta["model_name"]
    max_length = max_length or int(meta.get("max_length", 512))

    # --- the fitted head as tensors ---
    kind = type(detector).__name__
    scaler, pca = detector.scaler, detector.pca
    if kind == "BinaryDetector":
        from sklearn.linear_model import LogisticRegression
        if not isinstance(detector.classifier, LogisticRegression):
            raise NotImplementedError(f"Only logistic regression heads are exported, not {type(detector.classifier).__name__}")
        head = ("lr", detector.classifier.coef_.astype(np.float32), detector.classifier.intercept_.astype(np.float32))
    elif kind == "OutlierDetections" and detector.detector_type == "ocsvm":
        svm = detector.outlier_detector
        if svm.kernel != "rbf":
            raise NotImplementedError("Only RBF one-class SVMs are exported")
        head = ("ocsvm", svm.support_vectors_.astype(np.float32), svm.dual_coef_.astype(np.float32),
                np.float32(svm.intercept_[0]), np.float32(svm._gamma))
    else:
        raise NotImplementedError(f"Cannot export a {kind} ({getattr(detector, 'detector_type', '')})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager").float().eval()
    layer = _cut_encoder(encoder, int(meta.get("layer", -1)))

    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))

    class SavedDetectorGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.register_buffer("mu", t(scaler.mean_))
            self.register_buffer("sd", t(scaler.scale_))
            self.use_pca = pca is not None
            if self.use_pca:
                components = pca.components_
                if getattr(pca, "whiten", False):
                    components = components / np.sqrt(pca.explained_variance_)[:, None]
                self.register_buffer("pca_mean", t(pca.mean_))
                self.register_buffer("pca_components", t(components))
            if head[0] == "lr":
                self.register_buffer("coef", t(head[1]))
                self.register_buffer("bias", t(head[2]))
            else:
                self.register_buffer("sv", t(head[1]))
                self.register_buffer("dual", t(head[2]))
                self.register_buffer("rho", t([head[3]]))
                self.register_buffer("gamma", t([head[4]]))

        def forward(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True).hidden_states[layer]
            if normalize:
                h = F.normalize(h, p=2, dim=-1)
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            n = m.sum(1).clamp_min(1)
            if pooling == "mean":
                x = (h * m).sum(1) / n
            elif pooling == "max":
                x = h.masked_fill(m == 0, float("-inf")).max(1).values
            elif pooling == "first":
                x = h[:, 0]
            elif pooling == "last":
                last = attention_mask.sum(1) - 1   # right padding: last valid token
                x = h[torch.arange(h.shape[0]), last]
            else:  # mean_std, population std as numpy's X.std(axis=0)
                mu = (h * m).sum(1) / n
                var = (((h - mu.unsqueeze(1)) ** 2) * m).sum(1) / n
                x = torch.cat([mu, var.clamp_min(0).sqrt()], dim=-1)
            x = (x - self.mu) / self.sd
            if self.use_pca:
                x = (x - self.pca_mean) @ self.pca_components.T
            if head[0] == "lr":
                return x @ self.coef.T + self.bias                      # AI log-odds
            d2 = (x * x).sum(-1, keepdim=True) - 2 * x @ self.sv.T + (self.sv * self.sv).sum(-1)
            df = torch.exp(-self.gamma * d2.clamp_min(0)) @ self.dual.T + self.rho
            return -df                                                  # P(AI) = sigmoid(-decision_function)

    graph = SavedDetectorGraph().eval()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fp32 = export_text_module(graph, tokenizer, work / "fp32.onnx")
        final = work / "model.onnx"
        if precision == "int8":
            quantize_weights_int8(fp32, final, block=32)
        elif precision == "fp32":
            fp32.rename(final)
        else:
            raise ValueError("precision must be int8 or fp32")
        tokenizer.save_pretrained(work / "tok")
        stem = Path(model_path).stem
        config = {
            "name": name or f"{model_name.split('/')[-1]} L{meta.get('layer')} {pooling} + {head[0]} ({meta.get('dataset_used')})",
            "description": description or (
                f"Trained here: {model_name} layer {meta.get('layer')} {pooling}"
                f"{' (L2-normalized tokens)' if normalize else ''} + {head[0].upper()}, fitted on "
                f"{meta.get('train_size', '?')} rows of {meta.get('dataset_used')} ({stem}). {precision} weights, "
                f"reads the first {max_length} tokens."),
            "model": "model.onnx",
            "labels": ["human", "AI-generated"],
            "output": "sigmoid",
            "preprocess": [{"op": "tokenize", "max_length": max_length}],
        }
        return write_web_model(out_dir, final, config, work / "tok" / "tokenizer.json",
                               work / "tok" / "tokenizer_config.json", register=register, move=True)
