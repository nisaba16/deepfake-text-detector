"""
Build a detector from a spec string, so every script accepts the same --detector values:

    julia[:<repo or dir>]                   Julia 1 decision model, zero-shot (default SupersonicLabs/Julia-1)
    laya[:<repo or dir>[/<subfolder>]]      Laya decision model, zero-shot (laya package)
    hf:<repo or short name>                 off-the-shelf classifier; short names: see classifier.PRETRAINED
    web:<folder>                            a browser model folder (config.json + model.onnx + tokenizer)
    saved:<detector.pkl>                    a detector saved by scripts/train_and_save_detector.py
    fastdetectgpt[:<scoring>[,<reference>]] Fast-DetectGPT analytic criterion (default Qwen/Qwen2.5-0.5B)
    binoculars[:<observer>,<performer>]     Binoculars (default Qwen2.5-0.5B base / instruct)
    ens:<spec>+<spec>[+...]                 mean of the members' standardized scores
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Detector


def build_detector(spec: str, *, device: Optional[str] = None, batch_size: Optional[int] = None,
                   max_length: Optional[int] = None, questions: Optional[Dict] = None,
                   **extra: Any) -> Detector:
    kind, _, arg = spec.partition(":")
    kw: Dict[str, Any] = {}
    if batch_size:
        kw["batch_size"] = batch_size

    if kind == "ens":
        from .ensemble import MeanEnsemble
        members = [build_detector(s, device=device, batch_size=batch_size, max_length=max_length,
                                  questions=questions, **extra) for s in arg.split("+")]
        return MeanEnsemble(members)
    if kind == "julia":
        from .decision import JULIA_CHECKPOINT, JuliaDetector
        if max_length:
            kw["max_length"] = max_length
        return JuliaDetector(arg or JULIA_CHECKPOINT, questions=questions, device=device, **kw)
    if kind == "laya":
        from .decision import LAYA_CHECKPOINT, LayaDetector
        checkpoint, subfolder = arg or LAYA_CHECKPOINT, None
        for sub in ("multilingual", "typed-decisions"):
            if checkpoint.endswith("/" + sub):
                checkpoint, subfolder = checkpoint[: -len(sub) - 1], sub
        return LayaDetector(checkpoint, subfolder=subfolder, questions=questions, device=device,
                            max_len=max_length, **kw)
    if kind == "hf":
        from .classifier import HFClassifierDetector
        if max_length:
            kw["max_length"] = max_length
        return HFClassifierDetector(arg, device=device, chunk_stride=extra.get("chunk_stride"), **kw)
    if kind == "web":
        from .webmodel import WebModelDetector
        return WebModelDetector(arg, **kw)
    if kind == "saved":
        from .saved import SavedDetector
        return SavedDetector(arg, device=device or "cuda:0", max_length=max_length or 512, **kw)
    if kind in ("fastdetectgpt", "binoculars"):
        from .lm_stats import Binoculars, FastDetectGPT
        if max_length:
            kw["max_length"] = max_length
        models = [m for m in arg.split(",") if m]
        if kind == "fastdetectgpt":
            return FastDetectGPT(*models[:2], device=device, **kw)
        if models and len(models) != 2:
            raise ValueError("binoculars:<observer>,<performer>")
        return Binoculars(*models, device=device, **kw)
    raise ValueError(f"Unknown detector spec {spec!r} (see textdet/detectors/registry.py)")
