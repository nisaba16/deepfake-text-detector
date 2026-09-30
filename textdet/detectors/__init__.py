"""
Detectors behind one interface (see base.py). Heavy dependencies are imported by each module on use.

    decision.py       Jev-like zero-shot decision models: Julia 1 (native), Laya (laya package)
    lm_stats.py       zero-shot likelihood statistics: Fast-DetectGPT, Binoculars
    classifier.py     off-the-shelf fine-tuned classifiers from the Hugging Face Hub (RAID-trained, ...)
    webmodel.py       a browser model folder (config.json + model.onnx + tokenizer), run with onnxruntime
    saved.py          the project's saved detectors (embedding/TF-IDF/perplexity/PHD features + classifier)
    ensemble.py       average of several detectors' standardized scores
    registry.py       build any of them from a spec string: julia, laya, hf:<repo>, web:<folder>, saved:<pkl>, ...
"""
from .base import Detector, Scores

__all__ = ["Detector", "Scores"]
