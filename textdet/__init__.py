"""
textdet: the reusable core of the deepfake text detector.

    textdet.data         load any labelled text table (local file or Hugging Face) as 0=human / 1=AI
    textdet.detectors    every detector behind one interface: score(texts) -> {head: Scores}
    textdet.evaluation   metrics (AUROC, TPR at fixed FPR, thresholded scores) and the evaluation runner
    textdet.export       ONNX folders for the browser app (deepfake-detection-38502/public/models/<id>/)

The scripts in scripts/ are thin command lines over these modules. The legacy `models/` package is kept as is:
saved detectors are pickles of `models.classifiers` classes, so it must stay importable under that name.
"""

__version__ = "0.1.0"
