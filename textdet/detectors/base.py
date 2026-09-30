"""
One interface for every detector: score(texts) -> {head: Scores}.

A head is one output of a detector: a question of a Jev-like decision model, a statistic of a zero-shot
method, or "default" for single-output detectors. Every head gives
    p_ai   P(AI) in [0, 1], thresholded at 0.5 unless a threshold is tuned
    score  a ranking score (higher = more likely AI), unrounded, used for ROC-AUC and TPR@FPR;
           often the log-odds of p_ai, or the raw statistic of a zero-shot method
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass
class Scores:
    p_ai: np.ndarray
    score: np.ndarray

    def __post_init__(self):
        self.p_ai = np.asarray(self.p_ai, dtype=np.float64).reshape(-1)
        self.score = np.asarray(self.score, dtype=np.float64).reshape(-1)
        if self.p_ai.shape != self.score.shape:
            raise ValueError(f"p_ai {self.p_ai.shape} and score {self.score.shape} differ")

    @classmethod
    def from_logits(cls, log_odds) -> "Scores":
        z = np.asarray(log_odds, dtype=np.float64)
        return cls(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))), z)

    @classmethod
    def concat(cls, parts: Sequence["Scores"]) -> "Scores":
        return cls(np.concatenate([p.p_ai for p in parts]), np.concatenate([p.score for p in parts]))


class Detector:
    """Base class. Subclasses set the naming attributes and implement score()."""

    #: what the detector is, for result file names: <train_dataset>_<tag>_<analysis>_<head>
    analysis: str = "detector"
    #: the data it was fitted on by us: "zeroshot" when it was not trained, "pretrained" when trained by others
    train_dataset: str = "zeroshot"

    @property
    def tag(self) -> str:
        """Filesystem-safe name of the checkpoint(s)."""
        return safe_name(getattr(self, "checkpoint", type(self).__name__))

    @property
    def heads(self) -> list:
        return ["default"]

    def score(self, texts: Sequence[str]) -> Dict[str, Scores]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """Columns stored next to the metrics in the summary files (checkpoint, revision, question...)."""
        return {"detector": type(self).__name__, "checkpoint": getattr(self, "checkpoint", None)}

    def summary_stem(self, head: str) -> str:
        """Result file stem, parsed back by scripts/analyze_cross_dataset.py."""
        return f"{self.train_dataset}_{self.tag}_{self.analysis}_{safe_name(head)}"

    def predictions_stem(self) -> str:
        """Stem of the per-text predictions files (one file per dataset, a column per head)."""
        return f"{self.train_dataset}_{self.tag}_{self.analysis}"

    def close(self) -> None:
        """Release the model (GPU memory) when the evaluation is done."""


def safe_name(s: str) -> str:
    """Filesystem-safe name, sanitized like train_and_save_detector.generate_model_name."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    return re.sub(r"_+", "_", s).strip("._-") or "model"


def batched(items: Sequence, size: int):
    for start in range(0, len(items), max(1, size)):
        yield items[start:start + size]
