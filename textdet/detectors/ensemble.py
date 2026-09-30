"""
Average of several detectors' ranking scores, each standardized first so no detector dominates by scale.

Standardization uses reference statistics (mean, std) per member. Without them the batch being scored is
used, which is fine for ranking metrics on a whole evaluation set but makes one text's score depend on the
others: pass `reference_texts` (e.g. a few hundred human texts of the target domain) for a fixed scale.
A learned combination (logistic stacking) is a training step and is left out on purpose.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import Detector, Scores, safe_name


class MeanEnsemble(Detector):
    analysis = "ensemble"
    train_dataset = "pretrained"

    def __init__(self, members: List[Detector], weights: Optional[Sequence[float]] = None,
                 reference_texts: Optional[Sequence[str]] = None):
        if len(members) < 2:
            raise ValueError("An ensemble needs at least two detectors")
        self.members = members
        self.weights = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=float)
        self.reference = None
        if reference_texts is not None:
            self.reference = [self._member_score(m, reference_texts) for m in members]
        self.checkpoint = "+".join(m.tag for m in members)
        if all(m.train_dataset == "zeroshot" for m in members):
            self.train_dataset = "zeroshot"

    @property
    def tag(self) -> str:
        return "ens_" + safe_name(self.checkpoint)[:120]

    def describe(self):
        return {"detector": "ensemble", "checkpoint": " + ".join(m.checkpoint if hasattr(m, "checkpoint") else m.tag
                                                                for m in self.members),
                "weights": ",".join(f"{w:g}" for w in self.weights)}

    @staticmethod
    def _member_score(member: Detector, texts: Sequence[str]) -> np.ndarray:
        heads = member.score(texts)
        head = "ensemble_mean" if "ensemble_mean" in heads else next(iter(heads))
        return heads[head].score

    def score(self, texts: Sequence[str]) -> Dict[str, Scores]:
        z = np.zeros(len(texts))
        for i, member in enumerate(self.members):
            s = self._member_score(member, texts)
            ref = self.reference[i] if self.reference is not None else s
            z += self.weights[i] * (s - ref.mean()) / (ref.std() + 1e-8)
        z /= self.weights.sum()
        return {"default": Scores.from_logits(z)}

    def close(self):
        for m in self.members:
            m.close()
