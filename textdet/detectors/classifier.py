"""
Off-the-shelf fine-tuned AI-text classifiers from the Hugging Face Hub, used as they are (no training).

PRETRAINED lists small open detectors with their output convention and their RAID leaderboard scores
(raid-bench.xyz, test split, per-domain thresholds: TPR at 5% / 1% FPR, without and with adversarial attacks).
They were trained on RAID (and MAGE for ModernBERT): score them on RAID *train* samples and you score
their training data. The MAGE wild sets and your own data are the fair tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .base import Detector, Scores, batched, safe_name


@dataclass(frozen=True)
class PretrainedSpec:
    repo: str
    params: str
    ai_index: int           # output index of the AI class; -1: one logit that scores HUMAN (AI log-odds = -logit)
    max_length: int
    raid: str               # RAID leaderboard: TPR@5%FPR / TPR@1%FPR, no attack | all attacks
    license: str


PRETRAINED: Dict[str, PretrainedSpec] = {
    "bert-tiny-raid": PretrainedSpec("ShantanuT01/BERT-tiny-RAID", "4.4M", -1, 512,
                                     "0.911 / 0.808 | 0.842 / 0.719", "MIT"),
    "e5-small-raid": PretrainedSpec("MayZhou/e5-small-lora-ai-generated-detector", "33M", 1, 512,
                                    "0.939 / 0.828 | 0.857 / 0.731", "MIT"),
    "tmr-roberta-raid": PretrainedSpec("Oxidane/tmr-ai-text-detector", "125M", 1, 512,
                                       "0.997 / 0.986 | 0.958 / 0.902", "MIT"),
    "modernbert-raid-mage": PretrainedSpec("GeorgeDrayson/modernbert-ai-detection-raid-mage", "150M", 1, 512,
                                           "0.991 / 0.982 | 0.941 / 0.882", "Apache-2.0"),
}


class HFClassifierDetector(Detector):
    """AutoModelForSequenceClassification checkpoint; P(AI) from its logits.

    Long texts: by default the first max_length tokens are scored (as the browser does). With
    chunk_stride, every window of max_length tokens is scored and the log-odds are averaged.
    """
    analysis = "hfclf"
    train_dataset = "pretrained"

    def __init__(self, checkpoint: str, ai_index: int = 1, device: Optional[str] = None, batch_size: int = 16,
                 max_length: int = 512, chunk_stride: Optional[int] = None, revision: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        spec = PRETRAINED.get(checkpoint)
        if spec is not None:  # a short name from PRETRAINED
            checkpoint, ai_index, max_length = spec.repo, spec.ai_index, min(max_length, spec.max_length)
        self.checkpoint, self.revision, self.ai_index = checkpoint, revision, ai_index
        self.batch_size, self.max_length, self.chunk_stride = batch_size, max_length, chunk_stride
        if device is None or (str(device).startswith("cuda") and not torch.cuda.is_available()):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, revision=revision).to(self.device).eval()

    @property
    def tag(self) -> str:
        return safe_name(self.checkpoint)

    def describe(self):
        return {"detector": "hf_classifier", "checkpoint": self.checkpoint, "revision": self.revision,
                "max_len": self.max_length, "chunk_stride": self.chunk_stride}

    def _log_odds(self, logits: np.ndarray) -> np.ndarray:
        """AI log-odds from the logits: one logit (sigmoid) or a softmax over classes."""
        if logits.shape[1] == 1:
            z = logits[:, 0]
            return -z if self.ai_index == -1 else z
        ai = logits[:, self.ai_index]
        others = np.delete(logits, self.ai_index, axis=1)
        # log P(AI) - log P(not AI) under the softmax
        return ai - np.logaddexp.reduce(others, axis=1)

    def _forward(self, encodings) -> np.ndarray:
        import torch
        with torch.inference_mode():
            enc = {k: v.to(self.device) for k, v in encodings.items()}
            return self.model(**enc).logits.float().cpu().numpy()

    def score(self, texts: Sequence[str]) -> Dict[str, Scores]:
        if self.chunk_stride:
            return {"default": Scores.from_logits(self._chunked(texts))}
        order = np.argsort([len(t) for t in texts])
        z = np.zeros(len(texts))
        for idx in tqdm(list(batched(order, self.batch_size)), desc=f"{self.tag}", leave=False):
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            z[idx] = self._log_odds(self._forward(enc))
        return {"default": Scores.from_logits(z)}

    def _chunked(self, texts: Sequence[str]) -> np.ndarray:
        enc = self.tokenizer(list(texts), truncation=True, max_length=self.max_length, stride=self.chunk_stride,
                             return_overflowing_tokens=True, padding=True, return_tensors="pt")
        owner = enc.pop("overflow_to_sample_mapping").numpy()
        enc.pop("offset_mapping", None)
        z = np.zeros(len(owner))
        for idx in batched(np.arange(len(owner)), self.batch_size):
            z[idx] = self._log_odds(self._forward({k: v[idx] for k, v in enc.items()}))
        return np.array([z[owner == i].mean() for i in range(len(texts))])

    def close(self):
        import torch
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
