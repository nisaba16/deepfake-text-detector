"""
Zero-shot likelihood statistics from small causal language models (no training).

    Fast-DetectGPT (Bao et al., ICLR 2024), analytic criterion: how much more likely the observed tokens are
        than tokens sampled from the model itself, normalized by the variance of that expectation.
        One model (scoring = reference by default). Higher = more likely AI.
    Binoculars (Hans et al., ICML 2024): log-perplexity of a performer model divided by the cross-entropy
        between an observer and the performer. Two models sharing a tokenizer. Lower = more likely AI;
        the ranking score here is its negative.

Both are domain-agnostic, which is their point next to classifiers trained on one corpus; both are weaker
with small models (on RAID, Binoculars with the Falcon-7B pair reaches TPR 0.79 at 5% FPR without attacks,
far below RAID-trained classifiers). Their raw statistics are not probabilities: p_ai is a logistic of the
statistic with `calibration` = (center, scale), to be fitted on human texts of the target domain
(--optimize_threshold fpr0.05 does the same for thresholded metrics); ROC-AUC and TPR@FPR use the raw score.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .base import Detector, Scores, batched, safe_name


class _CausalLM:
    def __init__(self, name: str, device, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        if dtype is None:   # bf16 needs Ampere or newer (a P100 has none): fp16 there, fp32 on CPU
            dtype = torch.float32 if device.type != "cuda" else (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()


def _device(device):
    import torch
    if device is None or (str(device).startswith("cuda") and not torch.cuda.is_available()):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class _LMStatDetector(Detector):
    analysis = "lmstat"
    train_dataset = "zeroshot"
    higher_is_ai = True

    def __init__(self, calibration: Optional[Tuple[float, float]] = None, batch_size: int = 8,
                 max_length: int = 512):
        self.calibration = calibration
        self.batch_size, self.max_length = batch_size, max_length

    def statistic(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def score(self, texts: Sequence[str]) -> Dict[str, Scores]:
        order = np.argsort([len(t) for t in texts])
        stat = np.zeros(len(texts))
        for idx in tqdm(list(batched(order, self.batch_size)), desc=self.tag, leave=False):
            stat[idx] = self.statistic([texts[i] for i in idx])
        score = stat if self.higher_is_ai else -stat
        center, scale = self.calibration or (0.0, 1.0)
        z = (score - center) / scale
        return {"default": Scores(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))), score)}

    def _encode(self, lm: _CausalLM, texts):
        enc = lm.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt")
        return {k: v.to(lm.model.device) for k, v in enc.items()}


class FastDetectGPT(_LMStatDetector):
    """Analytic Fast-DetectGPT criterion (sampling discrepancy), reference model = scoring model by default."""

    def __init__(self, scoring_model: str = "Qwen/Qwen2.5-0.5B", reference_model: Optional[str] = None,
                 device: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        dev = _device(device)
        self.scoring = _CausalLM(scoring_model, dev)
        self.reference = _CausalLM(reference_model, dev) if reference_model and reference_model != scoring_model else None
        self.checkpoint = scoring_model + (f"+{reference_model}" if self.reference else "")

    @property
    def tag(self) -> str:
        return "fastdetectgpt_" + safe_name(self.checkpoint)

    def describe(self):
        return {"detector": "fast_detectgpt", "checkpoint": self.checkpoint, "max_len": self.max_length}

    def statistic(self, texts):
        import torch
        enc = self._encode(self.scoring, texts)
        with torch.inference_mode():
            all_s = self.scoring.model(**enc).logits
            all_r = self.reference.model(**enc).logits if self.reference else all_s
            out = []
            # One text at a time: [tokens, vocab] fp32 tensors (Qwen: 151k vocab) are large enough per text
            for b in range(all_s.shape[0]):
                n = int(enc["attention_mask"][b].sum())
                labels = enc["input_ids"][b, 1:n]
                lprobs = torch.log_softmax(all_s[b, :n - 1].float(), dim=-1)
                probs_ref = torch.softmax(all_r[b, :n - 1].float(), dim=-1)
                ll = lprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                mean_ref = (probs_ref * lprobs).sum(-1)
                var_ref = (probs_ref * lprobs.square()).sum(-1) - mean_ref.square()
                out.append(float((ll - mean_ref).sum() / var_ref.sum().clamp_min(1e-8).sqrt()) if n > 1 else 0.0)
        return np.array(out)


class Binoculars(_LMStatDetector):
    """Binoculars score (performer log-perplexity / observer-performer cross-entropy). Lower = AI."""
    higher_is_ai = False

    def __init__(self, observer: str = "Qwen/Qwen2.5-0.5B", performer: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 device: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        dev = _device(device)
        self.observer = _CausalLM(observer, dev)
        self.performer = _CausalLM(performer, dev)
        if self.observer.tokenizer.get_vocab() != self.performer.tokenizer.get_vocab():
            raise ValueError("Binoculars needs two models with the same tokenizer")
        self.checkpoint = f"{observer}+{performer}"

    @property
    def tag(self) -> str:
        return "binoculars_" + safe_name(self.checkpoint)

    def describe(self):
        return {"detector": "binoculars", "checkpoint": self.checkpoint, "max_len": self.max_length}

    def statistic(self, texts):
        import torch
        enc = self._encode(self.observer, texts)
        with torch.inference_mode():
            all_obs = self.observer.model(**enc).logits
            all_perf = self.performer.model(**enc).logits
            out = []
            for b in range(all_obs.shape[0]):          # one text at a time, as in FastDetectGPT.statistic
                n = int(enc["attention_mask"][b].sum())
                if n < 2:
                    out.append(1.0)
                    continue
                labels = enc["input_ids"][b, 1:n]
                obs, perf = all_obs[b, :n - 1].float(), all_perf[b, :n - 1].float()
                log_ppl = torch.nn.functional.cross_entropy(perf, labels)                 # mean over tokens
                x_ent = (torch.softmax(obs, -1) * -torch.log_softmax(perf, -1)).sum(-1).mean()
                out.append(float(log_ppl / x_ent))
        return np.array(out)
