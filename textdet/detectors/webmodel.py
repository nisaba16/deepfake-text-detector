"""
Run a browser model folder (deepfake-detection-38502 contract) in Python, exactly as the page does:

    <folder>/config.json      task "text", labels, output (logits | probs | sigmoid), preprocess [tokenize]
    <folder>/model.onnx       input_ids / attention_mask (/ token_type_ids) -> one score per label
    <folder>/tokenizer.json   (+ tokenizer_config.json)

So the file that ships to the browser is the one that gets evaluated, on any dataset, and parity with the
PyTorch detector it was exported from can be checked (textdet.export.parity).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .base import Detector, Scores, batched, safe_name

# The page colors labels with these patterns (src/components/Detector.tsx): the same decide which is AI
FAKE_LABEL = re.compile(r"fake|generated|synthetic|manipulated|machine|^ai$", re.I)


def truncate_keep_specials(ids: List[int], body: List[int], max_length: Optional[int]) -> List[int]:
    """Python port of encode() in src/lib/preprocess/text.ts: cut the text, keep the special tokens."""
    if not max_length or len(ids) <= max_length:
        return ids
    specials = len(ids) - len(body)
    start = 0
    while start < specials and ids[start:start + len(body)] != body:
        start += 1
    return ids[:start] + body[:max(max_length - specials, 0)] + ids[start + len(body):]


class WebModelDetector(Detector):
    """A text model folder of the browser app, run with onnxruntime."""
    analysis = "webmodel"
    train_dataset = "pretrained"

    def __init__(self, folder: str, batch_size: int = 1, providers: Optional[Sequence[str]] = None,
                 threads: Optional[int] = None):
        """batch_size 1 reproduces the page (one text per run, no padding); larger batches pad on the right,
        which is exact for models that honor attention_mask."""
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.folder = Path(folder)
        self.config = json.loads((self.folder / "config.json").read_text())
        if self.config.get("task") != "text":
            raise ValueError(f"{folder}: not a text model (task={self.config.get('task')})")
        self.labels = self.config["labels"]
        self.output = self.config.get("output", "logits")
        step = next((s for s in self.config["preprocess"] if s["op"] == "tokenize"), {})
        self.max_length = step.get("max_length")
        self.tokenizer = Tokenizer.from_file(str(self.folder / "tokenizer.json"))
        tok_cfg = self.folder / "tokenizer_config.json"
        pad = json.loads(tok_cfg.read_text()).get("pad_token") if tok_cfg.exists() else None
        pad = pad.get("content") if isinstance(pad, dict) else pad
        self.pad_id = (self.tokenizer.token_to_id(pad) if pad else None) or 0
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        model = self.config.get("model", "model.onnx")
        if re.match(r"^https?://", model):
            raise ValueError("The model file is hosted remotely: download it next to config.json first")
        self.session = ort.InferenceSession(str(self.folder / model), opts,
                                            providers=list(providers or ["CPUExecutionProvider"]))
        self.input_names = [i.name for i in self.session.get_inputs()]
        unknown = set(self.input_names) - {"input_ids", "attention_mask", "token_type_ids"}
        if unknown:
            raise ValueError(f"Inputs {sorted(unknown)} are not fed by the page (input_ids, attention_mask, token_type_ids)")
        ai = [i for i, label in enumerate(self.labels) if FAKE_LABEL.search(label)]
        if len(ai) != 1:
            raise ValueError(f"Cannot tell which of {self.labels} is the AI label")
        self.ai_index = ai[0]
        self.batch_size = batch_size
        self.checkpoint = str(self.folder)

    @property
    def tag(self) -> str:
        return safe_name(self.folder.name)

    def describe(self):
        return {"detector": "webmodel", "checkpoint": str(self.folder), "revision": self.config.get("version"),
                "max_len": self.max_length}

    def encode(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if self.max_length and len(ids) > self.max_length:
            body = self.tokenizer.encode(text, add_special_tokens=False).ids
            ids = truncate_keep_specials(ids, body, self.max_length)
        return ids

    def raw_outputs(self, texts: Sequence[str]) -> np.ndarray:
        encoded = [self.encode(t) for t in texts]
        order = np.argsort([len(e) for e in encoded])
        out: Optional[np.ndarray] = None
        for idx in tqdm(list(batched(order, self.batch_size)), desc=self.tag, leave=False,
                        disable=len(texts) < 50):
            rows = [encoded[i] for i in idx]
            n = max(len(r) for r in rows)
            ids = np.full((len(rows), n), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(rows), n), dtype=np.int64)
            for j, r in enumerate(rows):
                ids[j, :len(r)] = r
                mask[j, :len(r)] = 1
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            res = self.session.run(None, {k: feeds[k] for k in self.input_names})[0].reshape(len(rows), -1)
            if out is None:
                out = np.zeros((len(texts), res.shape[1]), dtype=np.float64)
            out[idx] = res
        return out if out is not None else np.zeros((0, len(self.labels)))

    def log_odds(self, raw: np.ndarray) -> np.ndarray:
        """AI log-odds from the raw outputs, with the page's output conventions."""
        if self.output == "sigmoid":           # one score: sigmoid gives P(labels[1])
            z = raw[:, 0]
            return z if self.ai_index == 1 else -z
        if self.output == "probs":
            p = np.clip(raw[:, self.ai_index], 1e-12, 1 - 1e-12)
            return np.log(p) - np.log1p(-p)
        others = np.delete(raw, self.ai_index, axis=1)
        return raw[:, self.ai_index] - np.logaddexp.reduce(others, axis=1)

    def score(self, texts: Sequence[str]) -> Dict[str, Scores]:
        return {"default": Scores.from_logits(self.log_odds(self.raw_outputs(texts)))}
