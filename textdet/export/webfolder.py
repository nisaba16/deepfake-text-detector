"""
Write a model folder for the browser app (deepfake-detection-38502/public/models/<id>/), the same contract
the image models follow:

    config.json            name, description, task "text", version, labels, output, preprocess [tokenize]
    model.onnx             one graph: input_ids / attention_mask (int64 [batch, seq]) -> one score per label
    tokenizer.json         read by tokenizers.js (+ tokenizer_config.json)
    ../index.json          the folder id is added to the list the page loads
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

FRONTEND_MODELS = Path(__file__).resolve().parents[3] / "deepfake-detection-38502" / "public" / "models"


def file_version(path: Path) -> str:
    """First 12 hex digits of the file's SHA-256: browsers re-download the model when it changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def tokenizer_with_template(tokenizer_json: Dict[str, Any], single: List[str]) -> Dict[str, Any]:
    """Copy of a tokenizer.json whose post-processor wraps one text as `single`, e.g. ["$A", "<eos>"].

    Used when the model graph adds its own prefix: the page then only tokenizes the text, and its
    truncation (which keeps special tokens) keeps the closing token.
    """
    tok = json.loads(json.dumps(tokenizer_json))
    added = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
    items, specials = [], {}
    for piece in single:
        if piece == "$A":
            items.append({"Sequence": {"id": "A", "type_id": 0}})
        else:
            items.append({"SpecialToken": {"id": piece, "type_id": 0}})
            specials[piece] = {"id": piece, "ids": [added[piece]], "tokens": [piece]}
    tok["post_processor"] = {"type": "TemplateProcessing", "single": items, "pair": items + [
        {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": specials}
    tok["truncation"] = None
    tok["padding"] = None
    return tok


def write_web_model(out_dir: Union[str, Path], model_file: Union[str, Path], config: Dict[str, Any],
                    tokenizer_json: Union[Dict[str, Any], str, Path],
                    tokenizer_config: Optional[Union[Dict[str, Any], str, Path]] = None,
                    register: bool = True, move: bool = False) -> Path:
    """Create or replace <out_dir> with the model, its config (version = hash of the model) and tokenizer.

    register: add the folder name to <out_dir>/../index.json (kept in order; the first entry is the default).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / config.get("model", "model.onnx")
    if Path(model_file).resolve() != target.resolve():
        (shutil.move if move else shutil.copyfile)(str(model_file), str(target))

    def dump(obj, name):
        if isinstance(obj, (str, Path)):
            shutil.copyfile(obj, out_dir / name)
        else:
            # Compact JSON: tokenizer.json files are fetched on every visit
            (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

    dump(tokenizer_json, "tokenizer.json")
    if tokenizer_config is not None:
        dump(tokenizer_config, "tokenizer_config.json")

    config = {**config, "task": "text", "version": file_version(target)}
    ordered = {k: config[k] for k in ("name", "description", "task", "version", "model", "labels", "output",
                                      "preprocess") if k in config}
    ordered.update({k: v for k, v in config.items() if k not in ordered})
    (out_dir / "config.json").write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")

    if register:
        index = out_dir.parent / "index.json"
        ids = json.loads(index.read_text()) if index.exists() else []
        if out_dir.name not in ids:
            ids.append(out_dir.name)
            index.write_text(json.dumps(ids) + "\n")

    sizes = {p.name: p.stat().st_size / 2 ** 20 for p in out_dir.iterdir() if p.is_file()}
    print(f"Wrote {out_dir}: " + ", ".join(f"{n} {s:.1f} MiB" for n, s in sorted(sizes.items())))
    return out_dir
