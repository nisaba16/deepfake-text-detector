"""
Calibrate a browser model on human texts: shift its output so the page's 50% line is the threshold that flags
only `fpr` of those human texts.

Why: detectors trained on one corpus are overconfident on human text from other domains. The RAID-trained e5-small
flags 46% of MAGE's human texts at P(AI) = 0.5 while ranking them well (AUROC 0.955). A threshold set on human
texts of the target domain fixes the false-positive rate without any AI example and without training; RAID's
leaderboard sets its thresholds the same way (per domain, on human texts).

The shift is a constant added to the AI score inside the graph, so the page needs no change:
    2-class logits   logits + [0, -t] (AI column)      ->  AI log-odds - t
    one sigmoid      raw - t (or + t when the logit scores human)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from ..detectors.webmodel import WebModelDetector
from ..evaluation.metrics import threshold_at_fpr
from .webfolder import file_version


def add_output_offset(model_path: Path, offset: np.ndarray) -> None:
    """model output := output + offset (broadcast over the batch), keeping the output name."""
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(str(model_path))
    g = model.graph
    out = g.output[0].name
    existing = next((t for t in g.initializer if t.name == out + "_offset"), None)
    if existing is not None:  # already calibrated: move the offset
        total = numpy_helper.to_array(existing) + offset.astype(np.float32).reshape(1, -1)
        existing.CopyFrom(numpy_helper.from_array(total.astype(np.float32), existing.name))
        onnx.save(model, str(model_path))
        return
    raw = out + "_uncalibrated"
    for node in g.node:
        node.output[:] = [raw if o == out else o for o in node.output]
        node.input[:] = [raw if i == out else i for i in node.input]
    g.initializer.append(numpy_helper.from_array(offset.astype(np.float32).reshape(1, -1), out + "_offset"))
    g.node.append(helper.make_node("Add", [raw, out + "_offset"], [out], name=out + "_calibration"))
    onnx.save(model, str(model_path))


def calibrate_web_model(folder: Path, human_texts: Sequence[str], fpr: float = 0.05,
                        source: str = "") -> Dict[str, float]:
    """Shift the model in `folder` so that P(AI) > 0.5 on at most `fpr` of `human_texts`. Idempotent: an existing
    calibration is replaced."""
    folder = Path(folder)
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text())
    previous = config.get("calibration", {}).get("offset", 0.0)
    det = WebModelDetector(str(folder), batch_size=8)
    z = det.score(list(human_texts))["default"].score + previous    # uncalibrated AI log-odds
    t = threshold_at_fpr(z, fpr)
    n_out = det.session.get_outputs()[0].shape[-1]
    delta = t - previous                                             # shift to add on top of the current one
    if det.output == "sigmoid":
        offset = np.array([-delta if det.ai_index == 1 else delta])
    else:
        offset = np.zeros(n_out)
        offset[det.ai_index] = -delta
    model_path = folder / config.get("model", "model.onnx")
    add_output_offset(model_path, offset)
    info = {"target_fpr": fpr, "offset": float(t), "n_human_texts": len(human_texts), "source": source}
    config["calibration"] = info
    config["version"] = file_version(model_path)
    desc = config.get("description", "").split(" Calibrated:")[0]
    config["description"] = (f"{desc} Calibrated: 50% = the threshold that flags {fpr:.0%} of "
                             f"{len(human_texts)} human texts ({source}).")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Calibrated {folder.name}: AI log-odds shifted by {-t:+.3f} (target FPR {fpr:.0%} on {len(human_texts)} "
          f"human texts of {source})")
    return info
