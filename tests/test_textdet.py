"""Fast checks of textdet (no model download): python -m pytest tests/test_textdet.py -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textdet.data import TextSet, labels_from_column, load_table, prepare, resolve, select_eval_part, subsample
from textdet.detectors.base import Scores
from textdet.detectors.webmodel import truncate_keep_specials
from textdet.evaluation.metrics import apply_threshold_and_score, threshold_at_fpr, tpr_at_fpr


def test_tpr_at_fpr_controls_false_positives():
    rng = np.random.default_rng(0)
    human, ai = rng.normal(0, 1, 1000), rng.normal(2, 1, 1000)
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    s = np.r_[human, ai]
    t = threshold_at_fpr(human, 0.05)
    assert np.mean(human > t) <= 0.05
    assert tpr_at_fpr(y, s, 0.05) == pytest.approx(np.mean(ai > t))
    assert 0.6 < tpr_at_fpr(y, s, 0.05) < 0.8   # N(2,1) vs N(0,1) at 5% FPR: ~0.64


def test_fpr_threshold_optimization_scores_held_out_rows():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    p = 1 / (1 + np.exp(-np.r_[rng.normal(-1, 1, 500), rng.normal(1, 1, 500)]))
    _, m = apply_threshold_and_score(y, (p >= .5).astype(int), p, optimize_threshold="fpr0.05")
    assert m["n_scored"] == 800 and m["fpr"] < 0.1 and "tpr_at_0.01fpr" in m


def test_row_selection_matches_the_historical_scripts():
    """select_eval_part and subsample must pick the rows the pre-refactor scripts picked."""
    from sklearn.model_selection import train_test_split
    labels = np.array([0, 1] * 50)
    texts = [f"t{i}" for i in range(100)]
    a, b = train_test_split(np.arange(100), test_size=0.5, random_state=42, stratify=labels)
    t_sel, _ = select_eval_part(texts, labels, "select")
    assert t_sel == [texts[i] for i in np.sort(a)]
    rng = np.random.default_rng(7)   # the historical stratified subsample
    idx = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:10] for c in np.unique(labels)])
    idx = rng.permutation(idx)
    t_sub, _ = subsample(texts, labels, 20, True, 7)
    assert t_sub == [texts[i] for i in idx]


def test_labels_from_column_conventions():
    assert labels_from_column(pd.Series([0, 1, 1])).tolist() == [0, 1, 1]
    assert labels_from_column(pd.Series(["human", "gpt4", "Human"])).tolist() == [0, 1, 0]
    assert labels_from_column(pd.Series([1, 0]), human_values=[1]).tolist() == [0, 1]   # MAGE: 1 = human


def test_load_table_and_resolve(tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame({"answer": ["a", None, "c"], "is_cheating": [0, 1, 1], "domain": ["d1", "d2", "d2"]}).to_csv(f, index=False)
    ts = load_table(str(f), meta_columns=["domain"])
    assert ts.texts == ["a", " ", "c"] and ts.labels.tolist() == [0, 1, 1] and "domain" in ts.meta
    assert resolve(f"mercor_ai:{f}").labels.tolist() == [0, 1, 1]
    assert resolve(f"other:{f}").name == "other"
    small = prepare(TextSet("s", ["a"] * 10, [0] * 5 + [1] * 5), n_rows=4, stratified=True)
    assert np.bincount(small.labels).tolist() == [2, 2]


def test_truncation_matches_the_page():
    ids, body = [2, 10, 11, 12, 13, 1], [10, 11, 12, 13]
    assert truncate_keep_specials(ids, body, 4) == [2, 10, 11, 1]
    assert truncate_keep_specials([10, 11, 12, 1], [10, 11, 12], 3) == [10, 11, 1]   # "$A <eos>" template
    assert truncate_keep_specials(ids, body, None) == ids


def test_scores_from_logits():
    s = Scores.from_logits([0.0, 2.0])
    assert s.p_ai[0] == pytest.approx(0.5) and s.score.tolist() == [0.0, 2.0]


def test_weight_only_int8_quantization(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    from textdet.export.quantize import quantize_weights_int8
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 1, (2048, 64)).astype(np.float32)
    w = rng.normal(0, 0.1, (64, 128)).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Gather", ["emb", "ids"], ["h"], axis=0), helper.make_node("MatMul", ["h", "w"], ["y"])],
        "g", [helper.make_tensor_value_info("ids", TensorProto.INT64, ["b", "t"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["b", "t", 128])],
        [numpy_helper.from_array(emb, "emb"), numpy_helper.from_array(w, "w")])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    src, dst = tmp_path / "f.onnx", tmp_path / "q.onnx"
    onnx.save(model, str(src))
    counts = quantize_weights_int8(src, dst, block=32)
    assert counts == {"matmul": 1, "gather": 1}
    ids = rng.integers(0, 2048, (2, 5)).astype(np.int64)
    ref = ort.InferenceSession(str(src)).run(None, {"ids": ids})[0]
    out = ort.InferenceSession(str(dst)).run(None, {"ids": ids})[0]
    assert np.abs(out - ref).max() < 0.02 * np.abs(ref).max()
    assert dst.stat().st_size < 0.4 * src.stat().st_size
