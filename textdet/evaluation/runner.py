"""
Evaluate any detector on any datasets, with the same metrics and the same result files for all of them.

Files (in output_dir):
    cross_dataset_summary_<stem>.csv      one row per dataset (the format analyze_cross_dataset.py ranks)
    predictions_<stem>_<dataset>.csv      per text: label, metadata, P(AI) and ranking score per head
    breakdown_<stem>_<dataset>.csv        per generator / domain / attack when the dataset has that column
<stem> is <train_dataset>_<model tag>_<analysis>_<head> (e.g. zeroshot_SupersonicLabs_Julia-1_jev_ai_generated).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..data.catalog import breakdown_column
from ..data.loading import TextSet
from ..detectors.base import Detector, Scores
from .metrics import apply_threshold_and_score, threshold_at_fpr, tpr_at_fpr

TABLE_COLUMNS = [("accuracy", "Acc"), ("f1", "F1"), ("fpr", "FPR"), ("roc_auc", "AUROC"),
                 ("tpr_at_0.01fpr", "TPR@1%"), ("tpr_at_0.05fpr", "TPR@5%")]


def breakdown(labels: np.ndarray, score: np.ndarray, groups: Sequence, fpr: float = 0.05) -> pd.DataFrame:
    """Per-group metrics. Groups with both classes (domains): AUROC and TPR within the group.
    AI-only groups (generators, attacks): TPR at the threshold set on ALL human rows for `fpr`."""
    labels, score, groups = np.asarray(labels), np.asarray(score), np.asarray(groups).astype(str)
    global_t = threshold_at_fpr(score[labels == 0], fpr)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        y, s = labels[m], score[m]
        row: Dict[str, Any] = {"group": g, "n": int(m.sum()), "n_ai": int(y.sum())}
        if len(np.unique(y)) == 2:
            row["roc_auc"] = roc_auc_score(y, s)
            row[f"tpr_at_{fpr:g}fpr_in_group"] = tpr_at_fpr(y, s, fpr)
        if y.any():
            row[f"tpr_at_{fpr:g}fpr_global"] = float(np.mean(s[y == 1] > global_t))
        if (~y.astype(bool)).any():
            row["fpr_at_global_threshold"] = float(np.mean(s[y == 0] > global_t))
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(dataset: str, metrics_by_head: Dict[str, Dict[str, Any]]):
    print(f"\nResults on {dataset}:")
    print(f"{'Head':<22}" + "".join(f"{short:>9}" for _, short in TABLE_COLUMNS) + f"{'Thresh':>9}")
    print("-" * (22 + 9 * (len(TABLE_COLUMNS) + 1)))
    for head, m in metrics_by_head.items():
        cells = "".join(f"{m.get(key, float('nan')):>9.4f}" for key, _ in TABLE_COLUMNS)
        print(f"{head:<22}{cells}{m.get('applied_threshold', 0.5):>9.3f}")


def evaluate_detector(detector: Detector, datasets: List[TextSet], *, threshold: Optional[float] = None,
                      optimize_threshold: Optional[str] = None, optimize_split: float = 0.2,
                      random_state: int = 42, eval_part: str = "all", output_dir: Optional[str] = None,
                      save_summary: bool = False, save_predictions: bool = False,
                      save_breakdown: bool = True) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Score every dataset; returns {head: {dataset: metrics}} and writes the files asked for."""
    out_dir = Path(output_dir) if output_dir else None
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ts in datasets:
        print(f"\n{'=' * 60}\nEvaluating on {ts.describe()}\n{'=' * 60}")
        start = time.time()
        heads: Dict[str, Scores] = detector.score(ts.texts)
        per_text = (time.time() - start) / max(1, len(ts))

        for head, s in heads.items():
            _, metrics = apply_threshold_and_score(
                ts.labels, (s.p_ai >= 0.5).astype(int), s.p_ai, threshold=threshold,
                optimize_threshold=optimize_threshold, optimize_split=optimize_split,
                random_state=random_state, ranking_score=s.score)
            metrics["eval_part"] = eval_part
            metrics["seconds_per_text"] = per_text
            results.setdefault(head, {})[ts.name] = metrics

        print_table(ts.name, {h: results[h][ts.name] for h in heads})

        group_col = breakdown_column(ts)
        if out_dir and (save_predictions or (save_breakdown and group_col)):
            out_dir.mkdir(parents=True, exist_ok=True)
        if save_breakdown and group_col and out_dir:
            for head, s in heads.items():
                bd = breakdown(ts.labels, s.score, ts.meta[group_col])
                bd.insert(0, "column", group_col)
                bd.to_csv(out_dir / f"breakdown_{detector.summary_stem(head)}_{ts.name}.csv", index=False)
        if save_predictions and out_dir:
            pred = ts.meta.copy()
            pred.insert(0, "true_label", ts.labels)
            for head, s in heads.items():
                pred[f"probability_fake_{head}"] = s.p_ai
                pred[f"log_odds_fake_{head}"] = s.score
            path = out_dir / f"predictions_{detector.predictions_stem()}_{ts.name}.csv"
            pred.to_csv(path, index=False)
            print(f"Predictions saved to: {path}")

    if save_summary and out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        info = detector.describe()
        for head, by_dataset in results.items():
            summary = pd.DataFrame(by_dataset).T
            for k, v in info.items():
                summary[k] = v
            if hasattr(detector, "question_column"):
                summary["question"] = detector.question_column(head)
            path = out_dir / f"cross_dataset_summary_{detector.summary_stem(head)}.csv"
            summary.to_csv(path)
            print(f"Summary saved to: {path}")
    return results
