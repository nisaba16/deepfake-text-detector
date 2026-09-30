"""
Metrics shared by every detector, so trained, zero-shot and off-the-shelf detectors are scored the same way.

Besides accuracy / F1 / ROC-AUC, the true-positive rate at a fixed false-positive rate (TPR@1%FPR,
TPR@5%FPR) is reported: it is RAID's headline metric and what matters when flagging a human is costly.
ROC-AUC can stay high while a detector flags many humans of an unseen domain with near-certainty.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

FPR_TARGETS = (0.01, 0.05)
THRESHOLD_METRICS = ("f1", "fpr0.01", "fpr0.05")


def threshold_at_fpr(human_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold t such that at most `fpr` of the human scores are > t (predict AI when score > t)."""
    s = np.sort(np.asarray(human_scores, dtype=float))[::-1]
    if len(s) == 0:
        return float("nan")
    k = int(np.floor(fpr * len(s)))          # number of humans allowed above the threshold
    return float(s[min(k, len(s) - 1)])


def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """TPR when the threshold is set on the human rows for a false-positive rate of at most `fpr`."""
    y_true, score = np.asarray(y_true), np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    t = threshold_at_fpr(score[y_true == 0], fpr)
    return float(np.mean(score[y_true == 1] > t))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Thresholded metrics, plus ROC-AUC and TPR at 1% / 5% FPR when a ranking score is given."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    y_true_arr = np.asarray(y_true)
    if len(y_true_arr):
        human = y_true_arr == 0
        # The two error rates separately: which side a detector errs on matters more than accuracy
        metrics['fpr'] = float(np.mean(np.asarray(y_pred)[human] == 1)) if human.any() else float('nan')
        metrics['fnr'] = float(np.mean(np.asarray(y_pred)[~human] == 0)) if (~human).any() else float('nan')

    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = float('nan')
        for fpr in FPR_TARGETS:
            metrics[f'tpr_at_{fpr:g}fpr'] = tpr_at_fpr(y_true, y_proba, fpr)

    return metrics


def _sweep_thresholds(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """Return (best_threshold, metric value on these rows).

    f1:        the threshold in [0, 1] (steps of 0.01) with the best F1
    fpr<rate>: the threshold that gives at most that false-positive rate on these rows' human texts
               (needs no AI examples: a target domain's human texts are enough to calibrate)
    """
    if y_proba is None:
        return 0.5, float('nan')

    if metric.startswith("fpr"):
        rate = float(metric[3:])
        t = threshold_at_fpr(y_proba[y_true == 0], rate)
        return float(t), float(np.mean(y_proba[y_true == 1] > t)) if (y_true == 1).any() else float('nan')

    thresholds = np.linspace(0.0, 1.0, 101)
    best_t = 0.5
    best_val = -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        val = f1_score(y_true, pred, average='binary', zero_division=0)
        if val > best_val:
            best_val = val
            best_t = t
    return float(best_t), float(best_val)


def apply_threshold_and_score(labels: np.ndarray, predictions: np.ndarray, prob_fake: np.ndarray | None,
                              threshold: float | None = None,
                              optimize_threshold: str | None = None,
                              optimize_split: float = 0.2,
                              random_state: int = 42,
                              ranking_score: np.ndarray | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the manual or optimized threshold on P(fake), then compute the metrics.

    Shared by the saved detectors and the zero-shot baselines so all are scored the same way.
    With optimize_threshold, the metrics cover only the rows the threshold was not tuned on.
    ranking_score (same order as P(fake)) replaces P(fake) for ROC-AUC and TPR@FPR when given.
    Returns (predictions for every row, metrics).
    """
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    applied_threshold = None
    optimized_info: Dict[str, Any] = {}
    scored = np.arange(len(labels))  # rows the metrics are computed on
    # fprX thresholds compare with "score > t" (ties stay human), f1 thresholds with "score >= t"
    strict = bool(optimize_threshold and optimize_threshold.startswith("fpr"))

    # Apply manual or optimized thresholding if probabilities are available
    if prob_fake is not None:
        if optimize_threshold is not None:
            # A small validation split to find the threshold, then apply it to the other rows
            try:
                X_idx = np.arange(len(labels))
                # ensure stratify only when both classes exist
                stratify = labels if len(np.unique(labels)) > 1 else None
                idx_train, idx_val = train_test_split(
                    X_idx, test_size=optimize_split, random_state=random_state, stratify=stratify
                ) if 0 < optimize_split < 1.0 and len(labels) > 10 else (X_idx, [])
                if len(idx_val) > 0:
                    best_t, best_val = _sweep_thresholds(labels[idx_val], prob_fake[idx_val], metric=optimize_threshold)
                    # Scoring the tuning rows too would inflate every thresholded metric
                    scored = np.sort(idx_train)
                else:
                    best_t, best_val = _sweep_thresholds(labels, prob_fake, metric=optimize_threshold)
                    print("⚠️  Threshold tuned on the rows it is scored on (no validation split): optimistic metrics")
                applied_threshold = best_t
                predictions = ((prob_fake > applied_threshold) if strict else (prob_fake >= applied_threshold)).astype(int)
                optimized_info = {
                    'optimize_metric': optimize_threshold,
                    'optimize_split': float(optimize_split),
                    'chosen_threshold': float(best_t),
                    'chosen_metric_value': float(best_val),
                }
            except Exception as e:
                # Fallback: ignore optimization on error
                print(f"⚠️  Threshold optimization failed ({e}): keeping the default threshold")

        if threshold is not None and applied_threshold is None:
            applied_threshold = float(threshold)
            predictions = (prob_fake >= applied_threshold).astype(int)

    # Compute metrics
    auc_input = ranking_score if ranking_score is not None else prob_fake
    metrics = compute_metrics(labels[scored], predictions[scored],
                              None if auc_input is None else np.asarray(auc_input)[scored])
    metrics['n_scored'] = int(len(scored))
    if applied_threshold is not None:
        metrics['applied_threshold'] = float(applied_threshold)
    if optimized_info:
        metrics.update(optimized_info)
    return predictions, metrics
