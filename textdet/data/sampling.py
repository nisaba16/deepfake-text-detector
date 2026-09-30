"""
Row selection shared by every evaluation, so all detectors are scored on the same rows.

select_eval_part: a fixed stratified split of an evaluation set into a 'select' part (choose configs, tune
thresholds) and a 'test' part (the final number only). Same seed and code as before the refactor, so the rows
of earlier result files are unchanged.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .loading import TextSet

EVAL_SPLIT_SEED = 42  # fixed, so the 'select' and 'test' parts never mix across runs


def select_eval_part(texts: List[str], labels: np.ndarray, part: str = "all",
                     test_frac: float = 0.5) -> Tuple[List[str], np.ndarray]:
    """Keep the whole evaluation set, or one part of a fixed stratified split of it.

    'select' rows are for choosing configs (sweeps, threshold tuning); 'test' rows are only scored for
    the final number, which picking the best of many configs on the same rows would inflate.
    """
    keep = eval_part_indices(labels, part, test_frac)
    return [texts[i] for i in keep], np.asarray(labels)[keep]


def eval_part_indices(labels: np.ndarray, part: str = "all", test_frac: float = 0.5) -> np.ndarray:
    labels = np.asarray(labels)
    if part == "all":
        return np.arange(len(labels))
    if part not in ("select", "test"):
        raise ValueError(f"part must be all, select or test, not {part!r}")
    stratify = labels if len(np.unique(labels)) > 1 else None
    idx_select, idx_test = train_test_split(np.arange(len(labels)), test_size=test_frac,
                                            random_state=EVAL_SPLIT_SEED, stratify=stratify)
    return np.sort(idx_select if part == "select" else idx_test)


def subsample_indices(labels: np.ndarray, n_rows: int, stratified: bool = False, seed: int = 42,
                      groups: Optional[Sequence] = None) -> np.ndarray:
    """Random subset of n_rows (for quick runs).

    stratified: the same number of rows per class (n_rows // n_classes each).
    groups: with stratified, balance over (class, group) cells instead, e.g. the generator column of a
        benchmark, so a small sample still covers every generator.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    if not stratified:
        return rng.permutation(len(labels))[:n_rows]
    keys = labels.astype(str) if groups is None else np.char.add(labels.astype(str), "|" + np.asarray(groups).astype(str))
    cells = np.unique(keys)
    per_cell = max(1, n_rows // len(cells))
    idx = np.concatenate([rng.permutation(np.flatnonzero(keys == c))[:per_cell] for c in cells])
    return rng.permutation(idx)


def subsample(texts: List[str], labels: np.ndarray, n_rows: int, stratified: bool, seed: int):
    """Random subset of n_rows for quick runs (n_rows // n_classes per class with stratified=True)."""
    idx = subsample_indices(labels, n_rows, stratified, seed)
    return [texts[i] for i in idx], np.asarray(labels)[idx]


def prepare(ts: TextSet, *, part: str = "all", test_frac: float = 0.5, n_rows: Optional[int] = None,
            stratified: bool = False, seed: int = 42, balance_column: Optional[str] = None) -> TextSet:
    """The evaluation part of a TextSet, then an optional subsample of it (the order of the old scripts)."""
    ts = ts.subset(eval_part_indices(ts.labels, part, test_frac))
    if n_rows and n_rows < len(ts):
        groups = ts.meta[balance_column].to_numpy() if balance_column and balance_column in ts.meta else None
        ts = ts.subset(subsample_indices(ts.labels, n_rows, stratified, seed, groups))
    return ts
