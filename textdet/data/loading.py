"""
Load any labelled text table as a TextSet: texts, labels (0 = human, 1 = AI) and the other columns as metadata.

Sources:
    data/x.csv, .tsv, .jsonl, .json, .parquet (optionally .gz)     local files (.zip: one CSV inside)
    hf://datasets/<org>/<name>/<file>                               one file of a Hugging Face dataset repo
    hf:<org>/<name>[:<config>][@<split>]                            a Hugging Face dataset (needs `datasets`)

Every source uses its own label convention (MAGE: 1 = human, RAID: a `model` column that is "human" for human
text), so `human_values` says which label values are human; every other value is AI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

TEXT_COLUMNS = ["text", "answer", "content", "passage", "generation", "document"]
LABEL_COLUMNS = ["label", "is_cheating", "generated", "target", "is_ai", "is_machine"]


@dataclass
class TextSet:
    """Texts with binary labels (0 = human, 1 = AI) and per-row metadata (generator, domain, attack, ...)."""
    name: str
    texts: List[str]
    labels: np.ndarray
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=int)
        if len(self.texts) != len(self.labels):
            raise ValueError(f"{self.name}: {len(self.texts)} texts but {len(self.labels)} labels")
        if len(self.meta) and len(self.meta) != len(self.texts):
            raise ValueError(f"{self.name}: metadata has {len(self.meta)} rows for {len(self.texts)} texts")
        if not len(self.meta):
            self.meta = pd.DataFrame(index=range(len(self.texts)))

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, idx: Iterable[int], name: Optional[str] = None) -> "TextSet":
        idx = np.asarray(list(idx), dtype=int)
        return TextSet(name or self.name, [self.texts[i] for i in idx], self.labels[idx],
                       self.meta.iloc[idx].reset_index(drop=True))

    def describe(self) -> str:
        counts = np.bincount(self.labels, minlength=2)
        return f"{self.name}: {len(self)} texts ({counts[0]} human, {counts[1]} AI)"


def sanitize_texts(texts: Sequence) -> List[str]:
    """Every text as a non-empty string accepted by tokenizers (missing or blank texts become a space)."""
    return [t.strip() if isinstance(t, str) and t.strip() else " " for t in texts]


def read_table(source: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a local file, a hf://datasets/... file or a hf:<org>/<name>[:config][@split] dataset."""
    if source.startswith("hf:") and not source.startswith("hf://"):
        return _read_hf_dataset(source[3:], columns)
    path = source[:-3] if source.endswith(".gz") else source
    suffix = Path(path).suffix.lower()
    if suffix in (".csv", ".zip"):   # a .zip holding one CSV (pandas decompresses it)
        return pd.read_csv(source, usecols=columns)
    if suffix == ".tsv":
        return pd.read_csv(source, sep="\t", usecols=columns)
    if suffix in (".jsonl", ".json"):
        df = pd.read_json(source, lines=suffix == ".jsonl")
        return df[list(columns)] if columns else df
    if suffix == ".parquet":
        return pd.read_parquet(source, columns=list(columns) if columns else None)
    raise ValueError(f"{source}: unsupported file type (use .csv, .tsv, .jsonl, .json or .parquet)")


def read_csv_sampled(source: str, group_columns: Sequence[str], per_group: int, *, seed: int = 42,
                     columns: Optional[Sequence[str]] = None, filters: Optional[dict] = None,
                     chunksize: int = 200_000) -> pd.DataFrame:
    """A uniform random sample of at most per_group rows per group, streamed from a CSV too big for memory.

    Each row gets a seeded random key and every group keeps its per_group smallest keys, so the sample
    does not depend on the chunk size or the file order (RAID's train.csv is 11 GB, grouped by source).
    """
    rng = np.random.default_rng(seed)
    kept: Optional[pd.DataFrame] = None
    for chunk in pd.read_csv(source, usecols=columns, chunksize=chunksize):
        chunk = _apply_filters(chunk, filters)
        chunk = chunk.assign(_key=rng.random(len(chunk)))
        merged = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        kept = (merged.sort_values("_key")
                .groupby(list(group_columns), dropna=False, sort=False).head(per_group))
    if kept is None:
        return pd.DataFrame(columns=columns)
    return kept.sort_values("_key").drop(columns="_key").reset_index(drop=True)


def _apply_filters(df: pd.DataFrame, filters: Optional[dict]) -> pd.DataFrame:
    for column, allowed in (filters or {}).items():
        allowed = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        df = df[df[column].astype(str).isin({str(a) for a in allowed})]
    return df


def _read_hf_dataset(spec: str, columns: Optional[Sequence[str]]) -> pd.DataFrame:
    try:
        import datasets
    except ImportError as e:
        raise ImportError("hf:<org>/<name> sources need `pip install datasets` "
                          "(or point to one file: hf://datasets/<org>/<name>/<file>.parquet)") from e
    spec, _, split = spec.partition("@")
    repo, _, config = spec.partition(":")
    ds = datasets.load_dataset(repo, config or None, split=split or "train")
    if columns:
        ds = ds.select_columns(list(columns))
    return ds.to_pandas()


def infer_column(df: pd.DataFrame, candidates: Sequence[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer the {what} column from {list(df.columns)}: pass it explicitly")


def labels_from_column(values: pd.Series, human_values: Optional[Sequence] = None) -> np.ndarray:
    """0 for human rows, 1 for AI rows.

    human_values: the values that mean human. Default: numeric/bool labels are used as is (1 or True = AI),
    string labels are human when they read "human" (case-insensitive), AI otherwise.
    """
    if human_values is not None:
        human = {str(v).lower() for v in human_values}
        return (~values.astype(str).str.lower().isin(human)).to_numpy(dtype=int)
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        labels = values.astype(int).to_numpy()
        if not set(np.unique(labels)) <= {0, 1}:
            raise ValueError(f"Label values {sorted(np.unique(labels))} are not 0/1: pass human_values")
        return labels
    return (values.astype(str).str.lower() != "human").to_numpy(dtype=int)


def load_table(source: str, *, name: Optional[str] = None, text_column: Optional[str] = None,
               label_column: Optional[str] = None, human_values: Optional[Sequence] = None,
               meta_columns: Sequence[str] = (), filters: Optional[dict] = None,
               sample_per_group: Optional[int] = None, group_columns: Sequence[str] = (),
               seed: int = 42, derive: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> TextSet:
    """Load a table as a TextSet.

    filters: {column: allowed value or list of values}, applied before labelling (e.g. {"attack": "none"}).
    meta_columns: columns kept as metadata for per-generator / per-domain breakdowns (missing ones are skipped).
    sample_per_group: stream a CSV and keep at most this many random rows per group_columns cell.
    derive: adds metadata columns computed from the raw table (e.g. generator and domain parsed from one column).
    """
    if sample_per_group:
        if not group_columns:
            raise ValueError("sample_per_group needs group_columns")
        df = read_csv_sampled(source, group_columns, sample_per_group, seed=seed, filters=filters)
    else:
        df = _apply_filters(read_table(source), filters)
    df = df.reset_index(drop=True)
    if derive is not None:
        df = derive(df)
    text_column = text_column or infer_column(df, TEXT_COLUMNS, "text")
    label_column = label_column or infer_column(df, LABEL_COLUMNS, "label")
    labels = labels_from_column(df[label_column], human_values)
    keep = [c for c in dict.fromkeys([label_column, *meta_columns]) if c in df.columns and c != text_column]
    return TextSet(name or Path(source).stem, sanitize_texts(df[text_column].tolist()), labels,
                   df[keep].reset_index(drop=True))
