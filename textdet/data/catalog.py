"""
Named datasets and the dataset spec strings every script accepts.

Spec strings (--datasets):
    mercor_ai:data/mercor-ai/train.csv       a catalog name with its local path (the historical form)
    mage_ood_gpt4                            a catalog name alone: its default source (downloaded if remote)
    raid_sample                              RAID train, a random sample per (generator, domain, attack) cell
    my_set:data/other.csv                    any other name: a generic table, columns inferred or given

See docs/TEXT_DETECTION_REVIEW.md for why these benchmarks and how to read their scores.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import pandas as pd

from .loading import TextSet, load_table

HF_MAGE = "hf://datasets/yaful/MAGE"
HF_RAID = "hf://datasets/liamdugan/raid"
# Local copies of hf://datasets/<org>/<name>/<file> files (scripts/prefetch_textdet.py), for nodes without
# internet and to avoid streaming RAID's 11 GB on every run
MIRROR = Path(os.environ.get("TEXTDET_DATA", "data/hf"))


def local_mirror(source: str) -> str:
    """MIRROR/<org>/<name>/<file> when that copy exists, else the source unchanged."""
    prefix = "hf://datasets/"
    if source.startswith(prefix):
        local = MIRROR / source[len(prefix):]
        if local.exists():
            return str(local)
    return source


def _mage_meta(df: pd.DataFrame) -> pd.DataFrame:
    """MAGE's `src` is '<domain>_human' or '<domain>_machine_<prompt>_<generator>'."""
    src = df["src"].astype(str)
    machine = src.str.split("_machine_", n=1)
    df = df.copy()
    df["domain"] = machine.str[0].str.replace("_human", "", regex=False)
    rest = machine.str[1].fillna("")
    df["prompt"] = rest.str.split("_", n=1).str[0].where(rest != "", "human")
    df["generator"] = rest.str.split("_", n=1).str[1].fillna("human").where(rest != "", "human")
    return df


@dataclass(frozen=True)
class DatasetSpec:
    """How to read one dataset as 0 = human / 1 = AI."""
    name: str
    source: str
    text_column: str
    label_column: str
    human_values: Optional[Sequence] = None      # label values meaning human (None: 0/False/"human")
    meta_columns: Sequence[str] = ()
    filters: Dict[str, object] = field(default_factory=dict)
    sample_per_group: Optional[int] = None        # stream the file, keep this many random rows per group
    group_columns: Sequence[str] = ()
    derive: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    breakdown: Optional[str] = None               # metadata column for per-group metrics
    description: str = ""

    def load(self, source: Optional[str] = None, seed: int = 42) -> TextSet:
        return load_table(source or local_mirror(self.source), name=self.name, text_column=self.text_column,
                          label_column=self.label_column, human_values=self.human_values,
                          meta_columns=self.meta_columns, filters=self.filters,
                          sample_per_group=self.sample_per_group, group_columns=self.group_columns,
                          seed=seed, derive=self.derive)


_RAID_META = ("model", "domain", "attack", "decoding", "repetition_penalty")

CATALOG: Dict[str, DatasetSpec] = {spec.name: spec for spec in [
    # --- the project's own datasets (local files) ---
    DatasetSpec("mercor_ai", "data/mercor-ai/train.csv", "answer", "is_cheating",
                description="Mercor AI challenge answers (is_cheating: 1 = AI)"),
    DatasetSpec("human_ai", "data/data_human/AI_Human.csv", "text", "generated",
                description="Kaggle AI_Human.csv essays (generated: 1 = AI)"),
    DatasetSpec("human_ai_sample", "data/data_human/AI_Human.csv", "text", "generated",
                sample_per_group=2000, group_columns=("generated",),
                description="AI_Human.csv streamed (1 GB, 487k rows): 2000 random essays per class. "
                            "The saved human_ai detectors were trained on this file: not a held-out test"),

    # --- MAGE (Li et al., ACL 2024): 27 generators, 10 domains; label 1 = HUMAN ---
    DatasetSpec("mage_test", f"{HF_MAGE}/test.csv", "text", "label", human_values=[1], meta_columns=("src",
                "domain", "generator", "prompt"), derive=_mage_meta, breakdown="domain",
                description="MAGE test split (in distribution for detectors trained on MAGE train, e.g. "
                            "hf:modernbert-raid-mage)"),
    DatasetSpec("mage_ood_gpt4", f"{HF_MAGE}/test_ood_set_gpt.csv", "text", "label", human_values=[1],
                meta_columns=("src",), description="MAGE wild set: unseen domains, GPT-4 generations"),
    DatasetSpec("mage_ood_gpt4_para", f"{HF_MAGE}/test_ood_set_gpt_para.csv", "text", "label", human_values=[1],
                meta_columns=("src",), description="MAGE wild set, GPT-4 texts paraphrased (attack)"),

    # --- RAID (Dugan et al., ACL 2024): 11 generators x 8 domains x 11 attacks; `model` is "human" for human ---
    DatasetSpec("raid_sample", f"{HF_RAID}/train.csv", "generation", "model", human_values=["human"],
                meta_columns=_RAID_META, sample_per_group=40, group_columns=("model", "domain", "attack"),
                breakdown="model",
                description="RAID train (labelled), 40 random rows per generator x domain x attack cell. "
                            "The hf:*-raid detectors were trained on RAID train: these can be their training rows"),
    DatasetSpec("raid_sample_clean", f"{HF_RAID}/train.csv", "generation", "model", human_values=["human"],
                meta_columns=_RAID_META, filters={"attack": "none"}, sample_per_group=150,
                group_columns=("model", "domain"), breakdown="model",
                description="RAID train without attacks, 150 random rows per generator x domain cell"),
]}


def resolve(spec: str, *, text_column: Optional[str] = None, label_column: Optional[str] = None,
            human_values: Optional[Sequence] = None, seed: int = 42) -> TextSet:
    """Load a dataset from a spec string ('name', 'name:path' or 'path'), see the module docstring.

    Catalog datasets always use their own columns and label convention (as the historical loaders did);
    text_column / label_column / human_values apply to the other tables.
    """
    name, sep, path = spec.partition(":")
    if sep and name in ("hf", "http", "https"):      # a bare URL or hf:org/name, not name:path
        name, path = "", spec
    if not sep and name not in CATALOG:               # a bare path
        name, path = "", spec
    if name in CATALOG:
        return CATALOG[name].load(path or None, seed=seed)
    if not path:
        raise ValueError(f"Unknown dataset {spec!r}: use one of {sorted(CATALOG)} or name:path")
    return load_table(path, name=name or None, text_column=text_column, label_column=label_column,
                      human_values=human_values)


def breakdown_column(ts: TextSet) -> Optional[str]:
    spec = CATALOG.get(ts.name)
    return spec.breakdown if spec and spec.breakdown in ts.meta else None


def load_dataset(dataset_name: str, data_path: str, text_col: str = None, label_col: str = None):
    """(texts, labels) the way scripts/load_and_evaluate.py always loaded them (legacy interface)."""
    spec = f"{dataset_name}:{data_path}" if dataset_name not in ("generic", "") else data_path
    ts = resolve(spec, text_column=text_col, label_column=label_col)
    return ts.texts, ts.labels
