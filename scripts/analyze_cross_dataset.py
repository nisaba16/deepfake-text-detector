#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"""^cross_dataset_summary_
        (?P<train_ds>human_ai|mercor_ai|[^_]+)_
        (?P<model>.+?)_
        (?P<analysis>embedding|tfidf|perplexity|phd|jev|lmstat|hfclf|webmodel|ensemble)
        (?:_layer(?P<layer>-?\d+))?
        (?:_(?P<pooling>mean_std|mean|last|statistical))?
        (?:_(?P<norm>l2norm))?
        _(?P<classifier>[^.]+)\.csv$
    """,
    re.VERBOSE,
)

SPECIALIZED_EMBEDDING_RE = re.compile(
    r"""^cross_dataset_summary_
        (.+?)                           # Everything before _specialized_embedding_
        _specialized_embedding_
        ([^.]+)\.csv$                   # classifier before .csv
    """,
    re.VERBOSE,
)

# Possible metric column names to look for
METRIC_COLUMNS = {
    "f1": ["f1", "f1_score", "F1", "f1_binary", "f1_macro"],
    "roc_auc": ["roc_auc", "roc_auc_score", "auc", "AUC"],
    "accuracy": ["accuracy", "acc", "Accuracy"],
    "precision": ["precision", "Precision"],
    "recall": ["recall", "Recall"],
}

# Possible dataset column names to identify target rows
DATASET_COLS = ["eval_dataset", "dataset", "target_dataset", "name"]

def parse_filename(path: Path) -> Optional[Dict[str, Any]]:
    # Try specialized embedding format first
    m = SPECIALIZED_EMBEDDING_RE.match(path.name)
    if m:
        prefix, classifier = m.groups()
        # prefix format: train_ds_model_name
        # Known train datasets: human_ai, mercor_ai
        # Strategy: match known datasets at the start, rest is model name
        
        known_datasets = ["human_ai", "mercor_ai"]
        train_ds = None
        model = None
        
        for ds in known_datasets:
            if prefix.startswith(ds + "_"):
                train_ds = ds
                model = prefix[len(ds) + 1:]  # Everything after "human_ai_"
                break
        
        if train_ds is None or model is None:
            # Fallback: assume first part before first underscore is train_ds
            parts = prefix.split('_', 1)
            train_ds = parts[0] if len(parts) > 0 else "unknown"
            model = parts[1] if len(parts) > 1 else prefix
        
        out = {
            "file": str(path),
            "train_dataset": train_ds,
            "model_name": model,
            "analysis_type": "embedding",
            "layer": None,
            "pooling": None,
            "normalized": False,
            "classifier": classifier,
            "is_specialized": True,
        }
        size_match = re.search(r"(\d+(?:\.\d+)?)\s*([KMG]?B)", out["model_name"], re.IGNORECASE)
        out["model_size"] = size_match.group(0) if size_match else None
        return out
    
    # Try regular embedding format
    m = FILENAME_RE.match(path.name)
    if m:
        return {
            "file": str(path),
            "train_dataset": m.group("train_ds"),
            "model_name": m.group("model"),
            "analysis_type": m.group("analysis"),
            "layer": int(m.group("layer")) if m.group("layer") else None,
            "pooling": m.group("pooling"),
            "normalized": m.group("norm") is not None,
            "classifier": m.group("classifier"),
            "is_specialized": False,
        }
    
    return None


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def pick_metric_value(df: pd.DataFrame, metric: str, train_ds: Optional[str]) -> Tuple[float, Optional[str]]:
    # pick metric column
    metric_col = find_column(df, METRIC_COLUMNS.get(metric, [metric]))
    if metric_col is None:
        # last resort: take first matching numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return np.nan, None
        metric_col = num_cols[0]

    # if there are multiple rows (per target dataset), try to prefer mercor row or non-train dataset
    idx = None
    ds_col = find_column(df, DATASET_COLS)

    if ds_col is not None:
        ds_series = df[ds_col].astype(str).str.lower()
        # Prefer mercor if present
        merc_idx = df[ds_series.str.contains("mercor", na=False)].index
        if len(merc_idx) > 0:
            idx = merc_idx[0]
        elif train_ds:
            # pick any row with dataset != train dataset
            non_train = df[~ds_series.str.contains(str(train_ds).lower(), na=False)]
            if len(non_train) > 0:
                idx = non_train.index[0]

    if idx is not None:
        return float(df.loc[idx, metric_col]), str(df.loc[idx, ds_col]) if ds_col else None

    # Fallback: average across rows
    return float(df[metric_col].mean()), None


BINARY_FAMILY = {"svm", "lr", "logreg", "neural", "mlp", "xgb", "xgboost"}
OUTLIER_FAMILY = {"ocsvm", "oneclasssvm", "elliptic", "elliptic_envelope", "iforest", "isoforest"}

ANALYSIS_FAMILY = {"jev": "zeroshot", "lmstat": "zeroshot",
                   "hfclf": "pretrained", "webmodel": "pretrained", "ensemble": "pretrained"}


def classifier_family(name: str) -> str:
    n = name.lower()
    if n in BINARY_FAMILY:
        return "binary"
    if n in OUTLIER_FAMILY:
        return "outlier"
    # default: assume binary for unknown supervised classifiers
    return "binary"

# ...existing code...

def load_and_flatten(input_dir: Path, metric: str, filter_model: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(input_dir.glob("cross_dataset_summary_*.csv")):
        meta = parse_filename(csv_path)
        if not meta:
            continue
        if filter_model and filter_model.lower() not in meta["model_name"].lower():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        metric_value, chosen_target = pick_metric_value(df, metric, meta.get("train_dataset"))
        # Detectors we did not train are ranked apart: zero-shot (Jev-like, likelihood statistics) and
        # pretrained (off-the-shelf classifiers, browser models, ensembles of them)
        fam = ANALYSIS_FAMILY.get(meta["analysis_type"]) or classifier_family(meta["classifier"])
        row = {
            **meta,
            "metric": metric,
            "metric_value": metric_value,
            "chosen_target_dataset": chosen_target,
            "source_file": csv_path.name,
            "family": fam,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    res["model_key"] = res["model_name"]
    res["config"] = res.apply(
        lambda r: (
            f"specialized_{r['classifier']}"
            if r.get("is_specialized", False)
            else (
                f"layer{r['layer']}_{r['pooling']}_{'l2' if r['normalized'] else 'noL2'}_{r['classifier']}"
                if r["analysis_type"] == "embedding"
                else f"{r['analysis_type']}_{r['classifier']}"
            )
        ),
        axis=1,
    )
    return res

def top_and_worst_per_model(df: pd.DataFrame, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    tops, worsts = [], []
    for model_name, g in df.groupby("model_key"):
        g_sorted = g.sort_values("metric_value", ascending=False)
        tops.append(g_sorted.head(top_k).copy())
        worsts.append(g_sorted.tail(top_k).copy())
    return (pd.concat(tops, ignore_index=True) if tops else pd.DataFrame(),
            pd.concat(worsts, ignore_index=True) if worsts else pd.DataFrame())

def main():
    parser = argparse.ArgumentParser(description="Analyze cross-dataset summaries and rank configs per model.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="roc_auc",
                        help="Ranking metric (roc_auc does not depend on the decision threshold)")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--filter_model", type=str, default=None)
    parser.add_argument("--family", type=str, default="all", choices=["all", "binary", "outlier", "zeroshot", "pretrained"],
                        help="Filter by family: binary / outlier (trained here), zeroshot (Jev-like, "
                             "Fast-DetectGPT, Binoculars), pretrained (off-the-shelf classifiers).")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    flat = load_and_flatten(input_dir, args.metric, args.filter_model)
    if flat.empty:
        print("No results found.")
        return

    # Optional filter by family
    if args.family != "all":
        flat = flat[flat["family"] == args.family].copy()

    # Overall top/worst for selected family
    top_df, worst_df = top_and_worst_per_model(flat, args.top_k)

    # Save overall
    flat_path = output_dir / f"all_results_flat_{args.metric}.csv"
    top_path = output_dir / f"per_model_top{args.top_k}_{args.metric}{'' if args.family=='all' else '_' + args.family}.csv"
    worst_path = output_dir / f"per_model_worst{args.top_k}_{args.metric}{'' if args.family=='all' else '_' + args.family}.csv"
    flat.to_csv(flat_path, index=False)
    top_df.to_csv(top_path, index=False)
    worst_df.to_csv(worst_path, index=False)
    print(f"Saved: {flat_path}\nSaved: {top_path}\nSaved: {worst_path}")

    # If all, also emit per-family splits
    if args.family == "all":
        for fam in ["binary", "outlier", "zeroshot", "pretrained"]:
            fam_df = flat[flat["family"] == fam].copy()
            if fam_df.empty:
                continue
            fam_top, fam_worst = top_and_worst_per_model(fam_df, args.top_k)
            fam_top.to_csv(output_dir / f"per_model_top{args.top_k}_{args.metric}_{fam}.csv", index=False)
            fam_worst.to_csv(output_dir / f"per_model_worst{args.top_k}_{args.metric}_{fam}.csv", index=False)

    # Console preview
    print("\nPreview (per model):")
    for (model_name, fam), g in top_df.groupby(["model_key", "family"]):
        best = g.sort_values('metric_value', ascending=False).head(3)
        print(f"- {model_name} [{fam}] top-3 {args.metric}:")
        for _, r in best.iterrows():
            print(f"    {r['metric_value']:.4f}  |  {r['config']}  |  {r['source_file']}")

if __name__ == "__main__":
    main()