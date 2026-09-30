#!/usr/bin/env python3
"""
Evaluate any detectors on any datasets with the same rows, metrics and result files.

Detectors (--detectors, see textdet/detectors/registry.py):
    julia | laya[:<repo>[/<subfolder>]]      Jev-like zero-shot decision models
    hf:<repo or short name>                  off-the-shelf classifiers (short names: bert-tiny-raid, e5-small-raid,
                                             tmr-roberta-raid, modernbert-raid-mage)
    web:<folder>                             a browser model folder, exactly as the page runs it
    saved:<detector.pkl>                     the project's saved detectors
    fastdetectgpt[:<model>] | binoculars[:<observer>,<performer>]
    ens:<spec>+<spec>                        mean of standardized scores

Datasets (--datasets, see textdet/data/catalog.py): mercor_ai:data/mercor-ai/train.csv, mage_ood_gpt4,
mage_ood_gpt4_para, mage_test, raid_sample, or name:any/table.{csv,jsonl,parquet}.

Example (quick, CPU):
    python scripts/evaluate_detectors.py --detectors julia hf:e5-small-raid \\
        --datasets mage_ood_gpt4 mage_ood_gpt4_para --n_rows 400 --stratified_sample \\
        --save_summary --output_dir evaluation_results/cross_dataset
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textdet.data import prepare, resolve
from textdet.detectors.decision import QUESTION_PRESETS, build_questions
from textdet.detectors.registry import build_detector
from textdet.evaluation import THRESHOLD_METRICS, evaluate_detector


def add_dataset_args(parser: argparse.ArgumentParser):
    parser.add_argument("--datasets", nargs='+', required=True,
                        help="Dataset specs: catalog names (mage_ood_gpt4, raid_sample, ...) or name:path")
    parser.add_argument("--text_column", type=str, default=None, help="Text column (generic tables)")
    parser.add_argument("--label_column", type=str, default=None, help="Label column (generic tables)")
    parser.add_argument("--human_values", nargs='*', default=None,
                        help="Label values that mean human (default: 0/False/'human'; MAGE uses 1)")
    parser.add_argument("--eval_part", type=str, default="all", choices=["all", "select", "test"],
                        help="Score the whole set, or a fixed stratified part: 'select' to choose configs, "
                             "'test' only for the final number")
    parser.add_argument("--eval_test_frac", type=float, default=0.5, help="Fraction of each set in the 'test' part")
    parser.add_argument("--n_rows", type=int, default=None, help="Score a random subset of N rows per dataset")
    parser.add_argument("--stratified_sample", action="store_true", help="With --n_rows, balance the classes")
    parser.add_argument("--balance_column", type=str, default=None,
                        help="With --n_rows --stratified_sample, also balance this metadata column (e.g. model)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed for --n_rows sampling and the threshold optimization split")


def add_scoring_args(parser: argparse.ArgumentParser):
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold on P(AI) (default 0.5)")
    parser.add_argument("--optimize_threshold", type=str, default=None, choices=THRESHOLD_METRICS,
                        help="Tune the threshold on a split of each set: f1, or fpr0.01 / fpr0.05 (a target "
                             "false-positive rate on its human texts); metrics then cover the other rows")
    parser.add_argument("--optimize_split", type=float, default=0.2, help="Fraction used to tune the threshold")
    parser.add_argument("--save_summary", action="store_true", help="Save one summary CSV per detector head")
    parser.add_argument("--save_predictions", action="store_true", help="Save per-text scores")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")


def add_question_args(parser: argparse.ArgumentParser):
    parser.add_argument("--questions", nargs='*', default=["ai_generated"], choices=sorted(QUESTION_PRESETS),
                        help="Jev-like detectors: question presets; with several, their mean is scored too")
    parser.add_argument("--instructions", type=str, default=None,
                        help="Custom noul question whose 'true' answer means AI-generated (added as 'custom')")
    parser.add_argument("--true_criterion", type=str, default=None, help="Description of the custom 'true' option")
    parser.add_argument("--false_criterion", type=str, default=None, help="Description of the custom 'false' option")


def load_datasets(args):
    datasets = []
    for spec in args.datasets:
        ts = resolve(spec, text_column=args.text_column, label_column=args.label_column,
                     human_values=args.human_values, seed=args.random_state)
        ts = prepare(ts, part=args.eval_part, test_frac=args.eval_test_frac, n_rows=args.n_rows,
                     stratified=args.stratified_sample, seed=args.random_state, balance_column=args.balance_column)
        print(f"Loaded {ts.describe()}")
        datasets.append(ts)
    return datasets


def run(args, detector_specs: List[str]) -> Dict[str, Dict]:
    datasets = load_datasets(args)
    questions = build_questions(args.questions, args.instructions, args.true_criterion, args.false_criterion)
    all_results = {}
    for spec in detector_specs:
        print(f"\n{'#' * 70}\n# Detector {spec}\n{'#' * 70}")
        detector = build_detector(spec, device=args.device, batch_size=args.batch_size,
                                  max_length=args.max_length, questions=questions, chunk_stride=args.chunk_stride)
        results = evaluate_detector(detector, datasets, threshold=args.threshold,
                                    optimize_threshold=args.optimize_threshold, optimize_split=args.optimize_split,
                                    random_state=args.random_state, eval_part=args.eval_part,
                                    output_dir=args.output_dir, save_summary=args.save_summary,
                                    save_predictions=args.save_predictions)
        for head, by_ds in results.items():
            all_results[spec if head == "default" else f"{spec} [{head}]"] = by_ds
        detector.close()
        del detector
        gc.collect()
    print_comparison(all_results)
    return all_results


def print_comparison(all_results: Dict[str, Dict]):
    rows = []
    for name, by_ds in all_results.items():
        for ds, m in by_ds.items():
            rows.append({"detector": name, "dataset": ds, "AUROC": m.get("roc_auc"),
                         "TPR@1%FPR": m.get("tpr_at_0.01fpr"), "TPR@5%FPR": m.get("tpr_at_0.05fpr"),
                         "acc@thr": m.get("accuracy"), "FPR@thr": m.get("fpr"), "s/text": m.get("seconds_per_text")})
    if rows:
        print("\n" + "=" * 100 + "\nCOMPARISON\n" + "=" * 100)
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(pd.DataFrame(rows).round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate detectors on datasets (same rows, same metrics)")
    parser.add_argument("--detectors", nargs='+', required=True, help="Detector specs (see module docstring)")
    add_dataset_args(parser)
    add_scoring_args(parser)
    add_question_args(parser)
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=None, help="Texts per forward pass")
    parser.add_argument("--max_length", type=int, default=None, help="Token budget per text (detector default)")
    parser.add_argument("--chunk_stride", type=int, default=None,
                        help="hf: classifiers: score every window of max_length tokens (this overlap) and "
                             "average, instead of the first window only")
    args = parser.parse_args()
    run(args, args.detectors)


if __name__ == "__main__":
    main()
