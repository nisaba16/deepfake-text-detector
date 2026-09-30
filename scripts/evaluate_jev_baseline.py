#!/usr/bin/env python3
"""
Jev-like zero-shot baseline: score datasets with a "System 1" decision model (no training) and report the
same metrics, with the same thresholding options, as cross_dataset_evaluation.py.

Backends: julia (Julia 1, mmBERT-small, native, default) or laya (Laya, ModernBERT-large, `pip install laya`),
kept for comparison. Datasets: any spec of textdet.data.catalog (mercor_ai:path, mage_ood_gpt4, raid_sample,
name:any/table.csv, ...), so the baseline runs on general datasets, not only the two historical ones.

Each question gets its own summary, cross_dataset_summary_zeroshot_<checkpoint>_jev_<question>.csv,
in the format cross_dataset_evaluation.py writes, so analyze_cross_dataset.py ranks the baseline next
to the trained detectors (family "zeroshot"). This is scripts/evaluate_detectors.py with one detector.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_detectors import add_dataset_args, add_question_args, add_scoring_args, run


def detector_spec(args) -> str:
    checkpoint = args.checkpoint or ""
    if args.backend == "laya" and args.subfolder:
        checkpoint = f"{checkpoint or 'convaiinnovations/laya'}/{args.subfolder}"
    return f"{args.backend}:{checkpoint}" if checkpoint else args.backend


def main():
    parser = argparse.ArgumentParser(description="Jev-like zero-shot baseline: decision model, no training")
    parser.add_argument("--backend", type=str, default="julia", choices=["julia", "laya"],
                        help="julia: Julia 1 (native, 144M, the model shipped to the browser); "
                             "laya: Laya (laya package, ModernBERT-large), the reference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Hugging Face repo or local directory (default: the backend's model)")
    parser.add_argument("--subfolder", type=str, default=None, choices=["multilingual", "typed-decisions"],
                        help="Laya only: checkpoint inside the repo (default: English ModernBERT-large, 512 tokens)")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Token budget per text, question included (Julia default 1024, up to 8192; "
                             "Laya: the checkpoint's own)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference (falls back to CPU when unavailable)")
    parser.add_argument("--batch_size", type=int, default=16, help="Texts per forward pass")
    add_dataset_args(parser)
    add_question_args(parser)
    add_scoring_args(parser)
    args = parser.parse_args()
    args.max_length, args.chunk_stride = args.max_len, None
    run(args, [detector_spec(args)])


if __name__ == "__main__":
    main()
