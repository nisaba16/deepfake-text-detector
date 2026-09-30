#!/usr/bin/env python3
"""
Cross-dataset evaluation script: Train on one dataset, evaluate on another.
This script helps measure how well models generalize across different datasets.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Ensure project modules are discoverable
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from load_and_evaluate import (
    load_saved_detector, 
    evaluate_detector_on_dataset, 
    print_evaluation_results,
    select_eval_part,
    warn_if_training_dataset,
)
from textdet.data import resolve


def run_cross_dataset_evaluation(args):
    """Run cross-dataset evaluation comparing different dataset combinations."""
    
    # Load the saved model
    print(f"Loading model: {args.model_path}")
    detector, metadata = load_saved_detector(args.model_path)
    
    # Print model info
    print(f"\nModel Information:")
    print(f"  Trained on: {metadata.get('dataset_used', 'Unknown')}")
    print(f"  Analysis Type: {metadata.get('analysis_type', 'Unknown')}")
    print(f"  Model: {metadata.get('model_name', 'Unknown')}")
    print(f"  Classifier: {metadata.get('classifier_type', 'Unknown')}")
    use_specialized = bool(metadata.get('use_specialized_extraction', False))
    if use_specialized:
        print(f"  Extraction: specialized (model-recommended method)")
    else:
        if metadata.get('layer'):
            print(f"  Layer: {metadata.get('layer')}")
        if metadata.get('pooling'):
            print(f"  Pooling: {metadata.get('pooling')}")
    
    # Evaluate on each dataset
    results = {}
    
    for dataset_config in args.datasets:
        # name:path (historical form), or a catalog name alone (mage_ood_gpt4, raid_sample, ...)
        ts = resolve(dataset_config)
        dataset_name = ts.name or dataset_config

        print(f"\n{'='*50}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*50}")

        # Load evaluation dataset
        texts, labels = ts.texts, ts.labels
        texts, labels = select_eval_part(texts, labels, args.eval_part, args.eval_test_frac)
        warn_if_training_dataset(metadata, dataset_name)
        print(f"Loaded {len(texts)} samples")
        print(f"Label distribution: {np.bincount(labels)} (0=real, 1=fake)")
        
        # Evaluate
        eval_results = evaluate_detector_on_dataset(
            detector, metadata, texts, labels,
            device=args.device, batch_size=args.batch_size,
            max_length=args.max_length,
            threshold=getattr(args, 'threshold', None),
            optimize_threshold=getattr(args, 'optimize_threshold', None),
            optimize_split=getattr(args, 'optimize_split', 0.2),
            random_state=getattr(args, 'random_state', 42)
        )
        
        # Print results
        model_name = Path(args.model_path).stem
        print_evaluation_results(eval_results, model_name, dataset_name)
        
        results[dataset_name] = {**eval_results['metrics'], 'eval_part': args.eval_part}
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print("CROSS-DATASET EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {Path(args.model_path).stem}")
    print(f"Trained on: {metadata.get('dataset_used', 'Unknown')}")
    print(f"\n{'Dataset':<15} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}")
    print(f"{'-'*75}")
    
    for dataset_name, metrics in results.items():
        roc_auc = metrics.get('roc_auc', float('nan'))
        roc_auc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
        print(f"{dataset_name:<15} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {roc_auc_str:<10}")
    
    # Save summary if requested
    if args.save_summary:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_df = pd.DataFrame(results).T
        summary_file = output_dir / f"cross_dataset_summary_{Path(args.model_path).stem}.csv"
        summary_df.to_csv(summary_file)
        print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation of saved models")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved detector (.pkl file)")
    
    # Datasets to evaluate on
    parser.add_argument("--datasets", nargs='+', required=True,
                       help="Dataset specs: name:path pairs (e.g., mercor_ai:data/mercor-ai/train.csv) or "
                            "catalog names (mage_ood_gpt4, raid_sample, ...; see textdet/data/catalog.py)")
    
    # Inference parameters
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for feature extraction")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")

    # Thresholding / optimization pass-through to load_and_evaluate
    parser.add_argument("--threshold", type=float, default=None,
                       help="Override decision threshold on P(fake); predictions = (P(fake) >= threshold)")
    parser.add_argument("--optimize_threshold", type=str, default=None, choices=["f1", "fpr0.01", "fpr0.05"],
                       help="Optimize a threshold on a validation split using the given metric (e.g., 'f1')")
    parser.add_argument("--optimize_split", type=float, default=0.2,
                       help="Validation split fraction when optimizing threshold")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for threshold optimization split")
    parser.add_argument("--eval_part", type=str, default="all", choices=["all", "select", "test"],
                       help="Score the whole set, or a fixed stratified part of it: 'select' to choose "
                            "configs (sweeps), 'test' only for the final number (use another --output_dir)")
    parser.add_argument("--eval_test_frac", type=float, default=0.5,
                       help="Fraction of each evaluation set in the 'test' part")
    
    # Output
    parser.add_argument("--save_summary", action="store_true",
                       help="Save summary CSV")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    run_cross_dataset_evaluation(args)


if __name__ == "__main__":
    main()