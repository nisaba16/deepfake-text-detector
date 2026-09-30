#!/usr/bin/env python3
"""
Script to load saved detectors and evaluate them on different datasets for cross-dataset evaluation.

The shared pieces live in textdet (data loading, metrics, the saved-detector feature pipeline); they are
re-exported here under their historical names for the scripts that import them.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

# Ensure project modules are discoverable
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from textdet.data import EVAL_SPLIT_SEED, load_dataset, resolve, sanitize_texts, select_eval_part  # noqa: F401
from textdet.detectors.saved import (  # noqa: F401
    clear_gpu_memory,
    get_features_from_metadata,
    instantiate_extractor_from_metadata,
    load_saved_detector,
    predict_prob_fake,
    replace_nan_with_column_means,
)
from textdet.evaluation.metrics import _sweep_thresholds, apply_threshold_and_score, compute_metrics  # noqa: F401


def load_mercor_ai_dataset(data_path: str) -> Tuple[list, np.ndarray]:
    """Load the Mercor AI dataset (answer, is_cheating: 1 = AI)."""
    return load_dataset("mercor_ai", data_path)


def load_human_ai_dataset(data_path: str) -> Tuple[list, np.ndarray]:
    """Load the AI_Human.csv dataset (text, generated: 1 = AI)."""
    return load_dataset("human_ai", data_path)


def warn_if_training_dataset(metadata: Dict[str, Any], dataset_name: str) -> None:
    """Scores on the dataset a detector was trained on include its training rows."""
    if metadata.get('dataset_used') == dataset_name:
        print(f"⚠️  The detector was trained on '{dataset_name}': these rows can include its training rows, "
              f"so this is not a held-out score.")


def evaluate_detector_on_dataset(detector, metadata: Dict[str, Any],
                                texts, labels: np.ndarray,
                                device: str = "cuda:0", batch_size: int = 8,
                                max_length: int = 512,
                                threshold: float | None = None,
                                optimize_threshold: str | None = None,
                                optimize_split: float = 0.2,
                                random_state: int = 42) -> Dict[str, Any]:
    """Evaluate a saved detector on a dataset."""
    print("Creating feature extractor...")
    extractor = instantiate_extractor_from_metadata(metadata, device=device, model_path=metadata.get('model_path'))

    # Extract features
    features = get_features_from_metadata(
        texts, extractor, metadata, batch_size=batch_size,
        max_length=max_length, show_progress=True, device=device
    )

    # Make predictions
    print("Making predictions...")
    predictions, prob_fake, probabilities = predict_prob_fake(detector, metadata, features)

    predictions, metrics = apply_threshold_and_score(
        labels, predictions, prob_fake,
        threshold=threshold,
        optimize_threshold=optimize_threshold,
        optimize_split=optimize_split,
        random_state=random_state,
    )

    # Clean up extractor
    del extractor
    clear_gpu_memory()

    return {
        'metrics': metrics,
        'predictions': predictions,
        'probabilities': probabilities,
        'features_shape': features.shape,
        'metadata': metadata
    }


def print_evaluation_results(results: Dict[str, Any], model_name: str, dataset_name: str):
    """Print formatted evaluation results."""
    metrics = results['metrics']
    metadata = results['metadata']
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Training Dataset: {metadata.get('dataset_used', 'Unknown')}")
    print(f"Analysis Type: {metadata.get('analysis_type', 'Unknown')}")
    print(f"Classifier: {metadata.get('classifier_type', 'Unknown')}")
    print(f"Features Shape: {results['features_shape']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    # Threshold info if available
    if 'applied_threshold' in metrics:
        print(f"  Threshold: {metrics['applied_threshold']:.3f}")
    if 'chosen_threshold' in metrics:
        print(f"  Best Threshold (opt): {metrics['chosen_threshold']:.3f} ({metrics.get('optimize_metric','')}: {metrics.get('chosen_metric_value', float('nan')):.4f})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Load saved detector and evaluate on dataset")
    
    # Model loading
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to saved detector (.pkl file)")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, required=True, 
                       choices=["mercor_ai", "human_ai", "generic"],
                       help="Dataset type to load")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to evaluation dataset CSV")
    parser.add_argument("--text_column", type=str, default=None, 
                       help="Name of text column (for generic datasets)")
    parser.add_argument("--label_column", type=str, default=None, 
                       help="Name of label column (for generic datasets)")
    
    # Inference parameters
    parser.add_argument("--device", type=str, default="cuda:0", 
                       help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for feature extraction")
    parser.add_argument("--max_length", type=int, default=512, 
                       help="Maximum sequence length")

    # Thresholding options
    parser.add_argument("--threshold", type=float, default=None,
                       help="Override decision threshold on P(fake); if set, predictions = (P(fake) >= threshold)")
    parser.add_argument("--optimize_threshold", type=str, default=None, choices=["f1", "fpr0.01", "fpr0.05"],
                       help="Optimize a threshold on a validation split of the evaluation set using the given metric (e.g., 'f1')")
    parser.add_argument("--optimize_split", type=float, default=0.2,
                       help="Fraction of eval set used as validation to select the threshold when --optimize_threshold is set")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for threshold optimization split")
    parser.add_argument("--eval_part", type=str, default="all", choices=["all", "select", "test"],
                       help="Score the whole set, or a fixed stratified part of it: 'select' to choose "
                            "configs, 'test' only for the final number")
    parser.add_argument("--eval_test_frac", type=float, default=0.5,
                       help="Fraction of the evaluation set in the 'test' part")
    
    # Output options
    parser.add_argument("--save_predictions", action="store_true", 
                       help="Save predictions to CSV file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load detector and metadata
    print(f"Loading detector from {args.model_path}...")
    detector, metadata = load_saved_detector(args.model_path)
    
    # Load evaluation dataset
    print(f"Loading {args.dataset_name} dataset from {args.data_path}...")
    texts, labels = load_dataset(args.dataset_name, args.data_path, 
                                args.text_column, args.label_column)
    texts, labels = select_eval_part(texts, labels, args.eval_part, args.eval_test_frac)
    warn_if_training_dataset(metadata, args.dataset_name)
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)} (0=real, 1=fake)")
    
    # Evaluate
    results = evaluate_detector_on_dataset(
        detector, metadata, texts, labels, 
        device=args.device, batch_size=args.batch_size, 
        max_length=args.max_length,
        threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        optimize_split=args.optimize_split,
        random_state=args.random_state
    )
    results['metrics']['eval_part'] = args.eval_part
    
    # Print results
    model_name = Path(args.model_path).stem
    print_evaluation_results(results, model_name, args.dataset_name)
    
    # Save results if requested
    if args.save_predictions:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions CSV
        prob_fake = None
        if results.get('probabilities') is not None:
            probs_np = np.asarray(results['probabilities'])
            if probs_np.ndim == 1:
                prob_fake = probs_np
            elif probs_np.ndim == 2:
                if probs_np.shape[1] == 2:
                    prob_fake = probs_np[:, 1]
                elif probs_np.shape[1] == 1:
                    prob_fake = probs_np[:, 0]
                else:
                    prob_fake = probs_np.ravel()
            else:
                prob_fake = probs_np.ravel()

        data_dict = {
            'prediction': results['predictions'],
            'true_label': labels
        }
        if prob_fake is not None:
            data_dict['probability_fake'] = prob_fake
        
        pred_df = pd.DataFrame(data_dict)
        pred_file = output_dir / f"predictions_{model_name}_{args.dataset_name}.csv"
        pred_df.to_csv(pred_file, index=False)
        
        # Save metrics JSON
        import json
        metrics_file = output_dir / f"metrics_{model_name}_{args.dataset_name}.json"
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()