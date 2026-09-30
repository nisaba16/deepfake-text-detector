import sys
import gc
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F

# Ensure project modules are discoverable
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from models.extractors import EmbeddingExtractor, pool_embeds_from_layer, l2_normalize_tokens, TFIDFExtractor
from models.text_features import PerplexityCalculator, TextIntrinsicDimensionCalculator
from models.classifiers import BinaryDetector, OutlierDetections
from models.specialized_extractors import get_specialized_extractor
import torch.nn.functional as F


def clear_gpu_memory():
    """Utility to clear GPU caches when working with large HF models."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def sanitize_texts(texts: List[str]) -> List[str]:
    """Ensure every text is a non-empty string accepted by tokenizers."""
    return [t.strip() if isinstance(t, str) and t.strip() else " " for t in texts]


def replace_nan_with_column_means(features: np.ndarray) -> np.ndarray:
    """Replace non-finite values (NaN/Inf) column-wise with column means; fallback to zeros.

    Handles both dense numpy arrays and sparse matrices. For sparse inputs, TF-IDF
    typically contains no NaNs and scipy.sparse doesn't support NaN ops; we simply
    return the matrix unchanged.
    """
    # Gracefully skip sparse inputs
    try:
        from scipy.sparse import issparse  # type: ignore
        if issparse(features):
            return features
    except Exception:
        pass

    arr = np.asarray(features, dtype=np.float32)

    # If no non-finite values, return as-is
    if np.isfinite(arr).all():
        return arr

    # Compute column means over finite values only
    finite_mask = np.isfinite(arr)
    # Avoid empty slices by setting invalid to NaN then using nanmean
    arr_masked = arr.copy()
    arr_masked[~finite_mask] = np.nan
    col_means = np.nanmean(arr_masked, axis=0)
    if np.isscalar(col_means):  # handle 1D features (shape: (n,))
        col_means = np.array([col_means])
    col_means = np.nan_to_num(col_means, nan=0.0)

    # Replace NaN/Inf with column means
    bad_rows, bad_cols = np.where(~finite_mask)
    if bad_rows.size > 0:
        arr[bad_rows, bad_cols] = col_means[bad_cols]
    return arr


def get_features(texts: List[str], extractor, args, show_progress: bool = True):
    """Run the selected feature extractor on a list of texts.

    Memory efficient embedding path: if args.memory_efficient is True and analysis_type=embedding,
    we call extractor.get_pooled_layer_embeddings to avoid materializing every layer & sequence.
    
    Specialized extraction: if args.use_specialized_extraction is True and the model has a 
    specialized extractor (e.g., Qwen embedding models, Sentence-Transformers), use that instead.
    """
    processed_texts = sanitize_texts(texts)
    print(f"Extracting features for {len(processed_texts)} texts...")

    # Check if we should use specialized extraction
    if (args.analysis_type == "embedding" and 
        getattr(args, 'use_specialized_extraction', False)):
        spec_extractor = get_specialized_extractor(args.model_name, args.device)
        if spec_extractor is not None:
            print(f"Using specialized extraction for {args.model_name}")
            print("  (No layer selection, pooling, or PCA - using model's recommended method)")
            features = spec_extractor.extract(
                processed_texts,
                batch_size=args.batch_size,
                max_length=args.max_length
            )
            return features
        else:
            print(f"⚠️  No specialized extractor found for {args.model_name}. Falling back to generic extraction.")

    if args.analysis_type == "embedding":
        if getattr(args, 'memory_efficient', False):
            print("Using memory-efficient single-layer pooled extraction path.")
            normalize = getattr(args, 'normalize', False)
            if normalize:
                print("  With L2 normalization post-pooling.")
            features = extractor.get_pooled_layer_embeddings(
                processed_texts,
                layer_idx=args.layer,
                pooling=args.pooling,
                batch_size=args.batch_size,
                max_length=args.max_length,
                show_progress=show_progress,
                normalize=normalize,
            )
        else:
            embeds_all = extractor.get_all_layer_embeddings(
                processed_texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                show_progress=show_progress,
            )
            # Determine valid layer index
            available_layers = sorted(list(embeds_all[0].keys())) if embeds_all else []
            chosen_layer = args.layer
            if chosen_layer < 0:
                # Python-style index into hidden_states, as in the memory-efficient path (-1 = last)
                chosen_layer += len(available_layers)
            if chosen_layer not in available_layers:
                # Fallback to last available layer
                if available_layers:
                    fallback = available_layers[-1]
                    print(f"⚠️  Requested layer {chosen_layer} not available. Using last available layer {fallback}.")
                    chosen_layer = fallback
                    try:
                        args.layer = chosen_layer
                    except Exception:
                        pass
                else:
                    raise ValueError("No layers available in embeddings.")

            layer_embeds = [embeds[chosen_layer] for embeds in embeds_all]
            if getattr(args, 'normalize', False):
                layer_embeds = l2_normalize_tokens(layer_embeds)
            features = pool_embeds_from_layer(layer_embeds, pooling=args.pooling)

    elif args.analysis_type == "perplexity":
        features = np.array(
            extractor.calculate_batch_perplexity(processed_texts, max_length=args.max_length)
        ).reshape(-1, 1)

    elif args.analysis_type == "phd":
        features = np.array(
            extractor.calculate_batch(processed_texts, max_length=args.max_length)
        ).reshape(-1, 1)

    elif args.analysis_type == "tfidf":
        dense = bool(getattr(args, 'tfidf_dense', False))
        features = extractor.fit_transform(processed_texts, dense=dense)

    else:
        raise ValueError(f"Unknown analysis_type: {args.analysis_type}")

    return features


def instantiate_extractor(args):
    """Factory that returns the appropriate feature extractor."""
    if args.analysis_type == "embedding":
        return EmbeddingExtractor(
            args.model_name,
            device=args.device,
            log_memory=getattr(args, 'log_memory', False),
            memory_interval=getattr(args, 'memory_log_interval', 1)
        )
    if args.analysis_type == "perplexity":
        return PerplexityCalculator(args.model_name, device=args.device)
    if args.analysis_type == "phd":
        return TextIntrinsicDimensionCalculator(
            args.model_name,
            device=args.device,
            layer_idx=args.layer,
        )
    if args.analysis_type == "tfidf":
        return TFIDFExtractor(
            max_features=getattr(args, 'tfidf_max_features', 20000),
            ngram_range=(getattr(args, 'tfidf_ngram_min', 1), getattr(args, 'tfidf_ngram_max', 2)),
            lowercase=True,
            min_df=getattr(args, 'tfidf_min_df', 1),
            max_df=getattr(args, 'tfidf_max_df', 1.0),
            use_idf=True,
            norm="l2",
            sublinear_tf=False,
            stop_words=getattr(args, 'tfidf_stop_words', None),
        )
    raise ValueError(f"Unsupported analysis_type {args.analysis_type}")


def generate_model_name(args, dataset_name: str) -> str:
    """Generate standardized, filesystem-safe model filename."""
    def sanitize_for_filename(s: str) -> str:
        # Replace any character that's not alphanumeric, dash, underscore, or dot with underscore
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
        # Collapse repeated underscores and trim leading/trailing separators
        s = re.sub(r"_+", "_", s).strip("._-")
        return s or "model"

    if args.analysis_type == "embedding":
        # Check if using specialized extraction (no layer/pooling tuning)
        if getattr(args, 'use_specialized_extraction', False):
            feature_type = "specialized_embedding"
        else:
            norm_tag = "_l2norm" if getattr(args, 'normalize', False) else ""
            feature_type = f"embedding_layer{args.layer}_{args.pooling}{norm_tag}"
    elif args.analysis_type == "perplexity":
        feature_type = "perplexity"
    elif args.analysis_type == "phd":
        feature_type = f"phd_layer{args.layer}"
    else:
        feature_type = args.analysis_type

    safe_dataset = sanitize_for_filename(dataset_name)
    safe_model = sanitize_for_filename(getattr(args, 'model_name', 'model'))
    safe_feature = sanitize_for_filename(feature_type)
    safe_classifier = sanitize_for_filename(getattr(args, 'classifier_type', 'clf'))

    return f"{safe_dataset}_{safe_model}_{safe_feature}_{safe_classifier}"


def save_detector_and_metadata(detector: BinaryDetector, args, save_dir: Path, model_name: str, extractor=None):
    """Save the trained detector and metadata for later use."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the detector
    detector_path = save_dir / f"{model_name}.pkl"
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    with open(detector_path, 'wb') as f:
        pickle.dump(detector, f)
    
    # Save metadata for reproducibility
    metadata = {
        'model_name': args.model_name,
        'analysis_type': args.analysis_type,
        'classifier_type': args.classifier_type,
        'layer': getattr(args, 'layer', None),
        'pooling': getattr(args, 'pooling', None),
        'normalize': getattr(args, 'normalize', False),
        'use_specialized_extraction': getattr(args, 'use_specialized_extraction', False),
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'validation_split': args.validation_split,
        'dataset_used': args.dataset_name,
        'text_column': args.text_column,
        'label_column': args.label_column,
        # Data loading/sampling provenance
        'n_rows': getattr(args, 'n_rows', None),
        'sample_frac': getattr(args, 'sample_frac', None),
        'stratified_sample': getattr(args, 'stratified_sample', False),
        'random_state': getattr(args, 'random_state', None),
        'train_size': getattr(args, 'train_size', None),
        'train_label_counts': getattr(args, 'train_label_counts', None),
    }

    if args.analysis_type == "tfidf":
        metadata.update({
            'tfidf_max_features': getattr(args, 'tfidf_max_features', None),
            'tfidf_ngram_min': getattr(args, 'tfidf_ngram_min', None),
            'tfidf_ngram_max': getattr(args, 'tfidf_ngram_max', None),
            'tfidf_min_df': getattr(args, 'tfidf_min_df', None),
            'tfidf_max_df': getattr(args, 'tfidf_max_df', None),
            'tfidf_dense': getattr(args, 'tfidf_dense', False),
            'svd_components': getattr(args, 'svd_components', None),
        })
    
    metadata_path = save_dir / f"{model_name}_metadata.pkl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Saved detector to {detector_path}")
    print(f"✅ Saved metadata to {metadata_path}")

    # Save fitted TF-IDF vectorizer alongside the detector when applicable
    if getattr(args, 'analysis_type', None) == 'tfidf' and extractor is not None:
        try:
            vec = getattr(extractor, 'vectorizer', None)
            is_fitted = getattr(extractor, '_is_fitted', False)
            if vec is not None and is_fitted:
                vec_path = save_dir / f"{model_name}_vectorizer.pkl"
                with open(vec_path, 'wb') as f:
                    pickle.dump(vec, f)
                print(f"✅ Saved TF-IDF vectorizer to {vec_path}")
        except Exception as e:
            print(f"⚠️  Warning: failed to save TF-IDF vectorizer: {e}")


def sample_rows(df: pd.DataFrame, label_col: str, n_rows: int = None, stratified: bool = False,
                seed: int = 42) -> pd.DataFrame:
    """Random subset of n_rows taken from the whole file (all rows when n_rows is None).

    The CSVs are not shuffled, so their first rows are a biased slice: the first 10k rows of
    AI_Human.csv only contain AI essays about two prompts. With stratified=True the subset is
    balanced: n_rows // n_classes rows per class (fewer if a class is smaller).
    """
    if not n_rows:
        return df
    if stratified:
        per_class = n_rows // df[label_col].nunique()
        parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in df.groupby(label_col)]
        return pd.concat(parts).sample(frac=1, random_state=seed)  # mix the classes
    return df.sample(min(n_rows, len(df)), random_state=seed)


def coerce_binary_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Labels as int 0/1 (numbers, bools or 'true'/'false'); rows without a label are dropped.

    Unparseable labels used to become 0 (human) silently; anything that is not 0/1 is now an error.
    """
    raw = df[label_col]
    if raw.dtype == object:
        raw = raw.astype(str).str.strip().str.lower().replace({'true': '1', 'false': '0'})
    labels = pd.to_numeric(raw, errors='coerce')
    missing = labels.isna()
    if missing.any():
        print(f"⚠️  Dropping {int(missing.sum())} rows without a numeric '{label_col}' label "
              f"(e.g. {df.loc[missing, label_col].astype(str).unique()[:3].tolist()})")
    labels = labels[~missing]
    if labels.empty or not labels.isin([0, 1]).all():
        raise ValueError(f"'{label_col}' must hold binary 0/1 labels, found {sorted(pd.unique(labels))[:10]}")
    df = df.loc[~missing].copy()
    df[label_col] = labels.astype(int)
    return df


def load_human_ai_dataset(data_path: str, n_rows: int = None, stratified: bool = False,
                          seed: int = 42) -> Tuple[List[str], np.ndarray]:
    """Load the AI_Human.csv dataset, optionally a random subset of n_rows."""
    df = sample_rows(pd.read_csv(data_path), 'generated', n_rows, stratified, seed)
    texts = sanitize_texts(df['text'].astype(str).tolist())
    labels = df['generated'].astype(int).to_numpy()  # 0=human, 1=AI
    return texts, labels


def _resolve_daigtv2_csv_path(data_path: str) -> Path:
    """Resolve a DAIGT v2 CSV path.

    Accepts either a direct CSV path or a directory containing the CSV.
    Prefers files like 'train_v2_*.csv' when multiple are present.
    """
    p = Path(data_path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return p
    if p.is_dir():
        # Prefer train_v2_* pattern; otherwise any CSV
        candidates = sorted(p.glob("train_v2_*.csv"))
        if not candidates:
            candidates = sorted(p.glob("*.csv"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not find DAIGT v2 CSV at {data_path}")


def load_daigtv2_dataset(data_path: str, n_rows: int = None, stratified: bool = False,
                         seed: int = 42) -> Tuple[List[str], np.ndarray]:
    """Load the DAIGT v2 dataset, optionally a random subset of n_rows.

    Expected columns: 'text' (string), 'label' (0/1). Additional columns are ignored.
    The function accepts a direct CSV path or a directory containing the CSV.
    """
    csv_path = _resolve_daigtv2_csv_path(data_path)
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(
            f"DAIGT v2 CSV must contain 'text' and 'label' columns. Found: {list(df.columns)}"
        )
    # Some DAIGT releases may have label as bool/str; coerce to int {0,1}
    df = coerce_binary_labels(df, 'label')
    df = sample_rows(df, 'label', n_rows, stratified, seed)
    texts = sanitize_texts(df['text'].astype(str).tolist())
    labels = df['label'].to_numpy()
    return texts, labels


def train_detector(train_texts: List[str], train_labels: np.ndarray, args) -> Tuple[object, object]:
    """Extract features for training data and fit the BinaryDetector."""
    print("Extracting training features...")
    extractor = instantiate_extractor(args)
    train_features = get_features(train_texts, extractor, args)
    train_features = replace_nan_with_column_means(train_features)

    print(f"Training features shape: {train_features.shape}")
    # Dimensionality reduction strategy per analysis type
    if args.analysis_type == "embedding":
        n_components = 0.95  # PCA keep 95% variance
    elif args.analysis_type == "tfidf":
        n_components = getattr(args, 'svd_components', None)  # int or None
    else:
        n_components = None
    input_dim = train_features.shape[1]

    outlier_types = {"elliptic", "ocsvm", "iforest"}
    if args.classifier_type in outlier_types:
        # For TF-IDF + one-class, ensure dense features (StandardScaler/PCA expect dense)
        from scipy.sparse import issparse
        if issparse(train_features):
            train_features = train_features.toarray().astype(np.float32)
        # Use OutlierDetections (one-class style) with its own PCA pipeline
        detector = OutlierDetections(
            detector_type=args.classifier_type,
            contamination=0.1,
            random_state=42,
            n_components=(0.95 if args.analysis_type == "embedding" else 0.95),
        )
        training_results = detector.fit(
            embeddings=train_features,
            labels=train_labels,
            validation_split=args.validation_split,
        )
    else:
        # Standard binary path
        detector = BinaryDetector(
            n_components=n_components,
            contamination=0.1,
            random_state=42,
            input_dim=None if n_components is not None else input_dim,
        )

        training_results = detector.fit(
            embeddings=train_features,
            labels=train_labels,
            validation_split=args.validation_split,
            classifier_type=args.classifier_type,
            pca=(n_components is not None),
        )

    print(f"Training completed. Validation accuracy: {training_results.get('val_accuracy', 'N/A')}")
    return detector, extractor


def run_training_pipeline(args):
    """Main training pipeline."""
    # Load training data (--n_rows: random subset of the whole file, balanced with --stratified_sample)
    n_rows = getattr(args, 'n_rows', None)
    stratified = getattr(args, 'stratified_sample', False)
    seed = getattr(args, 'random_state', 42)
    if args.dataset_name == "human_ai":
        train_texts, train_labels = load_human_ai_dataset(args.train_data_path, n_rows, stratified, seed)
    elif args.dataset_name in {"daigtv2", "daigt_v2", "daigt"}:
        # Dedicated loader with fixed columns
        train_texts, train_labels = load_daigtv2_dataset(args.train_data_path, n_rows, stratified, seed)
    else:
        # For other datasets, load from CSV with specified columns
        train_df = pd.read_csv(args.train_data_path)
        # Allow dataset-specific smart defaults if user didn't override columns
        text_col = getattr(args, 'text_column', None)
        label_col = getattr(args, 'label_column', None)
        if text_col is None:
            # Try common names
            for cand in ("text", "content", "answer"):
                if cand in train_df.columns:
                    text_col = cand
                    break
        if label_col is None:
            for cand in ("label", "generated", "is_cheating", "target"):
                if cand in train_df.columns:
                    label_col = cand
                    break
        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not infer text/label columns. Available columns: {list(train_df.columns)}.\n"
                f"Pass --text_column and --label_column explicitly."
            )
        args.text_column, args.label_column = text_col, label_col  # record the inferred columns in metadata
        train_df = coerce_binary_labels(train_df, label_col)
        train_df = sample_rows(train_df, label_col, n_rows, stratified, seed)
        train_texts = sanitize_texts(train_df[text_col].astype(str).tolist())
        train_labels = train_df[label_col].to_numpy()

    # Optional sampling for faster experiments
    if getattr(args, 'sample_frac', None):
        frac = float(args.sample_frac)
        if not (0 < frac <= 1):
            raise ValueError("--sample_frac must be in (0,1]")
        print(f"Sampling {frac:.2%} of the training data with random_state={getattr(args, 'random_state', 42)}")
        # Rebuild a DataFrame to sample in a label-aware manner
        df_tmp = pd.DataFrame({
            'text': train_texts,
            'label': train_labels
        })
        # Stratified sample per label when possible
        sampled = df_tmp.groupby('label', group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=getattr(args, 'random_state', 42))
        ) if len(np.unique(train_labels)) > 1 else df_tmp.sample(frac=frac, random_state=getattr(args, 'random_state', 42))

        train_texts = sampled['text'].tolist()
        train_labels = sampled['label'].to_numpy()

    # Persist basic dataset stats for reproducibility in saved metadata
    args.train_size = len(train_texts)
    try:
        label_counts = np.bincount(train_labels).tolist()
    except Exception:
        # Fallback in case labels are not integer-typed for any reason
        unique, counts = np.unique(train_labels, return_counts=True)
        label_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    args.train_label_counts = label_counts

    print(f"Training on {len(train_texts)} samples from {args.dataset_name} dataset")
    print(f"Label distribution: {np.bincount(train_labels)} (0=real, 1=fake)")

    # Train detector
    detector, extractor = train_detector(train_texts, train_labels, args)

    # Generate model name and save
    model_name = generate_model_name(args, args.dataset_name)
    save_dir = Path("saved_models")
    save_detector_and_metadata(detector, args, save_dir, model_name, extractor=extractor)

    # Clean up extractor to free memory (only after saving TF-IDF vectorizer if applicable)
    if extractor:
        del extractor
    if args.analysis_type == "embedding":
        clear_gpu_memory()

    print(f"✅ Training pipeline completed. Model saved as: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detector on dataset and save for cross-dataset evaluation")

    # Model and feature extraction
    parser.add_argument("--model_name", type=str, required=False, default="sentence-transformers/all-distilroberta-v1",
                        help="HF model to use for feature extraction (ignored when --analysis_type=tfidf)")
    parser.add_argument("--analysis_type", type=str, default="embedding", choices=["embedding", "perplexity", "phd", "tfidf"],
                        help="Feature family to compute; use 'tfidf' for classical TF-IDF vectors")
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="svm",
        choices=["neural", "svm", "lr", "xgb", "elliptic", "ocsvm", "iforest"],
        help="Classifier head: binary (svm/lr/xgb/neural) or outlier (elliptic/ocsvm/iforest)"
    )

    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="human_ai", help="Dataset identifier for naming")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Name of the text column (default: first of text/content/answer found)")
    parser.add_argument("--label_column", type=str, default=None,
                        help="Name of the label column (default: first of label/generated/is_cheating/target found)")

    # Feature extraction parameters
    parser.add_argument("--layer", type=int, default=22, help="Layer index for embedding/phd analysis")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "last", "attn_mean", "mean_std", "statistical", "covariance"],
        help="Pooling strategy when analysis_type=embedding (supports: mean, max, last, attn_mean, mean_std, statistical/covariance)"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Forward batch size for HF extractor")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length for feature extractors")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to run on")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio for BinaryDetector")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient single-layer pooled extraction for embeddings")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize token embeddings before pooling (embedding analysis only)")
    parser.add_argument("--use_specialized_extraction", action="store_true", 
                        help="Use model-specific extraction method (if available) instead of layer/pooling/PCA. "
                             "Supported models: Qwen3-Embedding-*, sentence-transformers/*, xlm-roberta-*, roberta-*. "
                             "Ignores --layer, --pooling, --normalize flags.")
    parser.add_argument("--log_memory", action="store_true", help="Print memory usage during embedding extraction")
    parser.add_argument("--memory_log_interval", type=int, default=1, help="Batches between memory log prints")

    # Data subsampling options for large CSVs
    parser.add_argument("--n_rows", type=int, default=None, help="Use a random subset of N rows from the whole CSV (not the first N: the files are not shuffled)")
    parser.add_argument("--sample_frac", type=float, default=None, help="Optionally sample a fraction of rows after loading (0<frac<=1)")
    parser.add_argument("--stratified_sample", action="store_true", help="With --n_rows, sample balanced classes (N/2 real, N/2 fake)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for sampling")

    # TF-IDF specific parameters
    parser.add_argument("--tfidf_max_features", type=int, default=20000, help="Max vocabulary size for TF-IDF")
    parser.add_argument("--tfidf_ngram_min", type=int, default=1, help="Minimum n-gram for TF-IDF")
    parser.add_argument("--tfidf_ngram_max", type=int, default=2, help="Maximum n-gram for TF-IDF")
    parser.add_argument("--tfidf_min_df", type=float, default=1, help="Min document frequency for TF-IDF")
    parser.add_argument("--tfidf_max_df", type=float, default=1.0, help="Max document frequency for TF-IDF")
    parser.add_argument("--tfidf_stop_words", type=str, default=None, help="Stop words for TF-IDF (e.g., 'english')")
    parser.add_argument("--tfidf_dense", action="store_true", help="Return dense arrays instead of sparse CSR for TF-IDF")
    parser.add_argument("--svd_components", type=int, default=None, help="Use TruncatedSVD with given components for TF-IDF")

    args = parser.parse_args()
    run_training_pipeline(args)