"""
The project's saved detectors (scripts/train_and_save_detector.py): a pickled `models.classifiers`
BinaryDetector / OutlierDetections next to <name>_metadata.pkl, over embedding / TF-IDF / perplexity / PHD
features. The feature extraction is rebuilt from the metadata exactly as in training.
"""
from __future__ import annotations

import gc
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..data.loading import sanitize_texts
from .base import Detector, Scores, safe_name

# The pickles reference the legacy `models` package of the repository root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))


def clear_gpu_memory():
    """Utility to clear GPU caches when working with large HF models.

    Safe on CPU-only environments (no-op for CUDA parts when unavailable).
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        # torch may be installed without CUDA; ignore
        pass
    gc.collect()


def replace_nan_with_column_means(features: np.ndarray) -> np.ndarray:
    """Replace non-finite values (NaN/Inf) with column means; fallback to zeros.

    If a scipy.sparse matrix is provided (as in TF-IDF), return unchanged since
    sparse matrices typically don't contain NaNs and require different handling.
    """
    try:
        from scipy.sparse import issparse  # type: ignore
        if issparse(features):
            return features
    except Exception:
        pass

    arr = np.asarray(features, dtype=np.float32)
    if np.isfinite(arr).all():
        return arr
    finite_mask = np.isfinite(arr)
    arr_masked = arr.copy()
    arr_masked[~finite_mask] = np.nan
    col_means = np.nanmean(arr_masked, axis=0)
    if np.isscalar(col_means):
        col_means = np.array([col_means])
    col_means = np.nan_to_num(col_means, nan=0.0)
    bad_rows, bad_cols = np.where(~finite_mask)
    if bad_rows.size > 0:
        arr[bad_rows, bad_cols] = col_means[bad_cols]
    return arr


def load_saved_detector(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """Load a saved detector and its metadata."""
    model_path = Path(model_path)

    # Load the detector
    with open(model_path, 'rb') as f:
        detector = pickle.load(f)

    # Load metadata if it exists
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.pkl"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

    # Attach model_path for downstream utilities (e.g., loading TF-IDF vectorizer)
    metadata['model_path'] = str(model_path)
    return detector, metadata


def instantiate_extractor_from_metadata(metadata: Dict[str, Any], device: str = "cuda:0", model_path: str | None = None):
    """Recreate the feature extractor from saved metadata."""
    from models.extractors import EmbeddingExtractor, TFIDFExtractor
    from models.text_features import PerplexityCalculator, TextIntrinsicDimensionCalculator

    analysis_type = metadata.get('analysis_type', 'embedding')
    model_name = metadata.get('model_name', 'Qwen/Qwen2.5-0.5B')
    layer = metadata.get('layer', 22)

    if analysis_type == "embedding":
        return EmbeddingExtractor(model_name, device=device)
    elif analysis_type == "perplexity":
        return PerplexityCalculator(model_name, device=device)
    elif analysis_type == "phd":
        return TextIntrinsicDimensionCalculator(
            model_name,
            device=device,
            layer_idx=layer,
        )
    elif analysis_type == "tfidf":
        # Load the fitted TF-IDF vectorizer saved alongside the detector
        if model_path is None:
            raise ValueError("model_path is required to load TF-IDF vectorizer for evaluation")
        vec_path = Path(model_path).with_name(Path(model_path).stem + "_vectorizer.pkl")
        if not vec_path.exists():
            raise FileNotFoundError(f"Expected TF-IDF vectorizer file not found: {vec_path}")
        with open(vec_path, 'rb') as f:
            fitted_vec = pickle.load(f)
        extractor = TFIDFExtractor(
            max_features=getattr(fitted_vec, 'max_features', None),
            ngram_range=getattr(fitted_vec, 'ngram_range', (1, 2)),
            lowercase=getattr(fitted_vec, 'lowercase', True),
            min_df=getattr(fitted_vec, 'min_df', 1),
            max_df=getattr(fitted_vec, 'max_df', 1.0),
            use_idf=getattr(fitted_vec, 'use_idf', True),
            norm=getattr(fitted_vec, 'norm', 'l2'),
            sublinear_tf=getattr(fitted_vec, 'sublinear_tf', False),
            stop_words=getattr(fitted_vec, 'stop_words', None),
        )
        # Replace with fitted vectorizer and mark as fitted
        extractor.vectorizer = fitted_vec
        extractor._is_fitted = True
        return extractor
    else:
        raise ValueError(f"Unsupported analysis_type {analysis_type}")


def get_features_from_metadata(texts: List[str], extractor, metadata: Dict[str, Any],
                               batch_size: int = 8, max_length: int = 512,
                               show_progress: bool = True, device: str = "cuda:0") -> np.ndarray:
    """Extract features using the same parameters as during training."""
    from models.extractors import pool_embeds_from_layer, l2_normalize_tokens

    processed_texts = sanitize_texts(texts)
    analysis_type = metadata.get('analysis_type', 'embedding')
    layer = metadata.get('layer', 22)
    pooling = metadata.get('pooling', 'mean')
    normalize = bool(metadata.get('normalize', False))
    use_specialized = bool(metadata.get('use_specialized_extraction', False))

    print(f"Extracting {analysis_type} features for {len(processed_texts)} texts...")

    if analysis_type == "embedding":
        # For specialized extraction, use the extractor's default method (no layer selection)
        spec_ext = None
        if use_specialized:
            print("Using specialized extraction (no layer selection)")
            extractor_name = metadata.get('model_name', '')
            # Try to get specialized extractor for all models (not just sentence-transformers/Qwen)
            from models.specialized_extractors import get_specialized_extractor
            spec_ext = get_specialized_extractor(extractor_name, device=device)
            if spec_ext is None:
                # Training falls back to the generic layer/pooling extraction in that case: do the same
                print(f"⚠️  No specialized extractor for {extractor_name}: generic extraction, as in training")
        if spec_ext is not None:
            features = spec_ext.extract(
                processed_texts,
                batch_size=batch_size,
                max_length=max_length,
            )
        else:
            # Regular embedding extraction with layer/pooling tuning
            embeds_all = extractor.get_all_layer_embeddings(
                processed_texts,
                batch_size=batch_size,
                max_length=max_length,
                show_progress=show_progress,
            )
            # Be robust to layer index if not available
            available_layers = sorted(list(embeds_all[0].keys())) if embeds_all else []
            chosen_layer = layer
            if chosen_layer < 0:
                # Python-style index into hidden_states, as in training's memory-efficient path (-1 = last)
                chosen_layer += len(available_layers)
            if chosen_layer not in available_layers and available_layers:
                fallback = available_layers[-1]
                print(f"⚠️  Requested layer {chosen_layer} not available. Using last available layer {fallback}.")
                chosen_layer = fallback
            layer_embeds = [embeds[chosen_layer] for embeds in embeds_all]
            if normalize:
                layer_embeds = l2_normalize_tokens(layer_embeds)
            features = pool_embeds_from_layer(layer_embeds, pooling=pooling)

    elif analysis_type == "perplexity":
        features = np.array(
            extractor.calculate_batch_perplexity(processed_texts, max_length=max_length)
        ).reshape(-1, 1)

    elif analysis_type == "phd":
        features = np.array(
            extractor.calculate_batch(processed_texts, max_length=max_length)
        ).reshape(-1, 1)

    elif analysis_type == "tfidf":
        # Use the fitted vectorizer to transform into sparse features
        # Keep sparse to let BinaryDetector apply SVD/MaxAbsScaler as trained
        features = extractor.transform(processed_texts, dense=False)

    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    return features


def predict_prob_fake(detector, metadata: Dict[str, Any], features) -> Tuple[np.ndarray, np.ndarray | None, Any]:
    """(predictions, P(fake) or None, raw probabilities) of a saved detector on extracted features."""
    from scipy.sparse import issparse

    features = replace_nan_with_column_means(features)
    # One-class detectors were fit on dense TF-IDF (their StandardScaler cannot center sparse input)
    if issparse(features) and metadata.get('classifier_type') in {"elliptic", "ocsvm", "iforest"}:
        features = features.toarray().astype(np.float32)

    results = detector.predict(features, return_probabilities=True, return_distances=True)
    if isinstance(results, list) and len(results) >= 2:
        predictions, probabilities = results[0], results[1]
    else:
        predictions, probabilities = results, None

    if probabilities is None:
        return np.asarray(predictions), None, None
    probs = np.asarray(probabilities)
    if probs.ndim == 1:
        prob_fake = probs                      # already P(fake)
    elif probs.shape[1] == 2:
        prob_fake = probs[:, 1]                # [P(real), P(fake)]
    else:
        prob_fake = probs.ravel()
    return np.asarray(predictions), prob_fake, probabilities


class SavedDetector(Detector):
    """A saved detector (.pkl + _metadata.pkl) behind the common interface."""
    analysis = "saved"

    def __init__(self, model_path: str, device: str = "cuda:0", batch_size: int = 8, max_length: int = 512):
        self.model_path = model_path
        self.detector, self.metadata = load_saved_detector(model_path)
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.checkpoint = Path(model_path).stem
        self.train_dataset = safe_name(self.metadata.get("dataset_used", "unknown")).replace("_", "-")

    @property
    def tag(self) -> str:
        return safe_name(self.checkpoint)

    def describe(self):
        return {"detector": "saved", "checkpoint": self.model_path,
                "analysis_type": self.metadata.get("analysis_type"), "classifier": self.metadata.get("classifier_type")}

    def summary_stem(self, head: str) -> str:
        # The historical name (cross_dataset_evaluation.py): the pickle's stem already says everything
        return Path(self.model_path).stem

    def predictions_stem(self) -> str:
        return Path(self.model_path).stem

    def score(self, texts: Sequence[str]) -> Dict[str, Scores]:
        extractor = instantiate_extractor_from_metadata(self.metadata, device=self.device, model_path=self.model_path)
        try:
            features = get_features_from_metadata(list(texts), extractor, self.metadata, batch_size=self.batch_size,
                                                  max_length=self.max_length, device=self.device)
        finally:
            del extractor
            clear_gpu_memory()
        predictions, prob_fake, _ = predict_prob_fake(self.detector, self.metadata, features)
        if prob_fake is None:
            prob_fake = predictions.astype(float)
        p = np.clip(prob_fake, 1e-12, 1 - 1e-12)
        return {"default": Scores(prob_fake, np.log(p) - np.log1p(-p))}
