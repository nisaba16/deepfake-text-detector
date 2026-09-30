"""
Inference backend abstraction layer.
Supports: Local (VPS), Modal, and Client-side inference modes.
"""
import asyncio
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class InferenceBackend(str, Enum):
    """Supported inference backends"""
    LOCAL = "local"
    MODAL = "modal"
    CLIENT_SIDE = "client"


class InferenceResult:
    """Standard result format across all backends"""
    def __init__(
        self,
        prediction: int,
        probability: float,
        confidence: float,
        backend: InferenceBackend,
        model_id: str,
        metadata: Dict[str, Any] = None,
        latency_ms: float = 0,
        error: Optional[str] = None
    ):
        self.prediction = prediction
        self.probability = probability
        self.confidence = confidence
        self.backend = backend
        self.model_id = model_id
        self.metadata = metadata or {}
        self.latency_ms = latency_ms
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "backend": self.backend.value,
            "model_id": self.model_id,
            "metadata": self.metadata,
            "latency_ms": self.latency_ms,
            "error": self.error
        }


class BaseInferenceEngine(ABC):
    """Abstract base class for inference engines"""
    
    @abstractmethod
    async def predict(
        self,
        text: str,
        model_id: str
    ) -> InferenceResult:
        """Run prediction using this backend"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if backend is available/configured"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of backend"""
        pass


class LocalInferenceEngine(BaseInferenceEngine):
    """VPS-local inference engine"""
    
    def __init__(self, models_dir: str = "/saved_models_prod"):
        self.models_dir = Path(models_dir).resolve()
        self.detector_cache = {}
        self.extractor_cache = {}
        self.available = False
        
        # Import here to avoid errors if libraries not installed
        try:
            import torch
            from models.extractors import EmbeddingExtractor, TFIDFExtractor
            from models.text_features import PerplexityCalculator, TextIntrinsicDimensionCalculator
            from models.classifiers import BinaryDetector
            
            self.torch = torch
            self.EmbeddingExtractor = EmbeddingExtractor
            self.TFIDFExtractor = TFIDFExtractor
            self.PerplexityCalculator = PerplexityCalculator
            self.TextIntrinsicDimensionCalculator = TextIntrinsicDimensionCalculator
            self.BinaryDetector = BinaryDetector
            self.available = True
        except ImportError as e:
            logger.warning(f"Local inference engine not available: {e}")
            self.available = False
    
    async def is_available(self) -> bool:
        """Check GPU/CPU availability"""
        if not self.available:
            return False
        
        # Check if we have enough resources
        try:
            has_gpu = self.torch.cuda.is_available()
            logger.info(f"GPU available for local inference: {has_gpu}")
            return True
        except:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check local backend health"""
        if not self.available:
            return {"status": "unavailable", "reason": "Dependencies not installed"}
        
        try:
            models_list = list(self.models_dir.glob("*.pkl"))
            return {
                "status": "healthy",
                "available_models": len(models_list),
                "gpu_available": self.torch.cuda.is_available(),
                "gpu_memory_mb": self._get_gpu_memory() if self.torch.cuda.is_available() else 0
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_gpu_memory(self) -> int:
        """Get available GPU memory in MB"""
        try:
            return int(self.torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        except:
            return 0
    
    async def predict(
        self,
        text: str,
        model_id: str
    ) -> InferenceResult:
        """Run local prediction"""
        import time
        start_time = time.time()
        
        try:
            if not self.available:
                return InferenceResult(
                    prediction=-1,
                    probability=0.0,
                    confidence=0.0,
                    backend=InferenceBackend.LOCAL,
                    model_id=model_id,
                    error="Local inference engine not available"
                )
            
            # Run in thread pool to avoid blocking
            result = await asyncio.to_thread(
                self._run_local_prediction,
                text,
                model_id
            )
            
            result.latency_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Local prediction error: {e}", exc_info=True)
            return InferenceResult(
                prediction=-1,
                probability=0.0,
                confidence=0.0,
                backend=InferenceBackend.LOCAL,
                model_id=model_id,
                error=str(e)
            )
    
    def _run_local_prediction(self, text: str, model_id: str) -> InferenceResult:
        """Actually run the prediction (in thread)"""
        
        # Load model and metadata
        model_path = self.models_dir / f"{model_id}.pkl"
        if not model_path.exists():
            model_path = self.models_dir / f"detector_{model_id}.pkl"
        
        if not model_path.exists():
            # Provide helpful diagnostics of similarly named files
            similar = [p.name for p in self.models_dir.glob(f"{model_id}*.pkl")]
            detector_similar = [p.name for p in self.models_dir.glob(f"detector_{model_id}*.pkl")]
            candidates = similar + detector_similar
            raise FileNotFoundError(
                f"Model not found: {model_id}. Looked for '{model_id}.pkl' and 'detector_{model_id}.pkl'. "
                f"Found candidates: {candidates if candidates else 'NONE'}"
            )
        
        # Load detector and metadata once per file version (a redeployed file has a new mtime)
        cache_key = (str(model_path), model_path.stat().st_mtime)
        if cache_key not in self.detector_cache:
            with open(model_path, "rb") as f:
                detector = pickle.load(f)
            metadata = {}
            metadata_path = model_path.with_name(model_path.stem + "_metadata.pkl")
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
            self.detector_cache[cache_key] = (detector, metadata)
        detector, metadata = self.detector_cache[cache_key]
        
        # Extract features based on analysis type using proper extractors
        analysis_type = metadata.get("analysis_type", "embedding")
        features = self._extract_features_properly(text, metadata, analysis_type)
        
        # Make prediction
        results = detector.predict(features, return_probabilities=True, return_distances=False)
        
        if isinstance(results, (list, tuple)) and len(results) >= 2:
            pred = int(results[0][0])
            prob = float(results[1][0])
        else:
            prob = float(np.asarray(results).squeeze())
            pred = int(prob >= 0.5)
        
        confidence = abs(prob - 0.5) * 2
        
        return InferenceResult(
            prediction=pred,
            probability=prob,
            confidence=confidence,
            backend=InferenceBackend.LOCAL,
            model_id=model_id,
            metadata={
                "analysis_type": analysis_type,
                "model_name": metadata.get("model_name"),
                "layer": metadata.get("layer"),
                "pooling": metadata.get("pooling"),
                "dataset": metadata.get("dataset_used")
            }
        )
    
    def _extract_features_properly(self, text: str, metadata: Dict[str, Any], analysis_type: str):
        """Extract features using the same method as during training"""
        import torch.nn.functional as F
        from models.extractors import pool_embeds_from_layer
        
        # Sanitize text
        processed_text = text if text and text.strip() else " "
        
        if analysis_type == "embedding":
            # Get extraction parameters from metadata
            model_name = metadata.get("model_name", "Qwen/Qwen2.5-0.5B")
            layer = metadata.get("layer", 16)
            pooling = metadata.get("pooling", "mean")
            normalize = metadata.get("normalize", False)
            batch_size = metadata.get("batch_size", 16)
            max_length = metadata.get("max_length", 512)
            use_specialized = metadata.get("use_specialized_extraction", False)
            device = "cuda" if self.torch.cuda.is_available() else "cpu"

            if use_specialized:
                # Trained on the model's recommended embedding (no layer/pooling): extract it the same way.
                # Without a specialized extractor, training used the generic path below.
                from models.specialized_extractors import get_specialized_extractor
                key = ("specialized", model_name)
                if key not in self.extractor_cache:
                    self.extractor_cache[key] = get_specialized_extractor(model_name, device)
                if self.extractor_cache[key] is not None:
                    return self.extractor_cache[key].extract([processed_text], batch_size=1, max_length=max_length)
            
            # Create extractor if not cached
            key = ("embedding", model_name)
            if key not in self.extractor_cache:
                logger.info(f"Creating EmbeddingExtractor for {model_name}")
                self.extractor_cache[key] = self.EmbeddingExtractor(model_name=model_name, device=device)
            
            extractor = self.extractor_cache[key]
            
            # Extract embeddings using get_pooled_layer_embeddings (more efficient)
            try:
                # Use the memory-efficient pooled extraction
                features = extractor.get_pooled_layer_embeddings(
                    texts=[processed_text],
                    layer_idx=layer,
                    pooling=pooling,
                    batch_size=1,
                    max_length=max_length,
                    show_progress=False,
                    normalize=normalize
                )
                return features
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                raise
        
        elif analysis_type == "perplexity":
            model_name = metadata.get("model_name", "Qwen/Qwen2.5-0.5B")
            max_length = metadata.get("max_length", 512)
            
            key = ("perplexity", model_name)  # keyed by analysis type: models can share a base model
            if key not in self.extractor_cache:
                logger.info(f"Creating PerplexityCalculator for {model_name}")
                self.extractor_cache[key] = self.PerplexityCalculator(
                    model_name=model_name,
                    device="cuda" if self.torch.cuda.is_available() else "cpu"
                )
            
            extractor = self.extractor_cache[key]
            perplexity = extractor.calculate_perplexity(processed_text, max_length=max_length)
            return np.array([[perplexity]])
        
        elif analysis_type == "phd":
            model_name = metadata.get("model_name", "Qwen/Qwen2.5-0.5B")
            layer = metadata.get("layer", 16)
            max_length = metadata.get("max_length", 512)
            
            key = ("phd", model_name, layer)
            if key not in self.extractor_cache:
                logger.info(f"Creating TextIntrinsicDimensionCalculator for {model_name}")
                self.extractor_cache[key] = self.TextIntrinsicDimensionCalculator(
                    model_name=model_name,
                    layer_idx=layer,
                    device="cuda" if self.torch.cuda.is_available() else "cpu"
                )
            
            extractor = self.extractor_cache[key]
            phd_value = extractor.calculate_batch([processed_text], max_length=max_length)[0]
            return np.array([[phd_value]])
        
        elif analysis_type == "tfidf":
            # For TF-IDF, we need the vectorizer that was fitted during training
            # It should be loaded with the detector or stored separately
            raise NotImplementedError("TF-IDF inference not yet implemented - requires fitted vectorizer")
        
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")


class ModalInferenceEngine(BaseInferenceEngine):
    """Modal serverless inference engine"""
    
    def __init__(self, token_id: str = "", token_secret: str = "", workspace: str = ""):
        self.token_id = token_id
        self.token_secret = token_secret
        self.workspace = workspace
        self.available = bool(token_id and token_secret)
        
        if self.available:
            try:
                import modal
                self.modal = modal
                logger.info("Modal backend initialized")
            except ImportError:
                logger.warning("Modal not installed")
                self.available = False
    
    async def is_available(self) -> bool:
        """Check if Modal API is configured"""
        return self.available
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Modal backend health"""
        if not self.available:
            return {"status": "unavailable", "reason": "Modal not configured"}
        
        try:
            # In production, call actual Modal health endpoint
            return {"status": "healthy", "provider": "Modal"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def predict(
        self,
        text: str,
        model_id: str
    ) -> InferenceResult:
        """Run prediction on Modal"""
        import time
        start_time = time.time()
        
        try:
            if not self.available:
                return InferenceResult(
                    prediction=-1,
                    probability=0.0,
                    confidence=0.0,
                    backend=InferenceBackend.MODAL,
                    model_id=model_id,
                    error="Modal not configured"
                )
            
            # In production, call actual Modal inference function
            # This is a placeholder
            logger.info(f"Running prediction on Modal for model: {model_id}")
            
            result = InferenceResult(
                prediction=1,
                probability=0.7,
                confidence=0.4,
                backend=InferenceBackend.MODAL,
                model_id=model_id,
                metadata={"remote": True}
            )
            
            result.latency_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Modal prediction error: {e}", exc_info=True)
            return InferenceResult(
                prediction=-1,
                probability=0.0,
                confidence=0.0,
                backend=InferenceBackend.MODAL,
                model_id=model_id,
                error=str(e)
            )


class ClientSideInferenceEngine(BaseInferenceEngine):
    """Client-side inference (VPS only sends model + features)"""
    
    def __init__(self, models_dir: str = "/app/saved_models"):
        self.models_dir = Path(models_dir)
        self.available = True
    
    async def is_available(self) -> bool:
        """Client-side inference always available"""
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Health status"""
        models = list(self.models_dir.glob("*.pkl"))
        return {
            "status": "healthy",
            "available_models": len(models),
            "mode": "client-side"
        }
    
    async def predict(
        self,
        text: str,
        model_id: str
    ) -> InferenceResult:
        """Prepare data for client-side inference"""
        import time
        start_time = time.time()
        
        try:
            # Load model metadata
            model_path = self.models_dir / f"{model_id}.pkl"
            if not model_path.exists():
                model_path = self.models_dir / f"detector_{model_id}.pkl"
            
            if not model_path.exists():
                return InferenceResult(
                    prediction=-1,
                    probability=0.0,
                    confidence=0.0,
                    backend=InferenceBackend.CLIENT_SIDE,
                    model_id=model_id,
                    error=f"Model not found: {model_id}"
                )
            
            # Load metadata
            metadata = {}
            metadata_path = model_path.with_name(model_path.stem + "_metadata.pkl")
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
            
            # Return "pending" - client will download model and run inference
            result = InferenceResult(
                prediction=-1,  # Not yet predicted
                probability=0.0,
                confidence=0.0,
                backend=InferenceBackend.CLIENT_SIDE,
                model_id=model_id,
                metadata={
                    "model_url": f"/api/models/{model_id}/download",
                    "analysis_type": metadata.get("analysis_type"),
                    "text_processed": True,
                    "requires_client_inference": True
                }
            )
            
            result.latency_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Client-side prep error: {e}", exc_info=True)
            return InferenceResult(
                prediction=-1,
                probability=0.0,
                confidence=0.0,
                backend=InferenceBackend.CLIENT_SIDE,
                model_id=model_id,
                error=str(e)
            )


class InferenceRouter:
    """Route inference to appropriate backend based on model config"""
    
    def __init__(self, settings):
        self.settings = settings
        self.local_engine = LocalInferenceEngine(models_dir=settings.SAVED_MODELS_DIR)
        self.modal_engine = ModalInferenceEngine(
            settings.MODAL_TOKEN_ID,
            settings.MODAL_TOKEN_SECRET,
            settings.MODAL_WORKSPACE
        )
        self.client_engine = ClientSideInferenceEngine()
    
    async def predict(
        self,
        text: str,
        model_id: str,
        prefer_backend: Optional[InferenceBackend] = None
    ) -> InferenceResult:
        """
        Route to appropriate backend.
        Tries preferred backend first, then fallbacks.
        """
        
        # Get model config
        model_config = self.settings.AVAILABLE_MODELS.get(model_id)
        if not model_config:
            logger.warning(f"Unknown model: {model_id}, using default backend")
            # Convert string to enum if needed
            default_backend = self.settings.DEFAULT_INFERENCE_BACKEND
            if isinstance(default_backend, str):
                default_backend = InferenceBackend(default_backend)
            preferred = prefer_backend or default_backend
            fallbacks = [default_backend]
        else:
            preferred = prefer_backend or model_config.preferred_backend
            fallbacks = model_config.fallback_backends
        
        # Try preferred backend
        backend_name = preferred.value if hasattr(preferred, 'value') else str(preferred)
        logger.info(f"Attempting prediction with {backend_name} backend for model {model_id}")
        result = await self._try_backend(preferred, text, model_id)
        
        if result.error is None:
            return result
        
        # Try fallbacks
        for fallback in fallbacks:
            if fallback == preferred:
                continue  # Skip already tried
            
            fallback_name = fallback.value if hasattr(fallback, 'value') else str(fallback)
            logger.warning(f"Fallback to {fallback_name} backend (reason: {result.error})")
            result = await self._try_backend(fallback, text, model_id)
            
            if result.error is None:
                return result
        
        # All backends failed
        logger.error(f"All backends failed for model {model_id}")
        return result
    
    async def _try_backend(
        self,
        backend: InferenceBackend,
        text: str,
        model_id: str
    ) -> InferenceResult:
        """Try a specific backend"""
        
        try:
            if backend == InferenceBackend.LOCAL:
                if await self.local_engine.is_available():
                    return await self.local_engine.predict(text, model_id)
                else:
                    return InferenceResult(
                        prediction=-1, probability=0.0, confidence=0.0,
                        backend=backend, model_id=model_id,
                        error="Local engine not available"
                    )
            
            elif backend == InferenceBackend.MODAL:
                if await self.modal_engine.is_available():
                    return await self.modal_engine.predict(text, model_id)
                else:
                    return InferenceResult(
                        prediction=-1, probability=0.0, confidence=0.0,
                        backend=backend, model_id=model_id,
                        error="Modal not configured"
                    )
            
            elif backend == InferenceBackend.CLIENT_SIDE:
                if self.settings.ALLOW_CLIENT_SIDE_INFERENCE:
                    return await self.client_engine.predict(text, model_id)
                else:
                    return InferenceResult(
                        prediction=-1, probability=0.0, confidence=0.0,
                        backend=backend, model_id=model_id,
                        error="Client-side inference disabled"
                    )
        
        except Exception as e:
            logger.error(f"Backend {backend.value} error: {e}", exc_info=True)
            return InferenceResult(
                prediction=-1, probability=0.0, confidence=0.0,
                backend=backend, model_id=model_id,
                error=str(e)
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check all backends"""
        return {
            "local": await self.local_engine.health_check(),
            "modal": await self.modal_engine.health_check(),
            "client_side": await self.client_engine.health_check()
        }
