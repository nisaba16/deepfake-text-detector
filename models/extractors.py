from typing import Dict, List, Union
import numpy as np
import torch
import os
import psutil
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional

# Optional sklearn import guarded for environments without it at runtime
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None  # type: ignore

### ------------- model to extract embedding from decoder or encoder language models
class EmbeddingExtractor(torch.nn.Module):
    """
    Extract hidden representation of LLMs, using only forward pass
    outputs is a a vector for each text
    """

    def __init__(self, model_name="distilbert-base-uncased", device=None, use_flash_attention=False, is_qwen3=None, log_memory: bool = False, memory_interval: int = 1):
        super().__init__()
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.log_memory = log_memory
        self.memory_interval = max(1, memory_interval)

        if is_qwen3 is None:
            self.is_qwen3 = 'qwen3' in model_name.lower() and 'embedding' in model_name.lower()
        else:
            self.is_qwen3 = is_qwen3

        padding_side = 'left' if self.is_qwen3 else 'right'

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)

        # Check if this is an encoder-decoder model (e.g., M2M-100, mBART)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        is_encoder_decoder = getattr(config, 'is_encoder_decoder', False)
        
        if is_encoder_decoder:
            print(f"Detected encoder-decoder model. Loading encoder only.")
            # For encoder-decoder models, load only the encoder
            from transformers import AutoModel as OriginalAutoModel
            full_model = OriginalAutoModel.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map=None,
                use_safetensors=True
            )
            # Extract just the encoder
            self.model = full_model.encoder
        else:
            # Force use of safetensors to bypass PyTorch 2.6 security requirement
            if use_flash_attention and self.device == 'cuda':
                print("Loading with Flash Attention 2...")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    device_map=None,
                    use_safetensors=True
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map=None,
                    use_safetensors=True
                )
        
        self.model = self.model.to(self.device)
        self.model.eval()

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            print('adding pasd token')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully. Hidden size: {self.model.config.hidden_size}")
        print(f"Number of layers: {self.model.config.num_hidden_layers}")

    def test_model(self, sample_text="This is a test sentence."):
        """
        Test if the model works with a simple text
        """
        print("Testing model with sample text...")
        try:
            embeddings = self.get_all_layer_embeddings(sample_text)
            print(f"Model test successful!")
            print(f"Number of layers: {len(embeddings)}")
            print(f"Embedding shape for layer -1: {embeddings[-1].shape}")
            return True
        except Exception as e:
            print(f" Model test failed: {e}")
            return False
        
    def get_all_layer_embeddings(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 16,
        max_length: int = 8192,  # Support long context
        show_progress=True
    ) -> List[Dict[int, np.ndarray]]:
        """
        Extract embeddings from ALL layers for one or more texts, in batches.
        we want as output only vectors that wont be batched anymore so this must also remove the padding made by tokenizer
        
        Args:
            texts: list[str] or str
            batch_size: number of texts per forward pass
            max_length: maximum sequence length
            
        Returns:
        all_texts_embeddings
                [
            {0: emb_layer0, 1: emb_layer1, ..., N: emb_layerN},   # embeddings for text 1
            {0: emb_layer0, 1: emb_layer1, ..., N: emb_layerN},   # embeddings for text 2
            ...
            ]

        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_texts_embeddings = []


        iterator = range(0, len(texts), batch_size)
        if show_progress and len(texts) > batch_size:
            iterator = tqdm(iterator, desc="Extracting embeddings")
        for batch_index, i in enumerate(iterator):
            batch = texts[i:i+batch_size]
            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                
            )
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            with torch.no_grad():
                outputs = self.model(**model_inputs, output_hidden_states=True)

            # Get hidden states (we now load only encoders, so always use outputs.hidden_states)
            hidden_states_tuple = outputs.hidden_states

            # Memory logging (GPU + CPU) after forward pass
            if self.log_memory and (batch_index % self.memory_interval == 0):
                self._log_memory_usage(batch_index, len(batch), model_inputs)

            attention_mask = model_inputs['attention_mask']
            
            # For each text in the batch
            for b in range(len(batch)):
                    text_layer_embeddings = {}
                    valid_indices = torch.where(attention_mask[b].bool())[0]
                    
                    if len(valid_indices) == 0:
                        print("attenton no valid indices")
                        print(outputs.hidden_states[b].shape)
                        print(f"Text: {repr(batch[b])}")
                        print(attention_mask[b].bool())

                    for layer_idx, hidden_state in enumerate(hidden_states_tuple):
                        
                        # Keep full sequence, no pooling
                        valid_tokens = hidden_state[b][valid_indices]
                        text_layer_embeddings[layer_idx] = valid_tokens.cpu().numpy()  # (seq, hidden)
                    
                    all_texts_embeddings.append(text_layer_embeddings)

        assert len(all_texts_embeddings) == len(texts)
        return all_texts_embeddings

    def _log_memory_usage(self, batch_index: int, batch_size: int, model_inputs: Dict[str, torch.Tensor]):
        """Log current memory usage (GPU + CPU) and approximate tensor footprint for embeddings.
        Args:
            batch_index: index of current batch
            batch_size: number of texts in batch
            model_inputs: tokenized input tensors
        """
        # CPU memory
        process = psutil.Process(os.getpid())
        rss_gb = process.memory_info().rss / (1024 ** 3)

        gpu_mem = None
        max_gpu = None
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            try:
                device_index = torch.device(self.device).index or 0
                gpu_mem = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
                max_gpu = torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)
            except Exception:
                pass

        # Estimate current hidden states footprint (they are on device in outputs.hidden_states)
        hidden_states_bytes = 0
        try:
            for hs in self.model._last_output.hidden_states:  # Not standard; fallback below if fails
                hidden_states_bytes += hs.element_size() * hs.nelement()
        except Exception:
            # We don't have stored hidden states; just approximate using embedding dim * seq len * layers
            try:
                seq_len = int(model_inputs['input_ids'].shape[1])
                layers = int(self.model.config.num_hidden_layers + 1)  # + embedding layer
                hidden_dim = int(self.model.config.hidden_size)
                dtype_size = torch.tensor([], dtype=torch.float16).element_size()
                hidden_states_bytes = seq_len * layers * hidden_dim * batch_size * dtype_size
            except Exception:
                hidden_states_bytes = 0

        hidden_states_gb = hidden_states_bytes / (1024 ** 3)

        msg = f"[Memory] Batch {batch_index} | batch_size={batch_size} | CPU RSS={rss_gb:.2f} GB"
        if gpu_mem is not None:
            msg += f" | GPU alloc={gpu_mem:.2f} GB (max {max_gpu:.2f} GB)"
        msg += f" | approx hidden_states={hidden_states_gb:.2f} GB"
        print(msg)

    # ---------------- Memory efficient single-layer pooled extraction -----------------
    def get_pooled_layer_embeddings(
        self,
        texts: Union[str, List[str]],
        layer_idx: int,
        pooling: str = "mean",
        batch_size: int = 16,
        max_length: int = 8192,
        show_progress: bool = True,
        return_attention: bool = False,
        normalize: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """Return pooled embeddings for a single layer without storing all layers.

        This is substantially more memory efficient than get_all_layer_embeddings because
        it only keeps the chosen layer for each batch and immediately discards the rest.

        Args:
            texts: list or single string
            layer_idx: which hidden state index to use (0..num_layers). Matches HF hidden_states indexing.
            pooling: one of mean|max|first|last|attn (attention-weighted mean)
            batch_size: batch size for forward pass
            max_length: maximum tokens per text
            show_progress: show tqdm progress bar
            return_attention: when pooling='attn', also return the per-text attention weights vector used for pooling
            normalize: if True, L2-normalize each token embedding before pooling (default: False), the same
                features as l2_normalize_tokens + pool_embeds_from_layer
        Returns:
            If return_attention is False: np.ndarray shape (num_texts, hidden_size)
            If return_attention is True and pooling='attn': (embeds: np.ndarray (N, H), weights: List[np.ndarray])
        """
        print(pooling)
        if isinstance(texts, str):
            texts = [texts]

        pooled_vectors: List[np.ndarray] = []
        attn_weights_list: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress and len(texts) > batch_size:
            iterator = tqdm(iterator, desc="Extracting pooled layer embeddings")

        for batch_index, i in enumerate(iterator):
            batch = texts[i:i + batch_size]
            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            with torch.no_grad():
                want_attn = pooling in {"attn", "attention", "attn_mean", "attn_weighted"}
                try:
                    outputs = self.model(**model_inputs, output_hidden_states=True, output_attentions=want_attn)
                except TypeError:
                    # Some models may not accept output_attentions kwarg
                    outputs = self.model(**model_inputs, output_hidden_states=True)

            # Get hidden states (we now load only encoders, so always use outputs.hidden_states)
            hidden_states = outputs.hidden_states
            if layer_idx >= len(hidden_states):
                raise ValueError(f"Requested layer_idx={layer_idx} but only {len(hidden_states)} hidden states returned")

            chosen = hidden_states[layer_idx]  # (batch, seq, hidden)
            attention_mask = model_inputs['attention_mask']  # (batch, seq)

            # Attentions may correspond to transformer layers only (no embedding layer). Map index when needed.
            attentions = getattr(outputs, "attentions", None) if 'want_attn' in locals() and want_attn else None
            # Map hidden_states index to attentions index: hidden_states[0] is embeddings; attentions[0] is layer 0
            attn_tensor = None
            if attentions is not None and isinstance(attentions, (tuple, list)) and len(attentions) > 0:
                if layer_idx == 0:
                    attn_tensor = None  # no attention for embedding layer
                else:
                    attn_idx = min(layer_idx - 1, len(attentions) - 1)
                    attn_tensor = attentions[attn_idx]  # (batch, heads, seq, seq)

            for b in range(chosen.size(0)):
                valid = attention_mask[b].bool()
                token_reps = chosen[b][valid]  # (valid_seq, hidden)
                if normalize:
                    # Per token, before pooling, in fp32 (fp16 norms overflow)
                    token_reps = F.normalize(token_reps.float(), p=2, dim=-1)
                if token_reps.shape[0] == 0:
                    vec = torch.zeros(chosen.size(-1), device=chosen.device)
                    w_vec = None
                else:
                    # Normalize pooling key to unify synonyms
                    pooling_key = pooling.lower()
                    if pooling_key in {"attn_mean", "attn_weighted", "attention"}:
                        pooling_key = "attn"

                    if pooling_key == 'mean':
                        vec = token_reps.mean(dim=0)
                    elif pooling_key == 'max':
                        vec, _ = token_reps.max(dim=0)
                    elif pooling_key == 'first':
                        vec = token_reps[0]
                    elif pooling_key == 'last':
                        vec = token_reps[-1]
                    elif pooling_key == 'mean_std':
                        # Concatenate mean and std over tokens -> shape (2*hidden,)
                        # Compute in float32 end-to-end to avoid fp16 overflow when converting std
                        mu = token_reps.float().mean(dim=0)
                        sigma = token_reps.float().std(dim=0, unbiased=False)
                        vec = torch.cat([mu, sigma], dim=0)  # keep float32
                    elif pooling_key in {"statistical", "covariance", "cov"}:
                        # Covariance pooling (upper-triangular flatten). Beware of dimensionality: H*(H+1)/2
                        # To avoid explosions on large H, cap to max hidden size via env var COV_MAX_HIDDEN (default 1024).
                        H = token_reps.shape[-1]
                        max_h = int(os.environ.get("COV_MAX_HIDDEN", "1024"))
                        X = token_reps.float()
                        Xc = X - X.mean(dim=0, keepdim=True)
                        if H <= max_h:
                            # Compute full covariance and flatten upper triangle
                            C = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))  # (H, H) float32
                            iu = torch.triu_indices(H, H)
                            vec = C[iu[0], iu[1]]  # keep float32
                        else:
                            # Fallback: use only the diagonal (variances) when hidden size is too large
                            var = Xc.var(dim=0, unbiased=False)
                            vec = var  # float32
                    elif pooling_key == "attn" and attn_tensor is not None:
                        # attn_tensor: (heads, seq, seq) for this sample
                        a = attn_tensor[b]
                        # Average across heads and query positions -> per-key weights (seq,)
                        # Mask out padding positions
                        w = a.mean(dim=0).mean(dim=0)  # (seq,)
                        w_valid = w[valid]
                        # Normalize to sum=1 with numerical safety
                        w_sum = w_valid.sum()
                        if torch.isfinite(w_sum) and w_sum > 0:
                            w_norm = w_valid / w_sum
                        else:
                            w_norm = torch.full_like(w_valid, 1.0 / max(1, w_valid.numel()))
                        vec = (token_reps * w_norm.unsqueeze(-1)).sum(dim=0)
                        w_vec = w_norm.detach().cpu().float().numpy()
                    elif pooling_key == "attn" and attn_tensor is None:
                        # Graceful fallback when attentions are unavailable (e.g., sdpa or flash attention)
                        # Use mean pooling and return uniform weights if requested
                        vec = token_reps.mean(dim=0)
                        seq_len = int(valid.sum().item())
                        if seq_len > 0:
                            w_vec = np.full((seq_len,), 1.0 / seq_len, dtype=np.float32)
                        else:
                            w_vec = None
                    else:
                        raise ValueError(f"Unsupported pooling: {pooling}")
                pooled_vectors.append(vec.detach().cpu().float().numpy())
                if return_attention and pooling.lower() in {"attn", "attention", "attn_mean", "attn_weighted"}:
                    # If attention not available, append a uniform vector matching valid tokens
                    if w_vec is None:
                        seq_len = int(valid.sum().item())
                        if seq_len == 0:
                            attn_weights_list.append(np.array([], dtype=np.float32))
                        else:
                            attn_weights_list.append(np.full((seq_len,), 1.0 / seq_len, dtype=np.float32))
                    else:
                        attn_weights_list.append(w_vec.astype(np.float32))

            # Optional memory logging
            if self.log_memory and (batch_index % self.memory_interval == 0):
                self._log_memory_usage(batch_index, len(batch), model_inputs)

            # Explicit cleanup
            del outputs, hidden_states, chosen, model_inputs
            if torch.cuda.is_available() and 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        embeds = np.vstack(pooled_vectors).astype(np.float32)
        
        if return_attention and pooling.lower() in {"attn", "attention", "attn_mean", "attn_weighted"}:
            return embeds, attn_weights_list
        return embeds

#### --------------------------------------------- from embeddings generate features for classification --------------------------------------------

def extract_embed_at_layer(list_embeds, layer_idx):
    """
    Args:
        list_embeds: obtained from EmbeddingExtractor
            [
            {0: emb_layer0, 1: emb_layer1, ..., N: emb_layerN},   # embeddings for text 1
            {0: emb_layer0, 1: emb_layer1, ..., N: emb_layerN},   # embeddings for text 2
            ...
            ]
        layer_idx: int for the layer
    Returns:
        List of numpy arrays: all hidden states per text at a specific layer  
            emb_layerk_textk has shape shape (seq,hidden_size)
                (here for each text seq will be different)
            [ emb_layerk_text0, ... , emb_layerk_textk, ..., emb_layerk_lasttext ]
    """

    if layer_idx not in list_embeds[0]:
        raise ValueError(f"Layer {layer_idx} not found. Available: {list(list_embeds[0].keys())}")
    
    selected_embeds_at_choosen_layer = [list_embeds_per_text[layer_idx] for list_embeds_per_text in list_embeds]

    return selected_embeds_at_choosen_layer

def l2_normalize_tokens(selected_embeds_at_choosen_layer):
    """L2-normalize every token vector of every text, in fp32 (fp16 norms overflow).

    The normalization of get_pooled_layer_embeddings(normalize=True), for the pool_embeds_from_layer
    path, so the memory-efficient and full extraction paths produce the same features.
    """
    return [F.normalize(torch.from_numpy(emb).float(), p=2, dim=-1).numpy()
            for emb in selected_embeds_at_choosen_layer]

def pool_embeds_from_layer(selected_embeds_at_choosen_layer, pooling='mean'):
    """
    Args:
        selected_embeds_at_choosen_layer: obtained from EmbeddingExtractor
            [ emb_layerk_text0, ... , emb_layerk_textk, ..., emb_layerk_lasttext]
        pooling: method to pool vectors

    Returns:
        numpy array of shape (num_texts, hidden_size)

    """
    list_pooled = []
    for idx, emb in enumerate(selected_embeds_at_choosen_layer):   # emb shape: (seq, hidden)
        if emb.size == 0:
            print(f"⚠️  Warning: Empty embedding at index {idx}, using zeros")
            print(emb)
            pooled_emb = np.zeros(emb.shape[1] if len(emb.shape) > 1 else 768)
        elif np.isnan(emb).all():
            print(f"⚠️  Warning: All-NaN embedding at index {idx}, using zeros")
            pooled_emb = np.zeros(emb.shape[1] if len(emb.shape) > 1 else 768)
        # Normalize pooling key and treat attention-based requests as mean in this stateless context
        pooling_key = str(pooling).lower()
        if pooling_key in {"attn", "attention", "attn_mean", "attn_weighted"}:
            pooling_key = 'mean'

        # Compute in float32 to avoid fp16 overflow in variance/std paths
        X = emb.astype(np.float32, copy=False)

        if pooling_key == 'mean':
            pooled_emb = X.mean(axis=0)  # (hidden,)
        elif pooling_key == 'max':
            pooled_emb = X.max(axis=0)  # (hidden,)
        elif pooling_key == 'first':
            pooled_emb = X[0]  # (hidden,)
        elif pooling_key == 'last':
            pooled_emb = X[-1]  # (hidden,)
        elif pooling_key == 'mean_std':
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)  # ddof=0 (population) matches unbiased=False above
            pooled_emb = np.concatenate([mu, sigma], axis=0).astype(np.float32, copy=False)  # (2*hidden,)
        elif pooling_key in {'statistical', 'covariance', 'cov'}:
            # Covariance pooling (upper triangular flatten) with safety cap
            H = X.shape[-1]
            max_h = int(os.environ.get("COV_MAX_HIDDEN", "1024"))
            Xc = X - X.mean(axis=0, keepdims=True)
            if H <= max_h:
                C = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))  # (H, H)
                iu = np.triu_indices(H)
                pooled_emb = C[iu].astype(np.float32, copy=False)
            else:
                # Fallback to diagonal (variances) if hidden size too large
                pooled_emb = Xc.var(axis=0).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        
        list_pooled.append(pooled_emb)
    
    return np.array(list_pooled)




if __name__ == "__main__":
    print("EmbeddingExtractor module: run your own test in scripts using the new memory-efficient methods.")


# ------------------------------ Classical NLP feature extractor (TF-IDF) ------------------------------
class TFIDFExtractor:
    """
    Lightweight wrapper around sklearn's TfidfVectorizer to plug TF-IDF features
    into the existing classifier pipeline.

    Notes:
    - Returns a scipy.sparse CSR matrix by default to be memory efficient.
      Set dense=True in transform/fit_transform to get a dense float32 ndarray
      if your downstream classifier requires dense inputs.
    - Keep max_features modest if you plan to densify (e.g., <= 10-20k).
    """

    def __init__(
        self,
        max_features: Optional[int] = 20000,
        ngram_range=(1, 2),
        lowercase: bool = True,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        use_idf: bool = True,
        norm: Optional[str] = "l2",
        sublinear_tf: bool = False,
        stop_words: Optional[str | list[str]] = None,
    ) -> None:
        if TfidfVectorizer is None:
            raise ImportError("scikit-learn is required for TFIDFExtractor but was not found.")

        # Sanitize df parameters to satisfy sklearn's type constraints
        # min_df: float in [0,1] or int in [1, inf)
        # max_df: float in (0,1] or int in [1, inf)
        min_df_val = min_df
        max_df_val = max_df
        try:
            # If user passed a float like 2.0, cast to int 2
            if isinstance(min_df_val, float) and min_df_val >= 1.0:
                if abs(min_df_val - round(min_df_val)) < 1e-9:
                    min_df_val = int(round(min_df_val))
            # If user passed a float > 1.0 for max_df, try to interpret as integer count if near-integer
            if isinstance(max_df_val, float) and max_df_val > 1.0:
                if abs(max_df_val - round(max_df_val)) < 1e-9:
                    max_df_val = int(round(max_df_val))
                else:
                    # Fallback: cap to 1.0 to keep as valid fraction
                    max_df_val = 1.0
        except Exception:
            # On any sanitization error, fall back to original values
            min_df_val = min_df
            max_df_val = max_df

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=lowercase,
            min_df=min_df_val,
            max_df=max_df_val,
            use_idf=use_idf,
            norm=norm,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words,
        )
        self._is_fitted = False

    def fit(self, texts: List[str]):
        """Fit the TF-IDF vocabulary/statistics on a corpus of raw texts."""
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: List[str], dense: bool = False):
        """Transform texts to TF-IDF features.

        Args:
            texts: list of raw text strings
            dense: if True, returns a dense np.ndarray[float32]. If False (default), returns CSR sparse matrix.
        """
        if not self._is_fitted:
            raise ValueError("TFIDFExtractor not fitted. Call fit() or fit_transform() first.")
        X = self.vectorizer.transform(texts)
        if dense:
            return X.toarray().astype(np.float32)
        return X

    def fit_transform(self, texts: List[str], dense: bool = False):
        """Fit on texts and return the TF-IDF feature matrix.

        Args:
            texts: list of raw text strings
            dense: if True, returns a dense np.ndarray[float32]. If False (default), returns CSR sparse matrix.
        """
        X = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        if dense:
            return X.toarray().astype(np.float32)
        return X

    def get_feature_names(self) -> List[str]:
        return list(self.vectorizer.get_feature_names_out())

    def vocabulary_size(self) -> int:
        voc = getattr(self.vectorizer, "vocabulary_", None)
        return len(voc) if voc is not None else 0
