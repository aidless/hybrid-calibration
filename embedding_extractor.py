"""
LLM Embedding Extraction Module.

Extracts frozen embeddings from a pretrained model.
Supports both sentence-transformers (MiniLM) and HuggingFace transformers.

CPU-optimized: small batches, fp32, no gradient. Offline-first.
"""

import os
import ssl
import pickle
import hashlib
import numpy as np
from typing import List, Optional

# Force offline mode BEFORE any HF imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

# SSL workaround for Windows
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import torch

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    BATCH_SIZE_EMBED,
    DATA_DIR,
)


class EmbeddingExtractor:
    """Extract frozen LLM embeddings for downstream tree ensembles."""

    # Map model names to local cache paths (for offline loading)
    _MODEL_PATH_MAP = {
        "BAAI/bge-small-zh-v1.5": [
            "models/bge",  # Project-local (portable)
            "~/.cache/modelscope/BAAI/bge-small-zh-v1___5",  # ModelScope
            "~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/*",
        ],
        "sentence-transformers/all-MiniLM-L6-v2": [
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
    }

    def _resolve_path(self, model_name: str) -> str:
        """Resolve model name to actual filesystem path."""
        if model_name in self._MODEL_PATH_MAP:
            candidates = self._MODEL_PATH_MAP[model_name]
            for candidate in candidates:
                # Expand glob patterns
                import glob
                if '*' in candidate:
                    matches = sorted(glob.glob(os.path.expanduser(candidate)))
                    for m in matches:
                        if os.path.isdir(m):
                            return m
                else:
                    path = os.path.expanduser(candidate)
                    if not os.path.isabs(path):
                        path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), path
                        )
                    if os.path.isdir(path):
                        return path
        return model_name  # Fallback: let HF try

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = "cpu",
        max_length: int = MAX_SEQ_LENGTH,
        batch_size: int = BATCH_SIZE_EMBED,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        load_path = self._resolve_path(model_name)
        print(f"[EmbeddingExtractor] Loading {model_name} from {load_path} on {device} ...")

        # Load with transformers AutoModel (works for both BGE and MiniLM)
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            load_path, local_files_only=True
        )
        self.model.to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embedding_dim = self.model.config.hidden_size
        print(f"[EmbeddingExtractor] dim={self.embedding_dim}, "
              f"type={type(self.model).__name__}")

    def extract(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings via mean-pooling of last hidden state.

        Returns:
            embeddings: np.ndarray of shape (n_texts, embedding_dim)
        """
        all_embeddings = []
        n = len(texts)

        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                if show_progress and i % (self.batch_size * 10) == 0:
                    print(f"  Embedding: {min(i + self.batch_size, n)}/{n}")

                encoded = self.tokenizer(
                    list(batch_texts),
                    padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state  # [batch, seq, dim]

                # Mean pooling over non-padding tokens
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def extract_with_cache(
        self,
        texts: List[str],
        dataset_name: str,
        split_name: str,
    ) -> np.ndarray:
        """
        Extract embeddings with disk cache.
        """
        cache_dir = os.path.join(DATA_DIR, "embeddings")
        os.makedirs(cache_dir, exist_ok=True)

        text_hash = hashlib.md5(
            "".join(texts[:100] + texts[-100:]).encode()
        ).hexdigest()[:8]
        model_short = self.model_name.replace("/", "_").replace("-", "_")
        cache_name = f"{dataset_name}_{split_name}_{model_short}_{text_hash}_{len(texts)}.npy"
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            print(f"[EmbeddingExtractor] Loading cached: {cache_path}")
            return np.load(cache_path)

        embeddings = self.extract(texts)
        np.save(cache_path, embeddings)
        print(f"[EmbeddingExtractor] Saved cache: {cache_path}")
        return embeddings


def build_feature_matrix(
    X_embeddings: np.ndarray,
    X_tabular: Optional[np.ndarray] = None,
    feature_set: str = "hybrid",
):
    """
    Combine embedding and tabular features based on feature_set.
    """
    parts = []

    if feature_set in ("embeddings", "hybrid"):
        parts.append(X_embeddings)

    if feature_set in ("tabular", "hybrid"):
        if X_tabular is None:
            raise ValueError(f"feature_set='{feature_set}' but X_tabular is None")
        parts.append(X_tabular)

    if not parts:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    return np.concatenate(parts, axis=1).astype(np.float32)
