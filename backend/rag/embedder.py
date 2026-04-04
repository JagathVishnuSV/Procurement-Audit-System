"""
backend/rag/embedder.py
──────────────────────────────────────────────────────────────────────────────
Singleton embedding wrapper around sentence-transformers all-MiniLM-L6-v2.

Why all-MiniLM-L6-v2?
- 384-dimensional dense vectors: small FAISS index, fast ANN search.
- Runs entirely locally — no API keys needed for RAG.
- Strong semantic recall on short clause snippets (50-200 tokens).

Usage
-----
    from backend.rag.embedder import get_embedder
    emb = get_embedder()
    vectors = emb.embed_texts(["Pricing clause text...", "Delivery terms..."])
    # → np.ndarray shape (2, 384) dtype float32
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import List

import numpy as np
from loguru import logger

from backend.config import get_settings

_lock = threading.Lock()


class ClauseEmbedder:
    """
    Thread-safe singleton that wraps SentenceTransformer.

    The model is loaded exactly once (on first call to `get_embedder()`).
    All subsequent calls return the same object without re-loading.
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model '{}'…", model_name)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.success(
            "Embedding model ready: '{}' | dim={}", model_name, self._dim
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        """Embedding dimensions (384 for all-MiniLM-L6-v2)."""
        return self._dim

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of text strings.

        Parameters
        ----------
        texts       : input strings (contract clause chunks)
        batch_size  : mini-batch size for the transformer
        normalize   : L2-normalise vectors (enables cosine similarity via dot product)

        Returns
        -------
        np.ndarray  shape (len(texts), self.dim)  dtype float32
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray  shape (1, self.dim)  dtype float32
        """
        return self.embed_texts([text])


@lru_cache(maxsize=1)
def get_embedder() -> ClauseEmbedder:
    """
    Return the global ClauseEmbedder singleton (lru_cache so loads once).

    This is safe to call at module import time or inside a FastAPI lifespan
    handler.  The first call triggers model download (~92 MB) which is then
    cached locally by HuggingFace.
    """
    settings = get_settings()
    return ClauseEmbedder(model_name=settings.EMBEDDING_MODEL)
