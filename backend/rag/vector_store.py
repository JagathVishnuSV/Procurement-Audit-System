"""
backend/rag/vector_store.py
──────────────────────────────────────────────────────────────────────────────
FAISS index management for contract clause embeddings.

Design
------
One FAISS IndexFlatIP (inner-product / cosine similarity) index holds all
contract chunks across all vendors.  Each vector maps to a payload dict
stored in a parallel list (self._payloads).

Persistence
-----------
  save()  – writes  <path>.index  +  <path>.payloads.json
  load()  – reads both files and restores in-memory state

Thread safety
--------------
A threading.Lock guards writes (add / save / reset).  Reads (search) are
lock-free because FAISS IndexFlatIP is read-safe concurrently.

Usage
-----
    from backend.rag.vector_store import get_vector_store
    vs = get_vector_store()
    vs.add_chunks(chunks, embeddings)       # add contract clauses
    results = vs.search("shipping fee", k=3)  # semantic search
    vs.save()
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from backend.config import get_settings


class ContractVectorStore:
    """
    FAISS-backed vector store for procurement contract clause embeddings.

    Attributes
    ----------
    _index    : faiss.IndexFlatIP  (cosine similarity via L2-normalised vectors)
    _payloads : list of dicts  {text, metadata}  in insertion order
    _dim      : embedding dimension (384 for all-MiniLM-L6-v2)
    """

    def __init__(self, index_path: str, dim: int = 384) -> None:
        import faiss
        self._path     = Path(index_path)
        self._dim      = dim
        self._lock     = threading.Lock()
        self._payloads: List[Dict[str, Any]] = []
        self._index    = faiss.IndexFlatIP(dim)
        logger.debug("ContractVectorStore initialised (dim={})", dim)

    # ── Population ─────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add clause chunks to the index.

        Parameters
        ----------
        chunks      : list of {"text": str, "metadata": dict} from chunker.py
        embeddings  : float32 array shape (len(chunks), self._dim)
                      Must already be L2-normalised for cosine similarity.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have equal length"
            )
        if not chunks:
            return

        vecs = embeddings.astype(np.float32)
        with self._lock:
            self._index.add(vecs)
            self._payloads.extend(chunks)

        logger.debug("Added {} chunks to FAISS (total={})", len(chunks), len(self._payloads))

    def remove_contract(self, contract_id: str) -> int:
        """
        Remove all vectors belonging to a contract and rebuild the index.

        Returns the number of vectors removed.

        Note: FAISS IndexFlatIP does not support incremental deletion,
        so a full rebuild is necessary.  On datasets < 100k clauses this
        is fast (< 1 second).
        """
        import faiss

        with self._lock:
            keep_mask = [
                p["metadata"].get("contract_id") != contract_id
                for p in self._payloads
            ]
            n_removed = sum(1 for m in keep_mask if not m)
            if n_removed == 0:
                return 0

            kept_payloads = [p for p, k in zip(self._payloads, keep_mask) if k]
            new_index = faiss.IndexFlatIP(self._dim)

            if kept_payloads:
                kept_texts = [p["text"] for p in kept_payloads]
                from backend.rag.embedder import get_embedder
                emb = get_embedder()
                kept_vecs = emb.embed_texts(kept_texts)
                new_index.add(kept_vecs)

            self._payloads = kept_payloads
            self._index    = new_index

        logger.info(
            "Removed {} vectors for contract_id={} (remaining={})",
            n_removed, contract_id, len(self._payloads),
        )
        return n_removed

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        k: int = 5,
        contract_id: Optional[str] = None,
        score_threshold: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across all indexed contract clauses.

        Parameters
        ----------
        query_text      : natural language query
        k               : max results to return
        contract_id     : restrict results to one contract (optional)
        score_threshold : minimum cosine similarity to include

        Returns
        -------
        List of result dicts, sorted by descending score:
            {"text": str, "metadata": dict, "score": float}
        """
        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no search results")
            return []

        from backend.rag.embedder import get_embedder
        emb       = get_embedder()
        query_vec = emb.embed_query(query_text)

        # Fetch more than k so we can post-filter by contract_id
        fetch_k = min(k * 4, self._index.ntotal) if contract_id else min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, fetch_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < score_threshold:
                continue
            payload = self._payloads[idx]
            if contract_id and payload["metadata"].get("contract_id") != contract_id:
                continue
            results.append({
                "text":     payload["text"],
                "metadata": payload["metadata"],
                "score":    round(float(score), 4),
            })
            if len(results) >= k:
                break

        return results

    def search_by_contract(
        self,
        query_text: str,
        contract_id: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper: search restricted to one contract."""
        return self.search(query_text, k=k, contract_id=contract_id)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist FAISS index + payloads to disk."""
        import faiss

        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            faiss.write_index(self._index, str(self._path) + ".index")
            payload_path = str(self._path) + ".payloads.json"
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(self._payloads, f, ensure_ascii=False)

        logger.success(
            "FAISS saved: {} vectors → {}", self._index.ntotal, self._path
        )

    def load(self) -> bool:
        """
        Load persisted index from disk.

        Returns True if loaded successfully, False if no saved state exists.
        """
        import faiss

        index_file   = str(self._path) + ".index"
        payload_file = str(self._path) + ".payloads.json"

        if not Path(index_file).exists():
            logger.info("No saved FAISS index found at '{}' — starting fresh", index_file)
            return False

        with self._lock:
            self._index = faiss.read_index(index_file)
            with open(payload_file, "r", encoding="utf-8") as f:
                self._payloads = json.load(f)

        logger.success(
            "FAISS loaded: {} vectors from '{}'", self._index.ntotal, index_file
        )
        return True

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    def contract_chunk_count(self, contract_id: str) -> int:
        """Count how many indexed chunks belong to a specific contract."""
        return sum(
            1 for p in self._payloads
            if p["metadata"].get("contract_id") == contract_id
        )

    def list_contract_ids(self) -> List[str]:
        """Return unique contract_ids present in the index."""
        return list({p["metadata"].get("contract_id", "") for p in self._payloads})

    def reset(self) -> None:
        """Clear all vectors and payloads from the in-memory index."""
        import faiss
        with self._lock:
            self._index    = faiss.IndexFlatIP(self._dim)
            self._payloads = []
        logger.info("FAISS index reset (all vectors cleared)")


@lru_cache(maxsize=1)
def get_vector_store() -> ContractVectorStore:
    """
    Return the global ContractVectorStore singleton.

    Automatically loads any previously persisted index from disk.
    """
    settings = get_settings()
    from backend.rag.embedder import get_embedder
    embedder = get_embedder()
    store    = ContractVectorStore(
        index_path=settings.FAISS_INDEX_PATH,
        dim=embedder.dim,
    )
    store.load()
    return store
