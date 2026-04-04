"""
backend/rag/clm_service.py
──────────────────────────────────────────────────────────────────────────────
Contract Lifecycle Management (CLM) service.

Orchestrates the full PDF ingestion pipeline:
  1. Hash the PDF (dedup guard)
  2. Persist the file to disk
  3. Create a Contract DB row (status=DRAFT)
  4. Chunk the PDF with chunker.py
  5. Embed chunks with embedder.py
  6. Insert vectors into FAISS vector_store.py
  7. Mark Contract row is_indexed=True, update chunk_count

Also exposes:
  - search_clauses(query, contract_id=None, k=5) → semantic search
  - reindex_contract(contract_id) → re-embed an existing contract
  - delete_contract_vectors(contract_id) → remove from FAISS

Usage
-----
    from backend.rag.clm_service import CLMService
    svc = CLMService()
    contract = await svc.ingest_pdf(
        pdf_bytes=file_bytes,
        filename="vendor_a.pdf",
        vendor_id=uuid,
        title="IT Services Agreement FY2025",
        session=async_db_session,
    )
"""

from __future__ import annotations

import hashlib
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.config import get_settings
from backend.models.contract import Contract, ContractStatus


# Directory where PDF files are stored on disk
_CONTRACT_STORE = Path("data/contracts")


class CLMService:
    """
    Stateless service class – safe to instantiate per-request or as a
    FastAPI dependency singleton.
    """

    def __init__(self) -> None:
        self._settings  = get_settings()
        _CONTRACT_STORE.mkdir(parents=True, exist_ok=True)

    # ── Ingestion ───────────────────────────────────────────────────────────────

    async def ingest_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        vendor_id: _uuid.UUID,
        title: str,
        session: AsyncSession,
        contract_number: Optional[str] = None,
        effective_date: Optional[datetime] = None,
        expiry_date: Optional[datetime] = None,
        total_value: Optional[float] = None,
        description: Optional[str] = None,
    ) -> Contract:
        """
        Full ingestion pipeline: PDF bytes → chunked FAISS vectors + DB row.

        Returns the created/updated Contract ORM object.
        """
        # 1. Dedup: check for identical PDF already indexed
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        existing = await self._find_by_hash(file_hash, session)
        if existing:
            logger.info(
                "PDF '{}' already indexed as contract_id={} (hash match)",
                filename, existing.id,
            )
            return existing

        # 2. Persist PDF to disk
        contract_id = _uuid.uuid4()
        safe_name   = f"{contract_id}_{Path(filename).name}"
        file_path   = _CONTRACT_STORE / safe_name
        file_path.write_bytes(pdf_bytes)
        logger.info("PDF saved → {}", file_path)

        # 3. Create Contract DB row (status=DRAFT until indexed)
        contract = Contract(
            id              = contract_id,
            vendor_id       = vendor_id,
            title           = title,
            contract_number = contract_number,
            file_path       = str(file_path.relative_to(Path("."))),
            file_hash       = file_hash,
            upload_date     = datetime.now(timezone.utc),
            effective_date  = effective_date,
            expiry_date     = expiry_date,
            total_value     = total_value,
            description     = description,
            status          = ContractStatus.DRAFT,
            faiss_index_id  = str(contract_id),
            embedding_model = self._settings.EMBEDDING_MODEL,
            is_indexed      = False,
        )
        session.add(contract)
        await session.flush()   # get the DB row written (but not committed yet)

        # 4. Chunk + 5. Embed + 6. FAISS insert
        chunk_count = await self._embed_and_index(
            pdf_bytes   = pdf_bytes,
            filename    = filename,
            contract_id = str(contract_id),
        )

        # 7. Update Contract row
        contract.chunk_count = chunk_count
        contract.is_indexed  = True
        contract.status      = ContractStatus.ACTIVE
        await session.flush()

        logger.success(
            "Contract '{}' indexed: {} chunks — contract_id={}",
            title, chunk_count, contract_id,
        )
        return contract

    async def ingest_text(
        self,
        text: str,
        vendor_id: _uuid.UUID,
        title: str,
        session: AsyncSession,
        source_name: str = "text_upload",
        **kwargs: Any,
    ) -> Contract:
        """
        Ingest plain text (e.g. seeded sample contracts) instead of a PDF.
        Uses chunk_text() directly, skipping the PyPDF step.
        """
        # Dedup: raise if identical text was already indexed
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = await self._find_by_hash(text_hash, session)
        if existing:
            raise ValueError(
                f"Text already indexed as contract_id={existing.id} (hash match)"
            )

        contract_id = _uuid.uuid4()

        contract = Contract(
            id              = contract_id,
            vendor_id       = vendor_id,
            title           = title,
            file_hash       = text_hash,
            upload_date     = datetime.now(timezone.utc),
            status          = ContractStatus.DRAFT,
            faiss_index_id  = str(contract_id),
            embedding_model = self._settings.EMBEDDING_MODEL,
            is_indexed      = False,
            **{k: v for k, v in kwargs.items() if hasattr(Contract, k)},
        )
        session.add(contract)
        await session.flush()

        chunk_count = await self._embed_and_index_text(
            text        = text,
            contract_id = str(contract_id),
            source_name = source_name,
        )

        contract.chunk_count = chunk_count
        contract.is_indexed  = True
        contract.status      = ContractStatus.ACTIVE
        await session.flush()

        logger.success(
            "Contract '{}' (text) indexed: {} chunks — contract_id={}",
            title, chunk_count, contract_id,
        )
        return contract

    # ── Search ───────────────────────────────────────────────────────────────────

    def search_clauses(
        self,
        query: str,
        contract_id: Optional[str] = None,
        k: int = 5,
        score_threshold: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Semantic clause search.

        Parameters
        ----------
        query           : natural language query
        contract_id     : restrict to one contract (None = search all)
        k               : max results
        score_threshold : minimum cosine similarity

        Returns
        -------
        List of {"text", "metadata", "score"} dicts, sorted by score desc.
        """
        from backend.rag.vector_store import get_vector_store
        vs = get_vector_store()
        return vs.search(
            query_text      = query,
            k               = k,
            contract_id     = contract_id,
            score_threshold = score_threshold,
        )

    # ── Re-indexing / Deletion ───────────────────────────────────────────────────

    async def reindex_contract(
        self,
        contract_id: str,
        session: AsyncSession,
    ) -> int:
        """
        Re-embed an existing contract (e.g. after embedding model upgrade).
        Falls back gracefully if the file is missing.

        Returns new chunk count.
        """
        from backend.rag.vector_store import get_vector_store
        stmt = select(Contract).where(Contract.faiss_index_id == contract_id)
        result = await session.execute(stmt)
        contract = result.scalar_one_or_none()
        if not contract:
            raise ValueError(f"Contract not found: {contract_id}")

        # Remove old vectors
        vs = get_vector_store()
        vs.remove_contract(contract_id)

        # Re-chunk and re-embed
        if contract.file_path and Path(contract.file_path).exists():
            pdf_bytes = Path(contract.file_path).read_bytes()
            chunk_count = await self._embed_and_index(
                pdf_bytes   = pdf_bytes,
                filename    = Path(contract.file_path).name,
                contract_id = contract_id,
            )
        else:
            logger.warning(
                "No PDF file for contract {} — skipping re-index", contract_id
            )
            chunk_count = 0

        contract.chunk_count    = chunk_count
        contract.is_indexed     = chunk_count > 0
        contract.embedding_model = self._settings.EMBEDDING_MODEL
        await session.flush()
        vs.save()

        return chunk_count

    async def delete_contract_vectors(
        self,
        contract_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        """
        Remove all FAISS vectors for a contract.
        Optionally updates the DB row's is_indexed flag.
        """
        from backend.rag.vector_store import get_vector_store
        vs      = get_vector_store()
        removed = vs.remove_contract(contract_id)
        vs.save()

        if session:
            stmt = select(Contract).where(Contract.faiss_index_id == contract_id)
            result = await session.execute(stmt)
            contract = result.scalar_one_or_none()
            if contract:
                contract.is_indexed  = False
                contract.chunk_count = 0
                await session.flush()

        return removed

    # ── Private helpers ──────────────────────────────────────────────────────────

    async def _embed_and_index(
        self,
        pdf_bytes:   bytes,
        filename:    str,
        contract_id: str,
    ) -> int:
        """Chunk a PDF, embed, and insert into FAISS. Returns chunk count."""
        from backend.rag.chunker import chunk_bytes
        from backend.rag.embedder import get_embedder
        from backend.rag.vector_store import get_vector_store

        chunks = chunk_bytes(pdf_bytes, contract_id=contract_id, filename=filename)
        if not chunks:
            logger.warning("No chunks produced for '{}' — check PDF content", filename)
            return 0

        emb  = get_embedder()
        vecs = emb.embed_texts([c["text"] for c in chunks])
        vs   = get_vector_store()
        vs.add_chunks(chunks, vecs)
        vs.save()
        return len(chunks)

    async def _embed_and_index_text(
        self,
        text:        str,
        contract_id: str,
        source_name: str,
    ) -> int:
        """Chunk raw text, embed, insert into FAISS. Returns chunk count."""
        from backend.rag.chunker import chunk_text
        from backend.rag.embedder import get_embedder
        from backend.rag.vector_store import get_vector_store

        chunks = chunk_text(text, contract_id=contract_id, source_name=source_name)
        if not chunks:
            return 0

        emb  = get_embedder()
        vecs = emb.embed_texts([c["text"] for c in chunks])
        vs   = get_vector_store()
        vs.add_chunks(chunks, vecs)
        vs.save()
        return len(chunks)

    async def _find_by_hash(
        self,
        file_hash: str,
        session: AsyncSession,
    ) -> Optional[Contract]:
        """Return existing Contract with matching SHA-256 hash, or None."""
        stmt = select(Contract).where(Contract.file_hash == file_hash)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
