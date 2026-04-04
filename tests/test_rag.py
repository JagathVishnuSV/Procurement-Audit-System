"""
tests/test_rag.py
──────────────────────────────────────────────────────────────────────────────
Sprint 3 unit tests: RAG engine, CLM service, and contract API endpoints.

Coverage
--------
1. Embedder       – shape, dtype, normalisation, singleton identity
2. Chunker        – text chunking, metadata fields, min-length filter
3. VectorStore    – add / search / remove / reset / persistence
4. CLMService     – ingest_text creates DB row, is_indexed=True, search works
5. API contracts  – upload, list, search, delete (in-memory DB + mock FAISS)
6. API cases      – list, get, status update, notes update

All tests run against SQLite in-memory (no PostgreSQL required).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from fastapi import status
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

# ── Fixtures imported from conftest ───────────────────────────────────────────
from tests.conftest import TEST_SETTINGS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedder
# ─────────────────────────────────────────────────────────────────────────────

class TestClauseEmbedder:
    """Unit tests for backend.rag.embedder.ClauseEmbedder."""

    def _make_embedder(self):
        """Build ClauseEmbedder with a tiny fake SentenceTransformer."""
        from backend.rag.embedder import ClauseEmbedder

        mock_st = MagicMock()
        mock_st.get_sentence_embedding_dimension.return_value = 384

        def _mock_encode(texts, normalize_embeddings=True, **kw):
            vecs = np.random.rand(len(texts), 384).astype(np.float32)
            if normalize_embeddings:
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

        mock_st.encode.side_effect = _mock_encode

        embedder = ClauseEmbedder.__new__(ClauseEmbedder)
        embedder._model = mock_st
        embedder._model_name = "test-model"
        embedder._dim = 384
        return embedder

    def test_embed_texts_shape(self):
        emb = self._make_embedder()
        texts = ["payment terms", "delivery clause", "termination rights"]
        result = emb.embed_texts(texts)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    def test_embed_texts_normalised(self):
        emb = self._make_embedder()
        result = emb.embed_texts(["test"], normalize=True)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_query_shape(self):
        emb = self._make_embedder()
        result = emb.embed_query("split billing prohibition")
        assert result.shape == (1, 384)

    def test_embed_query_normalised(self):
        emb = self._make_embedder()
        result = emb.embed_query("round-number invoice")
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_dim_property(self):
        emb = self._make_embedder()
        assert emb.dim == 384

    def test_singleton_identity(self):
        """get_embedder() must return the same object on repeated calls."""
        from backend.rag.embedder import get_embedder
        # Clear lru_cache between tests
        get_embedder.cache_clear()
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_cls.return_value = mock_instance
            a = get_embedder()
            b = get_embedder()
            assert a is b
        get_embedder.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunker
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    """Unit tests for backend.rag.chunker.*"""

    CONTRACT_TEXT = """
SERVICES AGREEMENT

1. SCOPE OF SERVICES
Vendor shall provide IT consulting services as described in Exhibit A.

2. PAYMENT TERMS
Invoices shall be submitted monthly. The Agency shall pay within 30 days.
Split invoicing is strictly prohibited.

3. TERMINATION
Either party may terminate this agreement with 30 days written notice.
""".strip()

    def test_chunk_text_returns_list(self):
        from backend.rag.chunker import chunk_text
        chunks = chunk_text(self.CONTRACT_TEXT, contract_id="cid-001")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_text_has_required_fields(self):
        from backend.rag.chunker import chunk_text
        chunks = chunk_text(self.CONTRACT_TEXT, contract_id="cid-001")
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["contract_id"] == "cid-001"

    def test_chunk_text_metadata_fields(self):
        from backend.rag.chunker import chunk_text
        chunks = chunk_text(self.CONTRACT_TEXT, contract_id="cid-001", source_name="agreement.txt")
        for chunk in chunks:
            meta = chunk["metadata"]
            assert "chunk_index" in meta
            assert "source" in meta

    def test_chunk_text_min_length_filter(self):
        from backend.rag.chunker import chunk_text, MIN_CHUNK_LEN
        chunks = chunk_text(self.CONTRACT_TEXT, contract_id="cid-001")
        for chunk in chunks:
            assert len(chunk["text"]) >= MIN_CHUNK_LEN

    def test_chunk_text_indices_are_sequential(self):
        from backend.rag.chunker import chunk_text
        chunks = chunk_text(self.CONTRACT_TEXT, contract_id="cid-001")
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_text_extra_metadata_merged(self):
        from backend.rag.chunker import chunk_text
        chunks = chunk_text(
            self.CONTRACT_TEXT,
            contract_id="cid-002",
            extra_metadata={"department": "procurement"},
        )
        for chunk in chunks:
            assert chunk["metadata"].get("department") == "procurement"


# ─────────────────────────────────────────────────────────────────────────────
# 3. ContractVectorStore
# ─────────────────────────────────────────────────────────────────────────────

class TestContractVectorStore:
    """Unit tests for backend.rag.vector_store.ContractVectorStore."""

    def _make_store(self, tmp_path) -> Any:
        from backend.rag.vector_store import ContractVectorStore
        return ContractVectorStore(str(tmp_path / "test_index"), dim=384)

    def _random_chunks(self, n: int, contract_id: str = "c1") -> tuple:
        chunks = [
            {"text": f"clause {i}", "metadata": {"contract_id": contract_id, "chunk_index": i}}
            for i in range(n)
        ]
        vecs = np.random.rand(n, 384).astype(np.float32)
        # L2-normalise so inner product == cosine
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms
        return chunks, vecs

    def test_add_and_count(self, tmp_path):
        store = self._make_store(tmp_path)
        chunks, vecs = self._random_chunks(5)
        store.add_chunks(chunks, vecs)
        assert store.total_vectors == 5

    def test_search_returns_results(self, tmp_path):
        store = self._make_store(tmp_path)
        chunks, vecs = self._random_chunks(10)
        store.add_chunks(chunks, vecs)

        with patch("backend.rag.embedder.get_embedder") as mock_get:
            fake_emb = MagicMock()
            fake_emb.embed_query.return_value = vecs[:1]  # use first vector as query
            mock_get.return_value = fake_emb
            results = store.search("payment terms", k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "metadata" in r

    def test_search_empty_index_returns_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        with patch("backend.rag.embedder.get_embedder"):
            results = store.search("anything")
        assert results == []

    def test_remove_contract(self, tmp_path):
        store = self._make_store(tmp_path)
        chunks_a, vecs_a = self._random_chunks(4, contract_id="contract-A")
        chunks_b, vecs_b = self._random_chunks(3, contract_id="contract-B")
        store.add_chunks(chunks_a, vecs_a)
        store.add_chunks(chunks_b, vecs_b)

        with patch("backend.rag.embedder.get_embedder") as mock_get:
            fake_emb = MagicMock()
            fake_emb.embed_texts.return_value = vecs_b
            mock_get.return_value = fake_emb
            removed = store.remove_contract("contract-A")

        assert removed == 4
        assert store.total_vectors == 3
        remaining_ids = store.list_contract_ids()
        assert "contract-A" not in remaining_ids
        assert "contract-B" in remaining_ids

    def test_reset_clears_index(self, tmp_path):
        store = self._make_store(tmp_path)
        chunks, vecs = self._random_chunks(5)
        store.add_chunks(chunks, vecs)
        store.reset()
        assert store.total_vectors == 0

    def test_contract_chunk_count(self, tmp_path):
        store = self._make_store(tmp_path)
        chunks, vecs = self._random_chunks(6, contract_id="mycontract")
        store.add_chunks(chunks, vecs)
        assert store.contract_chunk_count("mycontract") == 6
        assert store.contract_chunk_count("other") == 0

    def test_save_and_load(self, tmp_path):
        store = self._make_store(tmp_path)
        chunks, vecs = self._random_chunks(4)
        store.add_chunks(chunks, vecs)
        store.save()

        store2 = self._make_store(tmp_path)
        loaded = store2.load()
        assert loaded is True
        assert store2.total_vectors == 4

    def test_load_nonexistent_returns_false(self, tmp_path):
        store = self._make_store(tmp_path / "nonexistent")
        result = store.load()
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLMService (unit, with SQLite)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCLMService:
    """Tests for CLMService.ingest_text using SQLite in-memory DB."""

    async def test_ingest_text_creates_contract(
        self,
        async_sqlite_session,
        async_vendor_id,
    ):
        from backend.rag.clm_service import CLMService

        clm = CLMService()
        text = "Vendor shall submit invoices bi-weekly. No split billing is permitted."

        with (
            patch("backend.rag.vector_store.get_vector_store") as mock_vs,
            patch("backend.rag.embedder.get_embedder") as mock_emb,
        ):
            fake_store = MagicMock()
            fake_store.add_chunks = MagicMock()
            fake_store.save = MagicMock()
            mock_vs.return_value = fake_store

            fake_embedder = MagicMock()
            fake_embedder.embed_texts.return_value = np.random.rand(3, 384).astype(np.float32)
            mock_emb.return_value = fake_embedder

            contract = await clm.ingest_text(
                text=text,
                vendor_id=async_vendor_id,
                title="Test Consulting Agreement",
                session=async_sqlite_session,
            )
            await async_sqlite_session.flush()

        assert contract.id is not None
        assert contract.is_indexed is True
        assert contract.chunk_count is not None
        assert contract.chunk_count >= 1
        assert contract.title == "Test Consulting Agreement"
        assert str(contract.vendor_id) == str(async_vendor_id)

    async def test_ingest_text_sets_embedding_model(
        self,
        async_sqlite_session,
        async_vendor_id,
    ):
        from backend.rag.clm_service import CLMService

        clm = CLMService()
        with (
            patch("backend.rag.vector_store.get_vector_store") as mock_vs,
            patch("backend.rag.embedder.get_embedder") as mock_emb,
        ):
            mock_vs.return_value = MagicMock()
            fake_embedder = MagicMock()
            fake_embedder.embed_texts.return_value = np.random.rand(2, 384).astype(np.float32)
            fake_embedder.model_name = "all-MiniLM-L6-v2"
            mock_emb.return_value = fake_embedder

            contract = await clm.ingest_text(
                text="Some contract text here. Termination clause. Payment schedule.",
                vendor_id=async_vendor_id,
                title="Another Agreement",
                session=async_sqlite_session,
            )

        assert contract.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    async def test_ingest_text_duplicate_hash_raises(
        self,
        async_sqlite_session,
        async_vendor_id,
    ):
        """Identical text submitted twice should raise ValueError on the second call."""
        from backend.rag.clm_service import CLMService

        text = "Vendor shall provide widgets. Payment within 30 days."
        clm = CLMService()

        with (
            patch("backend.rag.vector_store.get_vector_store") as mock_vs,
            patch("backend.rag.embedder.get_embedder") as mock_emb,
        ):
            fake_store = MagicMock()
            mock_vs.return_value = fake_store
            fake_embedder = MagicMock()
            fake_embedder.embed_texts.return_value = np.random.rand(2, 384).astype(np.float32)
            mock_emb.return_value = fake_embedder

            # First ingest — should succeed
            await clm.ingest_text(
                text=text,
                vendor_id=async_vendor_id,
                title="Widget Agreement",
                session=async_sqlite_session,
            )
            await async_sqlite_session.flush()

            # Second ingest of same text — should raise ValueError
            with pytest.raises(ValueError, match="already indexed"):
                await clm.ingest_text(
                    text=text,
                    vendor_id=async_vendor_id,
                    title="Widget Agreement Duplicate",
                    session=async_sqlite_session,
                )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Contracts API (FastAPI TestClient with SQLite)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestContractsAPI:
    """Integration-style tests for the /api/v1/contracts endpoints."""

    async def test_list_contracts_empty(self, async_test_client):
        response = await async_test_client.get("/api/v1/contracts")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert isinstance(data["items"], list)

    async def test_search_contracts_returns_structure(self, async_test_client):
        response = await async_test_client.get(
            "/api/v1/contracts/search", params={"q": "payment terms"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query"] == "payment terms"
        assert isinstance(data["results"], list)
        assert "total" in data

    async def test_get_nonexistent_contract_404(self, async_test_client):
        fake_id = str(uuid.uuid4())
        response = await async_test_client.get(f"/api/v1/contracts/{fake_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_nonexistent_contract_404(self, async_test_client):
        fake_id = str(uuid.uuid4())
        response = await async_test_client.delete(f"/api/v1/contracts/{fake_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_upload_empty_pdf_rejected(self, async_test_client):
        response = await async_test_client.post(
            "/api/v1/contracts/upload",
            data={
                "vendor_id": str(uuid.uuid4()),
                "title": "Test Contract",
            },
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code in (
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    async def test_upload_wrong_content_type_rejected(self, async_test_client):
        response = await async_test_client.post(
            "/api/v1/contracts/upload",
            data={
                "vendor_id": str(uuid.uuid4()),
                "title": "Bad File",
            },
            files={"file": ("doc.docx", b"PK fake docx content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cases API
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCasesAPI:
    """Integration-style tests for the /api/v1/cases endpoints."""

    async def test_list_cases_empty(self, async_test_client):
        response = await async_test_client.get("/api/v1/cases")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert data["total"] >= 0

    async def test_get_nonexistent_case_404(self, async_test_client):
        fake_id = str(uuid.uuid4())
        response = await async_test_client.get(f"/api/v1/cases/{fake_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_update_nonexistent_case_status_404(self, async_test_client):
        fake_id = str(uuid.uuid4())
        response = await async_test_client.patch(
            f"/api/v1/cases/{fake_id}/status",
            json={"status": "IN_REVIEW"},
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_update_nonexistent_case_notes_404(self, async_test_client):
        fake_id = str(uuid.uuid4())
        response = await async_test_client.patch(
            f"/api/v1/cases/{fake_id}/notes",
            json={"auditor_notes": "Reviewed — possible split invoice pattern."},
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_list_cases_filter_by_status(self, async_test_client):
        response = await async_test_client.get(
            "/api/v1/cases", params={"status": "OPEN"}
        )
        assert response.status_code == status.HTTP_200_OK

    async def test_list_cases_filter_by_verdict(self, async_test_client):
        response = await async_test_client.get(
            "/api/v1/cases", params={"verdict": "FRAUD"}
        )
        assert response.status_code == status.HTTP_200_OK


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sample contracts content sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleContracts:
    """Ensure sample contracts module loads correctly and has expected keys."""

    def test_sample_contracts_loaded(self):
        from backend.rag.sample_contracts import SAMPLE_CONTRACTS
        assert len(SAMPLE_CONTRACTS) == 5

    def test_sample_contracts_keys(self):
        from backend.rag.sample_contracts import SAMPLE_CONTRACTS
        expected_keys = {"apex_it", "buildright", "medsupply", "logitrans", "cloudscale"}
        assert set(SAMPLE_CONTRACTS.keys()) == expected_keys

    def test_each_contract_has_required_fields(self):
        from backend.rag.sample_contracts import SAMPLE_CONTRACTS
        for key, meta in SAMPLE_CONTRACTS.items():
            assert "title" in meta, f"Missing title for {key}"
            assert "text" in meta, f"Missing text for {key}"
            assert len(meta["text"]) > 200, f"Suspiciously short text for {key}"

    def test_sample_contracts_chunckable(self):
        """All 5 sample contracts should produce at least 3 chunks each."""
        from backend.rag.chunker import chunk_text
        from backend.rag.sample_contracts import SAMPLE_CONTRACTS
        for key, meta in SAMPLE_CONTRACTS.items():
            chunks = chunk_text(meta["text"], contract_id=f"seed-{key}")
            assert len(chunks) >= 3, (
                f"Expected >=3 chunks for '{key}', got {len(chunks)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Additional conftest fixtures required by this module
# (added below so they live alongside the tests that need them)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vendor_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest_asyncio.fixture
async def async_vendor_id(async_sqlite_session) -> uuid.UUID:
    """Insert a real Vendor row into the test DB and return its UUID."""
    from backend.models.vendor import Vendor, RiskTier
    vendor = Vendor(
        name="Test Vendor LLC",
        normalized_name="test vendor llc",
        risk_tier=RiskTier.LOW,
        total_spend_ytd=0,
    )
    async_sqlite_session.add(vendor)
    await async_sqlite_session.flush()
    return vendor.id


@pytest_asyncio.fixture
async def async_sqlite_session(async_sqlite_engine):
    """Yield a single AsyncSession backed by SQLite for CLMService tests."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    factory = async_sessionmaker(
        bind=async_sqlite_engine, class_=AsyncSession,
        autocommit=False, autoflush=False, expire_on_commit=False,
    )
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def async_test_client(async_sqlite_engine):
    """Async HTTPX client wired to the FastAPI app with a SQLite DB session."""
    from backend.api.main import app
    from backend.database import get_db
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    factory = async_sessionmaker(
        bind=async_sqlite_engine, class_=AsyncSession,
        autocommit=False, autoflush=False, expire_on_commit=False,
    )

    async def override_get_db():
        async with factory() as session:
            yield session

    # Patch CLM search to avoid loading real FAISS in API tests
    with patch("backend.api.routers.contracts.CLMService") as mock_clm_cls:
        mock_clm = MagicMock()
        mock_clm.search_clauses.return_value = []
        mock_clm.delete_contract_vectors = AsyncMock(return_value=0)
        mock_clm_cls.return_value = mock_clm

        app.dependency_overrides[get_db] = override_get_db
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
        app.dependency_overrides.clear()
