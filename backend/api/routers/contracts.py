"""
backend/api/routers/contracts.py
──────────────────────────────────────────────────────────────────────────────
Contract Lifecycle Management (CLM) API.

Endpoints
─────────
POST   /api/v1/contracts/upload           Upload a PDF contract for indexing
GET    /api/v1/contracts                  List all contracts (paginated)
GET    /api/v1/contracts/search           Semantic clause search across FAISS
GET    /api/v1/contracts/{contract_id}    Single contract detail
DELETE /api/v1/contracts/{contract_id}    Remove contract + FAISS vectors
"""

from __future__ import annotations

import uuid
from typing import Annotated, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.contract import Contract, ContractStatus
from backend.rag.clm_service import CLMService

router = APIRouter(prefix="/contracts", tags=["contracts"])


# ── Dependency ─────────────────────────────────────────────────────────────────

def get_clm_service() -> CLMService:
    return CLMService()


# ── Response schemas ───────────────────────────────────────────────────────────

class ContractSummary(BaseModel):
    id: uuid.UUID
    title: str
    vendor_id: uuid.UUID
    contract_number: Optional[str]
    status: ContractStatus
    is_indexed: bool
    chunk_count: Optional[int]
    total_value: Optional[float]
    effective_date: Optional[str]
    expiry_date: Optional[str]

    model_config = {"from_attributes": True}


class ContractDetail(ContractSummary):
    description: Optional[str]
    embedding_model: Optional[str]
    file_path: Optional[str]
    faiss_index_id: Optional[str]


class ClauseResult(BaseModel):
    text: str
    score: float
    contract_id: Optional[str]
    source: Optional[str]
    page: Optional[int]
    chunk_index: Optional[int]


class SearchResponse(BaseModel):
    query: str
    results: List[ClauseResult]
    total: int


class UploadResponse(BaseModel):
    contract_id: uuid.UUID
    title: str
    chunk_count: int
    is_indexed: bool
    message: str


class DeleteResponse(BaseModel):
    contract_id: uuid.UUID
    vectors_removed: int
    message: str


class ContractListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ContractSummary]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_summary(c: Contract) -> ContractSummary:
    return ContractSummary(
        id=c.id,
        title=c.title,
        vendor_id=c.vendor_id,
        contract_number=c.contract_number,
        status=c.status,
        is_indexed=bool(c.is_indexed),
        chunk_count=c.chunk_count,
        total_value=c.total_value,
        effective_date=c.effective_date.isoformat() if c.effective_date else None,
        expiry_date=c.expiry_date.isoformat() if c.expiry_date else None,
    )


def _to_detail(c: Contract) -> ContractDetail:
    return ContractDetail(
        **_to_summary(c).model_dump(),
        description=c.description,
        embedding_model=c.embedding_model,
        file_path=c.file_path,
        faiss_index_id=c.faiss_index_id,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF contract",
)
async def upload_contract(
    file: UploadFile = File(..., description="PDF contract file"),
    vendor_id: uuid.UUID = Form(..., description="Vendor UUID"),
    title: str = Form(..., max_length=500, description="Human-readable contract title"),
    contract_number: Optional[str] = Form(None, max_length=100),
    total_value: Optional[float] = Form(None, ge=0),
    description: Optional[str] = Form(None, max_length=4000),
    db: AsyncSession = Depends(get_db),
    clm: CLMService = Depends(get_clm_service),
) -> UploadResponse:
    """
    Accept a multipart PDF upload, chunk and embed its text,
    store vectors in FAISS, and persist a Contract row in the database.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted (content-type: application/pdf)",
        )

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    if len(pdf_bytes) > 50 * 1024 * 1024:  # 50 MB cap
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 50 MB limit",
        )

    try:
        contract = await clm.ingest_pdf(
            pdf_bytes=pdf_bytes,
            filename=file.filename or "contract.pdf",
            vendor_id=vendor_id,
            title=title,
            session=db,
            contract_number=contract_number,
            total_value=total_value,
            description=description,
        )
        await db.commit()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        logger.exception("Contract upload failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process contract — check server logs",
        )

    return UploadResponse(
        contract_id=contract.id,
        title=contract.title,
        chunk_count=contract.chunk_count or 0,
        is_indexed=bool(contract.is_indexed),
        message="Contract uploaded and indexed successfully",
    )


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Semantic clause search across all indexed contracts",
)
async def search_clauses(
    q: str = Query(..., min_length=3, max_length=500, description="Natural language query"),
    contract_id: Optional[str] = Query(None, description="Restrict search to one contract"),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    score_threshold: float = Query(0.20, ge=0.0, le=1.0),
    clm: CLMService = Depends(get_clm_service),
) -> SearchResponse:
    """
    Semantic search across all contract clauses stored in FAISS.
    Optionally restrict to a specific contract with `contract_id`.
    """
    hits = clm.search_clauses(
        query=q,
        contract_id=contract_id,
        k=k,
        score_threshold=score_threshold,
    )

    results = [
        ClauseResult(
            text=h["text"],
            score=h["score"],
            contract_id=h["metadata"].get("contract_id"),
            source=h["metadata"].get("source"),
            page=h["metadata"].get("page"),
            chunk_index=h["metadata"].get("chunk_index"),
        )
        for h in hits
    ]

    return SearchResponse(query=q, results=results, total=len(results))


@router.get(
    "",
    response_model=ContractListResponse,
    summary="List all contracts",
)
async def list_contracts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[ContractStatus] = Query(None, alias="status"),
    vendor_id: Optional[uuid.UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> ContractListResponse:
    """Return a paginated list of all contracts with optional filters."""
    stmt = select(Contract).order_by(Contract.created_at.desc())

    if status_filter:
        stmt = stmt.where(Contract.status == status_filter)
    if vendor_id:
        stmt = stmt.where(Contract.vendor_id == vendor_id)

    # Count total
    from sqlalchemy import func
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Paginate
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    contracts = result.scalars().all()

    return ContractListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[_to_summary(c) for c in contracts],
    )


@router.get(
    "/{contract_id}",
    response_model=ContractDetail,
    summary="Get a single contract by ID",
)
async def get_contract(
    contract_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ContractDetail:
    """Return full contract detail including RAG fields."""
    result = await db.execute(select(Contract).where(Contract.id == contract_id))
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract {contract_id} not found",
        )

    return _to_detail(contract)


@router.delete(
    "/{contract_id}",
    response_model=DeleteResponse,
    summary="Delete a contract and remove its FAISS vectors",
)
async def delete_contract(
    contract_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    clm: CLMService = Depends(get_clm_service),
) -> DeleteResponse:
    """
    Soft-delete the DB row (status → TERMINATED) and remove all FAISS
    vectors associated with this contract.
    """
    result = await db.execute(select(Contract).where(Contract.id == contract_id))
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract {contract_id} not found",
        )

    vectors_removed = await clm.delete_contract_vectors(
        contract_id=str(contract_id), session=db
    )

    # Soft-delete: mark as TERMINATED rather than hard-delete
    contract.status = ContractStatus.TERMINATED
    contract.is_indexed = False
    await db.flush()
    await db.commit()

    return DeleteResponse(
        contract_id=contract_id,
        vectors_removed=vectors_removed,
        message=f"Contract terminated and {vectors_removed} FAISS vectors removed",
    )
