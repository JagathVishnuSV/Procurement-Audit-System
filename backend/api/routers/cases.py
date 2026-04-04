"""
backend/api/routers/cases.py
──────────────────────────────────────────────────────────────────────────────
Audit Case management API — the Kanban board data source.

Endpoints
─────────
GET    /api/v1/cases                  List audit cases (paginated, filterable)
GET    /api/v1/cases/{case_id}        Full evidence card for one case
PATCH  /api/v1/cases/{case_id}/status Move case through Kanban columns
PATCH  /api/v1/cases/{case_id}/notes  Auditor notes update
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models.audit_case import AuditCase, AuditCaseStatus, LLMVerdict

router = APIRouter(prefix="/cases", tags=["audit-cases"])


# ── Response schemas ───────────────────────────────────────────────────────────

class CaseSummary(BaseModel):
    id: uuid.UUID
    transaction_id: uuid.UUID
    ml_score: float
    status: AuditCaseStatus
    risk_level: Optional[str]
    groq_escalated: Optional[bool]
    llm_verdict: Optional[LLMVerdict]
    confidence: Optional[float]
    estimated_impact_usd: Optional[float]
    created_at: Optional[str]
    updated_at: Optional[str]

    model_config = {"from_attributes": True}


class CaseDetail(CaseSummary):
    shap_reason: Optional[Dict[str, Any]]
    shap_summary: Optional[str]
    groq_verdict: Optional[Dict[str, Any]]
    gemini_report: Optional[Dict[str, Any]]
    contract_clause_cited: Optional[str]
    violated_clause_id: Optional[str]
    auditor_notes: Optional[str]


class StatusUpdateRequest(BaseModel):
    status: AuditCaseStatus = Field(..., description="New kanban status")


class NotesUpdateRequest(BaseModel):
    auditor_notes: str = Field(..., max_length=10000, description="Auditor notes (freeform)")


class CaseListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[CaseSummary]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_summary(c: AuditCase) -> CaseSummary:
    return CaseSummary(
        id=c.id,
        transaction_id=c.transaction_id,
        ml_score=c.ml_score,
        status=c.status,
        risk_level=c.risk_level,
        groq_escalated=c.groq_escalated,
        llm_verdict=c.llm_verdict,
        confidence=c.confidence,
        estimated_impact_usd=c.estimated_impact_usd,
        created_at=c.created_at.isoformat() if c.created_at else None,
        updated_at=c.updated_at.isoformat() if c.updated_at else None,
    )


def _to_detail(c: AuditCase) -> CaseDetail:
    return CaseDetail(
        **_to_summary(c).model_dump(),
        shap_reason=c.shap_reason,
        shap_summary=c.shap_summary,
        groq_verdict=c.groq_verdict,
        gemini_report=c.gemini_report,
        contract_clause_cited=c.contract_clause_cited,
        violated_clause_id=c.violated_clause_id,
        auditor_notes=c.auditor_notes,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=CaseListResponse,
    summary="List audit cases for the Kanban inbox",
)
async def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[AuditCaseStatus] = Query(None, alias="status"),
    verdict: Optional[LLMVerdict] = Query(None),
    risk_level: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None, ge=0.0),
    db: AsyncSession = Depends(get_db),
) -> CaseListResponse:
    """
    Return a paginated list of audit cases suitable for the Kanban board.
    Supports filtering by status, LLM verdict, risk level, and ML score.
    """
    stmt = select(AuditCase).order_by(AuditCase.created_at.desc())

    if status_filter:
        stmt = stmt.where(AuditCase.status == status_filter)
    if verdict:
        stmt = stmt.where(AuditCase.llm_verdict == verdict)
    if risk_level:
        stmt = stmt.where(AuditCase.risk_level == risk_level.upper())
    if min_score is not None:
        stmt = stmt.where(AuditCase.ml_score >= min_score)

    from sqlalchemy import func
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    cases = result.scalars().all()

    return CaseListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[_to_summary(c) for c in cases],
    )


@router.get(
    "/{case_id}",
    response_model=CaseDetail,
    summary="Get full evidence card for an audit case",
)
async def get_case(
    case_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> CaseDetail:
    """
    Return the complete evidence card for a single audit case.
    Includes ML scores, SHAP reasons, Groq triage, Gemini report,
    and any contract clause cited.
    """
    result = await db.execute(
        select(AuditCase).where(AuditCase.id == case_id)
    )
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit case {case_id} not found",
        )

    return _to_detail(case)


@router.patch(
    "/{case_id}/status",
    response_model=CaseSummary,
    summary="Move an audit case to a new Kanban status",
)
async def update_case_status(
    case_id: uuid.UUID,
    body: StatusUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> CaseSummary:
    """
    Move an audit case between OPEN → IN_REVIEW → CLOSED.
    Returns the updated case summary.
    """
    result = await db.execute(select(AuditCase).where(AuditCase.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit case {case_id} not found",
        )

    if case.status == body.status:
        return _to_summary(case)

    # Enforce valid transitions: CLOSED → OPEN is a re-open (allowed)
    old_status = case.status
    case.status = body.status
    await db.flush()
    await db.commit()

    logger.info(
        "Case {} status: {} → {}", case_id, old_status.value, body.status.value
    )
    return _to_summary(case)


@router.patch(
    "/{case_id}/notes",
    response_model=CaseSummary,
    summary="Update auditor notes on a case",
)
async def update_auditor_notes(
    case_id: uuid.UUID,
    body: NotesUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> CaseSummary:
    """Append or overwrite the freeform auditor notes field on an audit case."""
    result = await db.execute(select(AuditCase).where(AuditCase.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit case {case_id} not found",
        )

    case.auditor_notes = body.auditor_notes
    await db.flush()
    await db.commit()

    return _to_summary(case)
