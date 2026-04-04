"""Sprint 4 audit orchestration API (Groq triage -> optional Gemini deep audit)."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.llm.gemini_audit import GeminiAuditResult, GeminiDeepAuditService
from backend.llm.groq_triage import GroqTriageResult, GroqTriageService
from backend.models.audit_case import AuditCase, AuditCaseStatus, LLMVerdict
from backend.models.contract import Contract, ContractStatus
from backend.models.transaction import Transaction
from backend.rag.clm_service import CLMService

router = APIRouter(prefix="/audit", tags=["audit-orchestration"])


def get_groq_service() -> GroqTriageService:
    return GroqTriageService()


def get_gemini_service() -> GeminiDeepAuditService:
    return GeminiDeepAuditService()


def get_clm_service() -> CLMService:
    return CLMService()


class AuditTriggerResponse(BaseModel):
    case_id: uuid.UUID
    transaction_id: uuid.UUID
    escalated: bool
    gemini_invoked: bool
    risk_level: Optional[str]
    llm_verdict: Optional[LLMVerdict]
    confidence: Optional[float]
    violated_clause_id: Optional[str]
    message: str


def _decimal_to_float(value: Decimal | None) -> float:
    return float(value) if value is not None else 0.0


def _build_triage_payload(tx: Transaction, case: AuditCase | None) -> Dict[str, Any]:
    return {
        "transaction_id": str(tx.id),
        "vendor_id": str(tx.vendor_id),
        "amount": _decimal_to_float(tx.amount),
        "date": tx.date.isoformat() if tx.date else None,
        "category": tx.category,
        "description": tx.description,
        "awarding_agency": tx.awarding_agency,
        "ml_score": tx.ml_score,
        "shap_summary": case.shap_summary if case else None,
    }


async def _retrieve_relevant_clauses(
    tx: Transaction,
    db: AsyncSession,
    clm: CLMService,
) -> List[Dict[str, Any]]:
    contract_result = await db.execute(
        select(Contract.id)
        .where(Contract.vendor_id == tx.vendor_id)
        .where(Contract.is_indexed.is_(True))
        .where(Contract.status == ContractStatus.ACTIVE)
    )
    contract_ids = [str(contract_id) for contract_id in contract_result.scalars().all()]

    if not contract_ids:
        return []

    query_parts = [
        tx.description or "",
        tx.category or "",
        "split billing payment threshold invoice prohibition",
    ]
    query = " ".join(part for part in query_parts if part).strip()
    if not query:
        query = "procurement payment and invoice compliance terms"

    all_hits: List[Dict[str, Any]] = []
    for contract_id in contract_ids:
        hits = clm.search_clauses(query=query, contract_id=contract_id, k=2, score_threshold=0.2)
        all_hits.extend(hits)

    all_hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return all_hits[:3]


@router.post(
    "/trigger/{transaction_id}",
    response_model=AuditTriggerResponse,
    summary="Run Groq triage and conditionally Gemini deep audit",
)
async def trigger_audit(
    transaction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    groq_service: GroqTriageService = Depends(get_groq_service),
    gemini_service: GeminiDeepAuditService = Depends(get_gemini_service),
    clm: CLMService = Depends(get_clm_service),
) -> AuditTriggerResponse:
    """
    Orchestrates Sprint 4 pipeline:
      1) Groq triage on real transaction context
      2) Gemini deep audit only if Groq escalates
      3) Persist outputs in audit_cases table
    """
    tx_result = await db.execute(
        select(Transaction)
        .options(selectinload(Transaction.vendor), selectinload(Transaction.audit_case))
        .where(Transaction.id == transaction_id)
    )
    tx = tx_result.scalar_one_or_none()

    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found",
        )

    case = tx.audit_case
    if case is None:
        case = AuditCase(
            transaction_id=tx.id,
            ml_score=float(tx.ml_score or 0.0),
            status=AuditCaseStatus.OPEN,
            estimated_impact_usd=_decimal_to_float(tx.amount),
        )
        db.add(case)
        await db.flush()

    triage_payload = _build_triage_payload(tx, case)

    try:
        triage: GroqTriageResult = await groq_service.triage_transaction(triage_payload)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    case.groq_verdict = triage.model_dump()
    case.groq_escalated = triage.escalate
    case.risk_level = triage.risk_level

    gemini_invoked = False

    if triage.escalate:
        clause_hits = await _retrieve_relevant_clauses(tx=tx, db=db, clm=clm)
        try:
            deep_audit: GeminiAuditResult = await gemini_service.audit_transaction(
                transaction_payload=triage_payload,
                clause_hits=clause_hits,
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )

        case.gemini_report = deep_audit.model_dump()
        case.llm_verdict = LLMVerdict(deep_audit.verdict)
        case.confidence = deep_audit.confidence
        case.violated_clause_id = deep_audit.violated_clause
        case.contract_clause_cited = deep_audit.cited_clause_text
        gemini_invoked = True
    else:
        case.gemini_report = None
        case.llm_verdict = LLMVerdict.INCONCLUSIVE
        case.confidence = None
        case.violated_clause_id = None
        case.contract_clause_cited = None

    await db.flush()
    await db.commit()

    logger.info(
        "Audit trigger completed: transaction_id={} case_id={} escalated={} gemini_invoked={}",
        tx.id,
        case.id,
        triage.escalate,
        gemini_invoked,
    )

    return AuditTriggerResponse(
        case_id=case.id,
        transaction_id=tx.id,
        escalated=triage.escalate,
        gemini_invoked=gemini_invoked,
        risk_level=case.risk_level,
        llm_verdict=case.llm_verdict,
        confidence=case.confidence,
        violated_clause_id=case.violated_clause_id,
        message="Audit pipeline executed successfully",
    )
