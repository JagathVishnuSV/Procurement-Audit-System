"""Manual and status endpoints for automated scoring orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import func, select

from backend.database import AsyncSessionLocal
from backend.models.audit_case import AuditCase
from backend.models.transaction import Transaction
from backend.orchestration import get_last_orchestration_summary, get_orchestration_engine

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


class OrchestrationRunResponse(BaseModel):
    scanned: int
    scored: int
    cases_created: int
    cases_updated: int
    llm_triaged: int
    llm_deep_audited: int
    high_risk: int
    medium_risk: int
    low_risk: int
    started_at: str
    completed_at: Optional[str]


class OrchestrationStatusResponse(BaseModel):
    total_transactions: int
    scored_transactions: int
    unscored_transactions: int
    audit_cases: int
    last_run: Optional[Dict[str, Any]]
    generated_at: str


@router.post("/run", response_model=OrchestrationRunResponse, summary="Run orchestration batch now")
async def run_orchestration_now(
    batch_size: int = Query(200, ge=1, le=5000),
    run_llm: bool = Query(False),
) -> OrchestrationRunResponse:
    engine = get_orchestration_engine()
    summary = await engine.run_once(batch_size=batch_size, run_llm=run_llm)
    return OrchestrationRunResponse(**summary.__dict__)


@router.get("/status", response_model=OrchestrationStatusResponse, summary="Current orchestration backlog/status")
async def orchestration_status() -> OrchestrationStatusResponse:
    async with AsyncSessionLocal() as db:
        total_transactions = int(await db.scalar(select(func.count(Transaction.id))) or 0)
        scored_transactions = int(
            await db.scalar(select(func.count(Transaction.id)).where(Transaction.is_scored.is_(True)))
            or 0
        )
        unscored_transactions = max(total_transactions - scored_transactions, 0)
        audit_cases = int(await db.scalar(select(func.count(AuditCase.id))) or 0)

    return OrchestrationStatusResponse(
        total_transactions=total_transactions,
        scored_transactions=scored_transactions,
        unscored_transactions=unscored_transactions,
        audit_cases=audit_cases,
        last_run=get_last_orchestration_summary(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
