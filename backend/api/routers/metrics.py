"""Executive KPI endpoints for Sprint 5 dashboard."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.action_plan import ActionPlan, ActionPlanStatus
from backend.models.audit_case import AuditCase, AuditCaseStatus
from backend.models.transaction import Transaction
from backend.orchestration.source_normalizer import normalize_source_dimensions

router = APIRouter(prefix="/metrics", tags=["metrics"])


class RoiResponse(BaseModel):
    completed_action_plans: int
    total_dollars_saved: float
    average_dollars_saved: float


class CoverageResponse(BaseModel):
    total_transactions: int
    audited_transactions: int
    open_cases: int
    in_review_cases: int
    closed_cases: int
    audit_coverage_pct: float


class PipelineResponse(BaseModel):
    total_transactions: int
    scored_transactions: int
    unscored_transactions: int
    score_coverage_pct: float
    audit_cases: int
    audit_coverage_pct: float


class SourceMetricRow(BaseModel):
    source_system: str
    agency_family: str
    category_family: str
    total_transactions: int
    scored_transactions: int
    audit_cases: int
    escalated_cases: int
    high_risk_cases: int
    avg_ml_score: float
    score_coverage_pct: float


def _to_float(value: Decimal | None) -> float:
    return float(value) if value is not None else 0.0


@router.get("/roi", response_model=RoiResponse, summary="Executive ROI KPI")
async def roi_metrics(db: AsyncSession = Depends(get_db)) -> RoiResponse:
    completed_count = await db.scalar(
        select(func.count(ActionPlan.id)).where(ActionPlan.status == ActionPlanStatus.COMPLETED)
    )

    total_saved = await db.scalar(
        select(func.coalesce(func.sum(ActionPlan.dollars_saved), 0))
        .where(ActionPlan.status == ActionPlanStatus.COMPLETED)
    )

    avg_saved = await db.scalar(
        select(func.coalesce(func.avg(ActionPlan.dollars_saved), 0))
        .where(ActionPlan.status == ActionPlanStatus.COMPLETED)
    )

    return RoiResponse(
        completed_action_plans=int(completed_count or 0),
        total_dollars_saved=_to_float(total_saved),
        average_dollars_saved=_to_float(avg_saved),
    )


@router.get("/coverage", response_model=CoverageResponse, summary="Audit coverage KPI")
async def coverage_metrics(db: AsyncSession = Depends(get_db)) -> CoverageResponse:
    total_transactions = int(
        await db.scalar(select(func.count(Transaction.id)))
        or 0
    )

    audited_transactions = int(
        await db.scalar(select(func.count(AuditCase.id)))
        or 0
    )

    status_counts_rows = await db.execute(
        select(AuditCase.status, func.count(AuditCase.id)).group_by(AuditCase.status)
    )
    status_counts: Dict[AuditCaseStatus, int] = {
        row[0]: int(row[1]) for row in status_counts_rows.all()
    }

    open_cases = status_counts.get(AuditCaseStatus.OPEN, 0)
    in_review_cases = status_counts.get(AuditCaseStatus.IN_REVIEW, 0)
    closed_cases = status_counts.get(AuditCaseStatus.CLOSED, 0)

    coverage = 0.0
    if total_transactions > 0:
        coverage = round((audited_transactions / total_transactions) * 100, 4)

    return CoverageResponse(
        total_transactions=total_transactions,
        audited_transactions=audited_transactions,
        open_cases=open_cases,
        in_review_cases=in_review_cases,
        closed_cases=closed_cases,
        audit_coverage_pct=coverage,
    )


@router.get("/pipeline", response_model=PipelineResponse, summary="Scoring and audit pipeline coverage")
async def pipeline_metrics(db: AsyncSession = Depends(get_db)) -> PipelineResponse:
    total_transactions = int(await db.scalar(select(func.count(Transaction.id))) or 0)
    scored_transactions = int(
        await db.scalar(select(func.count(Transaction.id)).where(Transaction.is_scored.is_(True)))
        or 0
    )
    unscored_transactions = max(total_transactions - scored_transactions, 0)
    audit_cases = int(await db.scalar(select(func.count(AuditCase.id))) or 0)

    score_coverage = round((scored_transactions / total_transactions) * 100, 2) if total_transactions else 0.0
    audit_coverage = round((audit_cases / total_transactions) * 100, 4) if total_transactions else 0.0

    return PipelineResponse(
        total_transactions=total_transactions,
        scored_transactions=scored_transactions,
        unscored_transactions=unscored_transactions,
        score_coverage_pct=score_coverage,
        audit_cases=audit_cases,
        audit_coverage_pct=audit_coverage,
    )


@router.get("/sources", response_model=List[SourceMetricRow], summary="Source-normalized risk metrics")
async def source_metrics(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> List[SourceMetricRow]:
    tx_rows = await db.execute(
        select(
            Transaction.id,
            Transaction.source,
            Transaction.awarding_agency,
            Transaction.category,
            Transaction.is_scored,
            Transaction.ml_score,
        )
    )

    case_rows = await db.execute(
        select(
            AuditCase.transaction_id,
            AuditCase.risk_level,
            AuditCase.groq_escalated,
        )
    )
    case_map = {
        str(transaction_id): {
            "risk_level": risk_level,
            "groq_escalated": bool(groq_escalated),
        }
        for transaction_id, risk_level, groq_escalated in case_rows.all()
    }

    groups: Dict[tuple, Dict[str, float]] = {}

    for tx_id, source, agency, category, is_scored, ml_score in tx_rows.all():
        dims = normalize_source_dimensions(
            raw_source=source.value if source else None,
            awarding_agency=agency,
            category=category,
        )
        key = (dims.source_system, dims.agency_family, dims.category_family)
        bucket = groups.setdefault(
            key,
            {
                "total": 0,
                "scored": 0,
                "audit_cases": 0,
                "escalated": 0,
                "high_risk": 0,
                "ml_sum": 0.0,
                "ml_count": 0,
            },
        )

        bucket["total"] += 1
        if is_scored:
            bucket["scored"] += 1
        if isinstance(ml_score, (int, float)):
            bucket["ml_sum"] += float(ml_score)
            bucket["ml_count"] += 1

        case = case_map.get(str(tx_id))
        if case:
            bucket["audit_cases"] += 1
            if case.get("groq_escalated"):
                bucket["escalated"] += 1
            if (case.get("risk_level") or "").upper() == "HIGH":
                bucket["high_risk"] += 1

    rows: List[SourceMetricRow] = []
    for (source_system, agency_family, category_family), bucket in groups.items():
        total = int(bucket["total"])
        scored = int(bucket["scored"])
        avg_ml = (bucket["ml_sum"] / bucket["ml_count"]) if bucket["ml_count"] else 0.0
        rows.append(
            SourceMetricRow(
                source_system=source_system,
                agency_family=agency_family,
                category_family=category_family,
                total_transactions=total,
                scored_transactions=scored,
                audit_cases=int(bucket["audit_cases"]),
                escalated_cases=int(bucket["escalated"]),
                high_risk_cases=int(bucket["high_risk"]),
                avg_ml_score=round(avg_ml, 4),
                score_coverage_pct=round((scored / total) * 100, 2) if total else 0.0,
            )
        )

    rows.sort(
        key=lambda row: (
            row.score_coverage_pct,
            row.total_transactions,
        ),
    )
    return rows[: max(1, min(limit, 200))]
