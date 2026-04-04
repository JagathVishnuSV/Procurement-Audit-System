"""Automated orchestration engine for scoring and audit-case coverage uplift."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Optional

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.config import get_settings
from backend.database import AsyncSessionLocal
from backend.llm.gemini_audit import GeminiDeepAuditService
from backend.llm.groq_triage import GroqTriageService
from backend.models.audit_case import AuditCase, AuditCaseStatus, LLMVerdict
from backend.models.transaction import Transaction
from backend.rag.clm_service import CLMService


@dataclass
class AutoOrchestrationSummary:
    scanned: int = 0
    scored: int = 0
    cases_created: int = 0
    cases_updated: int = 0
    llm_triaged: int = 0
    llm_deep_audited: int = 0
    high_risk: int = 0
    medium_risk: int = 0
    low_risk: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None


_last_summary: Optional[AutoOrchestrationSummary] = None


def _risk_from_score(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.60:
        return "MEDIUM"
    return "LOW"


def _float_amount(value: Decimal | None) -> float:
    return float(value) if value is not None else 0.0


class OrchestrationEngine:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = asyncio.Lock()

    async def run_once(
        self,
        batch_size: Optional[int] = None,
        run_llm: Optional[bool] = None,
    ) -> AutoOrchestrationSummary:
        summary = AutoOrchestrationSummary()
        use_batch_size = batch_size or self._settings.ORCH_BATCH_SIZE
        use_run_llm = self._settings.ORCH_RUN_LLM if run_llm is None else run_llm

        async with self._lock:
            async with AsyncSessionLocal() as db:
                scorer = self._get_scorer()
                tx_rows = await db.execute(
                    select(Transaction)
                    .options(selectinload(Transaction.audit_case))
                    .where(Transaction.is_scored.is_(False))
                    .order_by(Transaction.date.desc())
                    .limit(use_batch_size)
                )
                txs = tx_rows.scalars().all()

                summary.scanned = len(txs)
                if not txs:
                    summary.completed_at = datetime.now(timezone.utc).isoformat()
                    self._set_last_summary(summary)
                    return summary

                groq_service = GroqTriageService() if use_run_llm else None
                gemini_service = GeminiDeepAuditService() if use_run_llm else None
                clm_service = CLMService() if use_run_llm else None
                llm_budget = self._settings.ORCH_MAX_LLM_PER_RUN

                for tx in txs:
                    temporal = await self._compute_temporal_features(db, tx)
                    result = scorer.score_transaction(
                        amount=float(tx.amount),
                        transaction_date=tx.date,
                        vendor_id=str(tx.vendor_id) if tx.vendor_id else "",
                        invoice_count_7d=temporal["invoice_count_7d"],
                        invoice_count_30d=temporal["invoice_count_30d"],
                        rolling_7d_sum=temporal["rolling_7d_sum"],
                        vendor_avg_amount=temporal["vendor_avg_amount"],
                        amount_zscore_30d=temporal["amount_zscore_30d"],
                        invoice_count_24h=temporal["invoice_count_24h"],
                        invoice_count_48h=temporal["invoice_count_48h"],
                    )

                    tx.is_scored = True
                    tx.ml_score = result.anomaly_score
                    summary.scored += 1

                    risk = _risk_from_score(result.anomaly_score)
                    if risk == "HIGH":
                        summary.high_risk += 1
                    elif risk == "MEDIUM":
                        summary.medium_risk += 1
                    else:
                        summary.low_risk += 1

                    case = tx.audit_case
                    case_policy = self._settings.ORCH_CASE_POLICY.upper()
                    should_create_case = (
                        case_policy == "ALL_SCORED"
                        or (case_policy == "ANOMALIES_ONLY" and result.is_anomaly)
                    )
                    if case is None and should_create_case:
                        case = AuditCase(
                            transaction_id=tx.id,
                            ml_score=result.anomaly_score,
                            shap_summary=result.top_reason,
                            shap_reason={"top_reason": result.top_reason, "shap_values": result.shap_values[:5]},
                            status=AuditCaseStatus.OPEN,
                            risk_level=risk,
                            estimated_impact_usd=_float_amount(tx.amount),
                        )
                        db.add(case)
                        await db.flush()
                        summary.cases_created += 1
                    elif case is not None:
                        case.ml_score = result.anomaly_score
                        case.shap_summary = result.top_reason
                        case.shap_reason = {"top_reason": result.top_reason, "shap_values": result.shap_values[:5]}
                        case.risk_level = risk
                        case.estimated_impact_usd = _float_amount(tx.amount)
                        summary.cases_updated += 1

                    if use_run_llm and case is not None and groq_service is not None and llm_budget > 0:
                        llm_budget = await self._run_llm_for_case(
                            tx=tx,
                            case=case,
                            ml_score=result.anomaly_score,
                            shap_summary=result.top_reason,
                            groq_service=groq_service,
                            gemini_service=gemini_service,
                            clm_service=clm_service,
                            llm_budget=llm_budget,
                            summary=summary,
                        )

                if use_run_llm and llm_budget > 0 and groq_service is not None:
                    pending_rows = await db.execute(
                        select(AuditCase)
                        .options(selectinload(AuditCase.transaction))
                        .where(AuditCase.groq_verdict.is_(None))
                        .order_by(AuditCase.created_at.desc())
                        .limit(max(1, llm_budget))
                    )
                    pending_cases = pending_rows.scalars().all()

                    for case in pending_cases:
                        tx = case.transaction
                        if tx is None:
                            continue
                        llm_budget = await self._run_llm_for_case(
                            tx=tx,
                            case=case,
                            ml_score=float(case.ml_score),
                            shap_summary=case.shap_summary,
                            groq_service=groq_service,
                            gemini_service=gemini_service,
                            clm_service=clm_service,
                            llm_budget=llm_budget,
                            summary=summary,
                        )
                        if llm_budget <= 0:
                            break

                await db.commit()

        summary.completed_at = datetime.now(timezone.utc).isoformat()
        self._set_last_summary(summary)
        return summary

    @staticmethod
    def _get_scorer():
        from backend.ml.scorer import get_scorer
        return get_scorer()

    async def _run_llm_for_case(
        self,
        tx: Transaction,
        case: AuditCase,
        ml_score: float,
        shap_summary: Optional[str],
        groq_service: GroqTriageService,
        gemini_service: Optional[GeminiDeepAuditService],
        clm_service: Optional[CLMService],
        llm_budget: int,
        summary: AutoOrchestrationSummary,
    ) -> int:
        if llm_budget <= 0:
            return llm_budget
        if ml_score < self._settings.ORCH_LLM_MIN_SCORE:
            return llm_budget

        triage_payload = {
            "transaction_id": str(tx.id),
            "vendor_id": str(tx.vendor_id),
            "amount": float(tx.amount),
            "date": tx.date.isoformat() if tx.date else None,
            "category": tx.category,
            "description": tx.description,
            "awarding_agency": tx.awarding_agency,
            "ml_score": ml_score,
            "shap_summary": shap_summary,
        }

        try:
            triage = await groq_service.triage_transaction(triage_payload)
            case.groq_verdict = triage.model_dump()
            case.groq_escalated = triage.escalate
            case.risk_level = triage.risk_level
            summary.llm_triaged += 1
        except Exception as exc:
            logger.warning("Orchestration triage skipped tx {}: {}", tx.id, exc)
            return llm_budget - 1

        if triage.escalate and gemini_service is not None and clm_service is not None:
            clause_hits = clm_service.search_clauses(
                query=" ".join(filter(None, [tx.description or "", tx.category or ""])).strip() or "procurement compliance",
                k=3,
                score_threshold=0.2,
            )
            try:
                report = await gemini_service.audit_transaction(triage_payload, clause_hits)
                case.gemini_report = report.model_dump()
                case.llm_verdict = LLMVerdict(report.verdict)
                case.confidence = report.confidence
                case.violated_clause_id = report.violated_clause
                case.contract_clause_cited = report.cited_clause_text
                summary.llm_deep_audited += 1
            except Exception as exc:
                logger.warning("Orchestration deep-audit skipped tx {}: {}", tx.id, exc)
        elif not triage.escalate:
            case.llm_verdict = LLMVerdict.INCONCLUSIVE

        return llm_budget - 1

    async def _compute_temporal_features(self, db: AsyncSession, tx: Transaction) -> Dict[str, float]:
        if tx.vendor_id is None or tx.date is None:
            return {
                "invoice_count_7d": 0.0,
                "invoice_count_30d": 0.0,
                "rolling_7d_sum": 0.0,
                "vendor_avg_amount": 0.0,
                "amount_zscore_30d": 0.0,
                "invoice_count_24h": 0.0,
                "invoice_count_48h": 0.0,
            }

        reference = tx.date
        window_30d = reference - timedelta(days=30)
        window_7d = reference - timedelta(days=7)
        window_24h = reference - timedelta(hours=24)
        window_48h = reference - timedelta(hours=48)

        stats_30d = await db.execute(
            select(
                func.count(Transaction.id),
                func.coalesce(func.avg(Transaction.amount), 0),
                func.coalesce(func.stddev_pop(Transaction.amount), 0),
            )
            .where(Transaction.vendor_id == tx.vendor_id)
            .where(Transaction.date < reference)
            .where(Transaction.date >= window_30d)
        )
        cnt_30d, avg_30d, std_30d = stats_30d.one()

        stats_7d = await db.execute(
            select(
                func.count(Transaction.id),
                func.coalesce(func.sum(Transaction.amount), 0),
            )
            .where(Transaction.vendor_id == tx.vendor_id)
            .where(Transaction.date < reference)
            .where(Transaction.date >= window_7d)
        )
        cnt_7d, sum_7d = stats_7d.one()

        cnt_24h = await db.scalar(
            select(func.count(Transaction.id))
            .where(Transaction.vendor_id == tx.vendor_id)
            .where(Transaction.date < reference)
            .where(Transaction.date >= window_24h)
        )
        cnt_48h = await db.scalar(
            select(func.count(Transaction.id))
            .where(Transaction.vendor_id == tx.vendor_id)
            .where(Transaction.date < reference)
            .where(Transaction.date >= window_48h)
        )

        avg_value = float(avg_30d or 0.0)
        std_value = float(std_30d or 0.0)
        amount_value = float(tx.amount or 0.0)
        zscore = 0.0
        if std_value > 0:
            zscore = (amount_value - avg_value) / std_value

        return {
            "invoice_count_7d": float(cnt_7d or 0),
            "invoice_count_30d": float(cnt_30d or 0),
            "rolling_7d_sum": float(sum_7d or 0.0),
            "vendor_avg_amount": avg_value,
            "amount_zscore_30d": zscore,
            "invoice_count_24h": float(cnt_24h or 0),
            "invoice_count_48h": float(cnt_48h or 0),
        }

    @staticmethod
    def _set_last_summary(summary: AutoOrchestrationSummary) -> None:
        global _last_summary
        _last_summary = summary


_engine_singleton: Optional[OrchestrationEngine] = None


def get_orchestration_engine() -> OrchestrationEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = OrchestrationEngine()
    return _engine_singleton


def get_last_orchestration_summary() -> Optional[Dict[str, object]]:
    if _last_summary is None:
        return None
    return asdict(_last_summary)
