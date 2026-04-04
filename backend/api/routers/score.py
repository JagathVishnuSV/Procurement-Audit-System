"""
backend/api/routers/score.py
──────────────────────────────────────────────────────────────────────────────
POST /api/v1/score        – score a single transaction
POST /api/v1/score/batch  – score up to 500 transactions

The scorer is injected as a FastAPI dependency (loaded once at startup).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from backend.ml.scorer import AnomalyResult, ProcurementScorer, get_scorer

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    transaction_id: Optional[UUID] = None
    vendor_id:      Optional[UUID] = None

    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    transaction_date: datetime = Field(..., description="ISO 8601 datetime")

    # Enrichments pulled from the Redis feature store by the caller
    # (or computed on-the-fly from DB if not cached yet)
    vendor_30d_spend:          float = Field(0.0, ge=0, description="Vendor 30-day rolling spend")
    days_since_last_invoice:   float = Field(0.0, ge=0, description="Days since vendor's last tx")
    vendor_avg_amount:         float = Field(0.0, ge=0, description="Vendor historical avg amount")
    # Temporal / velocity features (features 5–11)
    invoice_count_7d:            float = Field(0.0, ge=0, description="Invoice count in last 7 days")
    invoice_count_30d:           float = Field(0.0, ge=0, description="Invoice count in last 30 days")
    rolling_7d_sum:              float = Field(0.0, ge=0, description="Vendor 7-day rolling spend sum")
    max_invoices_7d_window:      float = Field(0.0, ge=0, description="Vendor historical 7-day burst count")
    consecutive_small_invoices:  float = Field(0.0, ge=0, description="Consecutive invoices under $10k")
    amount_std_30d:              float = Field(0.0, ge=0, description="Std dev of amounts in last 30 days")
    # Ultra-short burst features (features 12–14)
    invoice_count_24h:           float = Field(0.0, ge=0, description="Invoice count in last 24 hours")
    invoice_count_48h:           float = Field(0.0, ge=0, description="Invoice count in last 48 hours")
    invoice_sum_48h:             float = Field(0.0, ge=0, description="Vendor 48-hour rolling spend sum")
    # Sprint 3 behavioral features (features 16–19)
    freq_change_rate:            float = Field(0.0, description="7-day vs 30-day invoice frequency ratio")
    amount_zscore_30d:           float = Field(0.0, description="Amount z-score vs vendor 30-day history")
    invoice_spacing_cv:          float = Field(0.0, ge=0, description="Coefficient of variation of inter-invoice gaps")
    small_invoice_cluster_14d:   float = Field(0.0, ge=0, description="Count of sub-$10k invoices in last 14 days")
    # Peer comparison features (features 20–21)
    amount_vs_category_avg:      float = Field(1.0, gt=0, description="Vendor amount ÷ agency peer average amount")
    invoice_freq_vs_category:    float = Field(1.0, ge=0, description="Vendor 30d invoice rate ÷ agency peer average rate")


class ShapFeature(BaseModel):
    feature:     str
    raw_value:   float
    shap_impact: float


class ScoreResponse(BaseModel):
    transaction_id:     Optional[UUID]
    vendor_id:          Optional[UUID]
    anomaly_score:      float = Field(..., description="0–1, higher = more suspicious (hybrid blended)")
    is_anomaly:         bool
    threshold:          float
    raw_decision_value: float
    shap_features:      List[ShapFeature]
    top_reason:         str
    # Hybrid component scores (for transparency / audit trail)
    ml_score:           float = Field(0.0, description="Two-stage IF+RF ML score")
    rule_score:         float = Field(0.0, description="Deterministic rule engine score")
    triggered_rules:    List[str] = Field(default_factory=list, description="Rules that fired")
    rule_details:       dict      = Field(default_factory=dict,  description="Rule explanations")
    vendor_risk_score:  float = Field(0.0, description="Vendor behavioral risk profile")
    graph_risk_score:   float = Field(0.0, description="Network/relationship risk score")


class BatchScoreRequest(BaseModel):
    transactions: Annotated[List[ScoreRequest], Field(max_length=500)]


class BatchScoreResponse(BaseModel):
    results:       List[ScoreResponse]
    total:         int
    anomaly_count: int


# ── Helper ─────────────────────────────────────────────────────────────────────

def _build_response(req: ScoreRequest, result: AnomalyResult) -> ScoreResponse:
    return ScoreResponse(
        transaction_id=req.transaction_id,
        vendor_id=req.vendor_id,
        anomaly_score=result.anomaly_score,
        is_anomaly=result.is_anomaly,
        threshold=result.threshold,
        raw_decision_value=result.raw_decision_value,
        shap_features=[ShapFeature(**f) for f in result.shap_values],
        top_reason=result.top_reason,
        ml_score=result.ml_score,
        rule_score=result.rule_score,
        triggered_rules=result.triggered_rules,
        rule_details=result.rule_details,
        vendor_risk_score=result.vendor_risk_score,
        graph_risk_score=result.graph_risk_score,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/score", response_model=ScoreResponse, status_code=200)
async def score_transaction(
    req: ScoreRequest,
    scorer: ProcurementScorer = Depends(get_scorer),
) -> ScoreResponse:
    """
    Score a single procurement transaction for anomalies.

    Returns an anomaly_score (0–1), a boolean flag, and SHAP feature
    attributions explaining the top driver of the score.
    """
    try:
        result = scorer.score_transaction(
            amount=req.amount,
            transaction_date=req.transaction_date,
            vendor_id=str(req.vendor_id) if req.vendor_id else "",
            vendor_30d_spend=req.vendor_30d_spend,
            days_since_last_invoice=req.days_since_last_invoice,
            vendor_avg_amount=req.vendor_avg_amount,
            invoice_count_7d=req.invoice_count_7d,
            invoice_count_30d=req.invoice_count_30d,
            rolling_7d_sum=req.rolling_7d_sum,
            max_invoices_7d_window=req.max_invoices_7d_window,
            consecutive_small_invoices=req.consecutive_small_invoices,
            amount_std_30d=req.amount_std_30d,
            invoice_count_24h=req.invoice_count_24h,
            invoice_count_48h=req.invoice_count_48h,
            invoice_sum_48h=req.invoice_sum_48h,
            freq_change_rate=req.freq_change_rate,
            amount_zscore_30d=req.amount_zscore_30d,
            invoice_spacing_cv=req.invoice_spacing_cv,
            small_invoice_cluster_14d=req.small_invoice_cluster_14d,
            amount_vs_category_avg=req.amount_vs_category_avg,
            invoice_freq_vs_category=req.invoice_freq_vs_category,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    return _build_response(req, result)


@router.post("/score/batch", response_model=BatchScoreResponse, status_code=200)
async def score_transactions_batch(
    req: BatchScoreRequest,
    scorer: ProcurementScorer = Depends(get_scorer),
) -> BatchScoreResponse:
    """
    Score up to 500 transactions in a single request.
    Failures for individual transactions are logged but do not abort the batch.
    """
    results: List[ScoreResponse] = []

    for tx in req.transactions:
        try:
            result = scorer.score_transaction(
                amount=tx.amount,
                transaction_date=tx.transaction_date,
                vendor_id=str(tx.vendor_id) if tx.vendor_id else "",
                vendor_30d_spend=tx.vendor_30d_spend,
                days_since_last_invoice=tx.days_since_last_invoice,
                vendor_avg_amount=tx.vendor_avg_amount,
                invoice_count_7d=tx.invoice_count_7d,
                invoice_count_30d=tx.invoice_count_30d,
                rolling_7d_sum=tx.rolling_7d_sum,
                max_invoices_7d_window=tx.max_invoices_7d_window,
                consecutive_small_invoices=tx.consecutive_small_invoices,
                amount_std_30d=tx.amount_std_30d,
                invoice_count_24h=tx.invoice_count_24h,
                invoice_count_48h=tx.invoice_count_48h,
                invoice_sum_48h=tx.invoice_sum_48h,
                freq_change_rate=tx.freq_change_rate,
                amount_zscore_30d=tx.amount_zscore_30d,
                invoice_spacing_cv=tx.invoice_spacing_cv,
                small_invoice_cluster_14d=tx.small_invoice_cluster_14d,
                amount_vs_category_avg=tx.amount_vs_category_avg,
                invoice_freq_vs_category=tx.invoice_freq_vs_category,
            )
            results.append(_build_response(tx, result))
        except Exception as exc:
            logger.warning("Skipped tx {} in batch: {}", tx.transaction_id, exc)

    return BatchScoreResponse(
        results=results,
        total=len(results),
        anomaly_count=sum(1 for r in results if r.is_anomaly),
    )
