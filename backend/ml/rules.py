"""
backend/ml/rules.py
──────────────────────────────────────────────────────────────────────────────
Rule-Based Fraud Engine for procurement anomaly detection.

Real enterprise audit systems (SAP Fraud Management, Deloitte Analytics)
always combine rules with ML because:
  • Rules catch *known* fraud patterns with 100% recall for those patterns.
  • ML finds *unknown* patterns but struggles with explainability.
  • Together: high precision AND high recall.

Rules implemented (based on ACFE and US GAO procurement fraud patterns):

  ID  Rule                       Weight  Signal
  --  -------------------------- ------  -----------------------------------
  R1  SPLIT_BILLING_BURST         0.30   ≥3 consecutive invoices < $10k
  R2  WEEKEND_HIGH_VALUE          0.20   Weekend payment + amount > 2× avg
  R3  BURST_24H                   0.35   >3 invoices within 24 hours
  R4  NEAR_APPROVAL_THRESHOLD     0.30   Amount 0-5% below $10k/$25k/$100k/$250k
  R5  LARGE_ROUND_NUMBER          0.20   Round amount ≥ $50k
  R6  VELOCITY_7D_HIGH            0.20   >5 invoices in 7 days
  R7  BURST_48H_HIGH_SPEND        0.25   48h spend > 3× vendor avg amount

Blending (in scorer.py):
  final_score = 0.50 * ml_score
              + 0.25 * rule_score
              + 0.15 * vendor_risk_score
              + 0.10 * graph_risk_score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ── Federal approval thresholds (avoid-oversight split points) ────────────────
_MICRO_PURCHASE        = 10_000.0
_SIMPLIFIED_THRESHOLD  = 25_000.0
_COMMERCIAL_ITEMS      = 100_000.0
_STANDARD_ACQUISITION  = 250_000.0
_APPROVAL_THRESHOLDS   = [
    _MICRO_PURCHASE,
    _SIMPLIFIED_THRESHOLD,
    _COMMERCIAL_ITEMS,
    _STANDARD_ACQUISITION,
]


@dataclass
class RuleResult:
    """Result from the rule engine evaluation."""
    rule_score:      float            # 0.0 – 1.0 composite risk score
    triggered_rules: List[str]        # list of rule IDs that fired
    rule_details:    Dict[str, str]   # human-readable explanations per rule


def evaluate_rules(
    amount: float,
    is_weekend: bool,
    amount_vs_vendor_avg: float,
    consecutive_small_invoices: float,
    invoice_count_24h: float,
    invoice_count_48h: float,
    invoice_count_7d: float,
    invoice_sum_48h: float,
    vendor_avg_amount: float,
    is_round_amount: bool,
    days_since_last_invoice: float,
) -> RuleResult:
    """
    Evaluate all deterministic procurement fraud rules.

    All parameters should match what the scorer already computes from the
    feature vector so no extra computation is needed.

    Returns
    -------
    RuleResult with a blended rule_score [0, 1] and list of triggered rules.
    """
    triggered: List[str]       = []
    details:   Dict[str, str]  = {}
    risk: float                = 0.0

    # ── R1: Split billing burst ───────────────────────────────────────────────
    # Classic split: break one large contract into many sub-threshold invoices
    # to avoid competitive bidding requirements.
    if consecutive_small_invoices >= 3:
        triggered.append("SPLIT_BILLING_BURST")
        weight = 0.30
        risk  += weight
        details["SPLIT_BILLING_BURST"] = (
            f"{int(consecutive_small_invoices)} consecutive invoices below the "
            f"${int(_MICRO_PURCHASE):,} micro-purchase threshold — possible split billing"
        )

    # ── R2: Weekend high-value payment ────────────────────────────────────────
    # Weekend payments of high-value amounts bypass normal approvals
    # (approvers may not check email; urgency pressure tactics).
    if is_weekend and amount_vs_vendor_avg > 2.0:
        triggered.append("WEEKEND_HIGH_VALUE")
        weight = 0.20
        risk  += weight
        details["WEEKEND_HIGH_VALUE"] = (
            f"Weekend payment of {amount_vs_vendor_avg:.1f}× vendor historical avg — "
            "high-value weekend transactions bypass normal approval chains"
        )

    # ── R3: 24-hour invoice burst ─────────────────────────────────────────────
    # Fraudsters rush before detection. >3 invoices in 24 hours is almost
    # always anomalous in federal procurement.
    if invoice_count_24h > 3:
        triggered.append("BURST_24H")
        weight = 0.35
        risk  += weight
        details["BURST_24H"] = (
            f"{int(invoice_count_24h)} invoices submitted within 24 hours — "
            "velocity burst typical of fraud before detection cutoff"
        )

    # ── R4: Near federal approval threshold ───────────────────────────────────
    # Amount is within 5% BELOW a threshold to avoid competitive bidding.
    for lim in _APPROVAL_THRESHOLDS:
        if lim * 0.95 <= amount < lim:
            triggered.append("NEAR_APPROVAL_THRESHOLD")
            weight = 0.30
            risk  += weight
            details["NEAR_APPROVAL_THRESHOLD"] = (
                f"Invoice amount ${amount:,.2f} is within 5% below the "
                f"${lim:,.0f} federal approval threshold — possible threshold manipulation"
            )
            break  # only flag once even if multiple brackets match

    # ── R5: Large round-number invoice ────────────────────────────────────────
    # Real transaction amounts are rarely exact round numbers at scale; exact
    # round numbers at ≥$50k are a classic red flag for fictitious invoices.
    if is_round_amount and amount >= 50_000.0:
        triggered.append("LARGE_ROUND_NUMBER")
        weight = 0.20
        risk  += weight
        details["LARGE_ROUND_NUMBER"] = (
            f"Invoice of exactly ${amount:,.2f} — large round-number amounts "
            "suggest fictitious or pre-arranged invoices"
        )

    # ── R6: 7-day invoice velocity too high ───────────────────────────────────
    # More than 5 invoices from the same vendor within a week is unusual for
    # standard government contracting cycles.
    if invoice_count_7d > 5:
        triggered.append("VELOCITY_7D_HIGH")
        weight = 0.20
        risk  += weight
        details["VELOCITY_7D_HIGH"] = (
            f"{int(invoice_count_7d)} invoices in last 7 days — unusually high "
            "frequency for a single vendor in government procurement"
        )

    # ── R7: 48-hour spend burst ────────────────────────────────────────────────
    # If 48h spend > 3× vendor average, something unusual is happening recently.
    _avg = max(vendor_avg_amount, 1.0)
    if invoice_sum_48h > _avg * 3.0 and invoice_count_48h >= 2:
        triggered.append("BURST_48H_HIGH_SPEND")
        weight = 0.25
        risk  += weight
        details["BURST_48H_HIGH_SPEND"] = (
            f"48-hour spending of ${invoice_sum_48h:,.2f} is {invoice_sum_48h/_avg:.1f}× "
            "vendor's typical invoice — concentrated short-window payment burst"
        )

    return RuleResult(
        rule_score=float(min(risk, 1.0)),
        triggered_rules=triggered,
        rule_details=details,
    )
