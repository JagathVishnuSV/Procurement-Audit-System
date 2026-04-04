"""
backend/ml/vendor_risk.py
──────────────────────────────────────────────────────────────────────────────
Vendor Risk Profiler.

Computes a historical risk score [0, 1] per vendor based on behavioral
patterns across ALL their past transactions — not just the current one.

Why this matters
────────────────
Transaction-level models evaluate one invoice at a time.  Vendor risk
captures *patterns* across time:
  • A vendor who routinely submits invoices on weekends is inherently
    more suspicious than one who did so once.
  • A vendor whose invoice amounts are highly erratic (coefficient of
    variation > 200%) is more likely to be padding invoices.

Vendor Risk Score formula:
  0.30 × split_invoice_score   (ratio of near-threshold invoices × 5, capped at 1)
  0.25 × weekend_payment_score (weekend ratio × 3, capped at 1)
  0.20 × round_amount_score    (round-number ratio × 3, capped at 1)
  0.15 × amount_volatility     (coefficient of variation / 5, capped at 1)
  0.10 × invoice_velocity      (invoices/month / 10, capped at 1)

Artefact: models/vendor_risk_scores.json

Usage (called by trainer.py after model training):
  profiler = VendorRiskProfiler()
  profiler.build_and_save(real_df)

Usage (called by scorer.py at startup):
  profiler = VendorRiskProfiler()
  profiler.load()
  score = profiler.score_vendor("some-vendor-uuid")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

VENDOR_RISK_PATH = Path("models/vendor_risk_scores.json")

# Amounts within 5% below these thresholds are near-threshold invoices
_APPROVAL_THRESHOLDS = [10_000.0, 25_000.0, 100_000.0, 250_000.0]


def _is_near_threshold(x: float) -> bool:
    for lim in _APPROVAL_THRESHOLDS:
        if lim * 0.95 <= x < lim:
            return True
    return False


def _is_round(x: float) -> bool:
    if x >= 10_000 and abs(x % 500) < 0.01:
        return True
    if abs(x % 1_000) < 0.01:
        return True
    return False


class VendorRiskProfiler:
    """
    Computes and serves per-vendor behavioral risk scores.

    Lifecycle:
      1. trainer.py calls build_and_save(real_df) → writes JSON
      2. scorer.py calls load() at startup → reads JSON
      3. scorer.py calls score_vendor(vendor_id) per transaction
    """

    def __init__(self) -> None:
        self._scores: Dict[str, float] = {}

    # ── Building ───────────────────────────────────────────────────────────────

    def build_and_save(self, df: pd.DataFrame) -> None:
        """
        Compute risk scores for every vendor in df and persist to JSON.

        Parameters
        ----------
        df : DataFrame with columns [vendor_id, amount, date]
        """
        df = df.copy()
        df["date"]   = pd.to_datetime(df["date"], utc=True)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

        scores: Dict[str, float] = {}

        for vendor_id, grp in df.groupby("vendor_id"):
            amts  = grp["amount"].values.astype(float)
            dates = grp["date"]
            n     = len(grp)

            if n == 0:
                scores[str(vendor_id)] = 0.05
                continue

            # ── Component 1: Split-invoice propensity ────────────────────────
            near_thr_count = sum(_is_near_threshold(a) for a in amts)
            split_score    = min((near_thr_count / max(n, 1)) * 5.0, 1.0)

            # ── Component 2: Weekend payment habit ───────────────────────────
            weekend_count = int((dates.dt.dayofweek >= 5).sum())
            weekend_score = min((weekend_count / max(n, 1)) * 3.0, 1.0)

            # ── Component 3: Round-number invoice habit ───────────────────────
            round_count = sum(_is_round(a) for a in amts)
            round_score = min((round_count / max(n, 1)) * 3.0, 1.0)

            # ── Component 4: Amount volatility (CV) ───────────────────────────
            mean_amt = float(amts.mean())
            std_amt  = float(amts.std()) if n > 1 else 0.0
            cv       = std_amt / max(mean_amt, 1.0)
            volatility_score = min(cv / 5.0, 1.0)

            # ── Component 5: Invoice velocity (invoices per month) ────────────
            if n > 1:
                span_days  = max((dates.max() - dates.min()).total_seconds() / 86400.0, 1.0)
                inv_per_mo = n / (span_days / 30.0)
            else:
                inv_per_mo = 1.0
            velocity_score = min(inv_per_mo / 10.0, 1.0)

            # ── Composite ─────────────────────────────────────────────────────
            risk = (
                0.30 * split_score
                + 0.25 * weekend_score
                + 0.20 * round_score
                + 0.15 * volatility_score
                + 0.10 * velocity_score
            )
            scores[str(vendor_id)] = round(float(np.clip(risk, 0.0, 1.0)), 4)

        self._scores = scores
        VENDOR_RISK_PATH.parent.mkdir(parents=True, exist_ok=True)
        VENDOR_RISK_PATH.write_text(json.dumps(scores, indent=2))
        logger.success(
            "Vendor risk profiles → {} ({} vendors)", VENDOR_RISK_PATH, len(scores)
        )

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load pre-computed scores from JSON (non-fatal if file absent)."""
        if VENDOR_RISK_PATH.exists():
            self._scores = json.loads(VENDOR_RISK_PATH.read_text())
            logger.info(
                "Vendor risk profiles loaded ({} vendors)", len(self._scores)
            )
        else:
            logger.warning(
                "Vendor risk file not found at {} — all vendors default to 0.10",
                VENDOR_RISK_PATH,
            )

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score_vendor(self, vendor_id: str) -> float:
        """
        Return the vendor's behavioral risk score [0, 1].

        Unknown vendors (not in training data) default to 0.10 — slight
        elevated suspicion for new/unseen vendors.
        """
        return self._scores.get(str(vendor_id), 0.10)

    def top_risky_vendors(self, n: int = 10) -> list:
        """Return the n highest-risk vendors as (vendor_id, score) tuples."""
        return sorted(self._scores.items(), key=lambda x: x[1], reverse=True)[:n]
