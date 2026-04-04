"""
backend/ml/scorer.py
──────────────────────────────────────────────────────────────────────────────
Production scorer.  Loads the pre-trained IsolationForest + RobustScaler +
second-stage RandomForest + rule engine + vendor risk profiles + graph risk
scores once at API startup and exposes score_transaction() for use by:

  • FastAPI  POST /api/v1/score
  • Kafka consumer (Sprint 3 stream processor)

Hybrid blended scoring:
  final_score = 0.50 × ml_score       (IF → RF two-stage)
              + 0.25 × rule_score      (deterministic rule engine)
              + 0.15 × vendor_risk     (vendor behavioral profile)
              + 0.10 × graph_risk      (network/relationship risk)

Score interpretation: 0.0 = most normal, 1.0 = most anomalous
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import shap
from loguru import logger

from backend.ml.features import FEATURE_NAMES, N_FEATURES, build_single_feature_vector
from backend.ml.graph_fraud import GraphFraudDetector
from backend.ml.rules import RuleResult, evaluate_rules
from backend.ml.vendor_risk import VendorRiskProfiler

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH        = Path("models/isolation_forest.joblib")
SCALER_PATH       = Path("models/scaler.joblib")
METADATA_PATH     = Path("models/model_metadata.json")
SECOND_STAGE_PATH = Path("models/second_stage_clf.joblib")

# Default threshold — overridden at load() time with the value stored in
# model_metadata.json by trainer.py (contamination auto-tune saves it there).
_DEFAULT_THRESHOLD: float = 0.5


def _load_metadata_threshold() -> float:
    """Read optimal_threshold from model_metadata.json, fallback to 0.5."""
    try:
        data = json.loads(METADATA_PATH.read_text())
        thr = float(data.get("optimal_threshold", _DEFAULT_THRESHOLD))
        logger.debug("Loaded optimal_threshold={} from {}", thr, METADATA_PATH)
        return thr
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning(
            "Could not read optimal_threshold from metadata ({}); using {}",
            exc, _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    anomaly_score:       float               # 0–1, higher = more suspicious
    is_anomaly:          bool
    threshold:           float
    raw_decision_value:  float               # raw IsolationForest output
    shap_values:         List[Dict]          # [{feature, raw_value, shap_impact}]
    top_reason:          str                 # human-readable top driver
    # Hybrid component scores (for transparency)
    ml_score:            float = 0.0         # two-stage IF+RF score
    rule_score:          float = 0.0         # deterministic rule engine
    triggered_rules:     List[str] = field(default_factory=list)
    rule_details:        Dict[str, str] = field(default_factory=dict)
    vendor_risk_score:   float = 0.0         # vendor behavioral profile
    graph_risk_score:    float = 0.0         # network/relationship risk


# ── Scorer class ──────────────────────────────────────────────────────────────

class ProcurementScorer:
    """
    Thread-safe scorer singleton.  Load once; call score_transaction() freely.

    Usage
    -----
    scorer = ProcurementScorer()
    scorer.load()
    result = scorer.score_transaction(amount=15_000, transaction_date=dt)
    """

    def __init__(self) -> None:
        self._model     = None
        self._scaler    = None
        self._clf2      = None   # second-stage RandomForest (optional)
        self._explainer = None
        self._loaded    = False
        self._threshold: float = _DEFAULT_THRESHOLD
        self._vendor_risk  = VendorRiskProfiler()    # vendor behavioral profiler
        self._graph_fraud  = GraphFraudDetector()    # network risk detector

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model + scaler from disk and initialise SHAP explainer."""
        if self._loaded:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python -m backend.ml.trainer"
            )

        logger.info("Loading IsolationForest from {}", MODEL_PATH)
        self._model     = joblib.load(MODEL_PATH)
        self._scaler    = joblib.load(SCALER_PATH)
        # Load the F1-optimal threshold saved by the trainer
        self._threshold = _load_metadata_threshold()
        # Second-stage RandomForest (trained in trainer.py step 9b)  — optional
        if SECOND_STAGE_PATH.exists():
            self._clf2 = joblib.load(SECOND_STAGE_PATH)
            logger.info("Second-stage RF loaded from {}", SECOND_STAGE_PATH)
        else:
            logger.warning(
                "No second-stage classifier at {} — using IF score only",
                SECOND_STAGE_PATH,
            )
        # TreeExplainer works directly with IsolationForest
        self._explainer = shap.TreeExplainer(self._model)
        # Load vendor + graph risk profiles (non-fatal if absent)
        self._vendor_risk.load()
        self._graph_fraud.load()
        self._loaded    = True
        logger.success(
            "ML scorer ready ({} features, threshold={}, second_stage={}).",
            N_FEATURES, self._threshold, self._clf2 is not None,
        )

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score_transaction(
        self,
        amount: float,
        transaction_date: datetime,
        vendor_id: str = "",
        vendor_30d_spend: float = 0.0,
        days_since_last_invoice: float = 0.0,
        vendor_avg_amount: float = 0.0,
        # New feature-store params (default 0 — backward-compatible)
        invoice_count_7d: float = 0.0,
        invoice_count_30d: float = 0.0,
        rolling_7d_sum: float = 0.0,
        max_invoices_7d_window: float = 0.0,
        consecutive_small_invoices: float = 0.0,
        amount_std_30d: float = 0.0,
        invoice_count_24h: float = 0.0,
        invoice_count_48h: float = 0.0,
        invoice_sum_48h: float = 0.0,
        # Sprint 3 behavioral features (default 0 — backward-compatible)
        freq_change_rate: float = 0.0,
        amount_zscore_30d: float = 0.0,
        invoice_spacing_cv: float = 0.0,
        small_invoice_cluster_14d: float = 0.0,
        # Peer comparison features (default 1.0 = vendor at agency average)
        amount_vs_category_avg: float = 1.0,
        invoice_freq_vs_category: float = 1.0,
    ) -> AnomalyResult:
        """Score a single transaction using all four detection layers."""
        if not self._loaded:
            raise RuntimeError("Scorer not loaded. Call scorer.load() first.")

        # ── Stage 1: Build 20-feature vector ───────────────────────────────────────
        X_raw = build_single_feature_vector(
            amount=amount,
            date=transaction_date,
            vendor_30d_spend=vendor_30d_spend,
            days_since_last_invoice=days_since_last_invoice,
            vendor_avg_amount=vendor_avg_amount,
            invoice_count_7d=invoice_count_7d,
            invoice_count_30d=invoice_count_30d,
            rolling_7d_sum=rolling_7d_sum,
            max_invoices_7d_window=max_invoices_7d_window,
            consecutive_small_invoices=consecutive_small_invoices,
            amount_std_30d=amount_std_30d,
            invoice_count_24h=invoice_count_24h,
            invoice_count_48h=invoice_count_48h,
            invoice_sum_48h=invoice_sum_48h,
            freq_change_rate=freq_change_rate,
            amount_zscore_30d=amount_zscore_30d,
            invoice_spacing_cv=invoice_spacing_cv,
            small_invoice_cluster_14d=small_invoice_cluster_14d,
            amount_vs_category_avg=amount_vs_category_avg,
            invoice_freq_vs_category=invoice_freq_vs_category,
        )
        X = self._scaler.transform(X_raw)

        # ── Stage 1: IsolationForest (fast, high recall) ───────────────────────
        raw_score  = float(self._model.decision_function(X)[0])
        normalized = float(np.clip(0.5 - raw_score, 0.0, 1.0))

        # ── Stage 2: RandomForest precision filter (if available) ────────────
        if self._clf2 is not None:
            X_2nd   = np.hstack([[[normalized]], X_raw]).astype(np.float32)
            rf_prob = float(self._clf2.predict_proba(X_2nd)[0, 1])
            ml_score = float(np.clip(0.4 * normalized + 0.6 * rf_prob, 0.0, 1.0))
        else:
            ml_score = normalized

        # ── Stage 3: Rule-based fraud engine ────────────────────────────────
        is_weekend = transaction_date.weekday() >= 5
        is_round   = bool(X_raw[0, FEATURE_NAMES.index("is_round_amount")] >= 1.0)
        rule_result: RuleResult = evaluate_rules(
            amount=amount,
            is_weekend=is_weekend,
            amount_vs_vendor_avg=float(X_raw[0, FEATURE_NAMES.index("amount_vs_vendor_avg")]),
            consecutive_small_invoices=consecutive_small_invoices,
            invoice_count_24h=invoice_count_24h,
            invoice_count_48h=invoice_count_48h,
            invoice_count_7d=invoice_count_7d,
            invoice_sum_48h=invoice_sum_48h,
            vendor_avg_amount=vendor_avg_amount,
            is_round_amount=is_round,
            days_since_last_invoice=days_since_last_invoice,
        )

        # ── Stage 4: Vendor risk profile ────────────────────────────────────
        vendor_risk  = self._vendor_risk.score_vendor(vendor_id)

        # ── Stage 5: Graph / network risk ──────────────────────────────────
        graph_risk   = self._graph_fraud.score_vendor(vendor_id)

        # ── Hybrid blended final score ──────────────────────────────────────
        anomaly_score = float(np.clip(
            0.50 * ml_score
            + 0.25 * rule_result.rule_score
            + 0.15 * vendor_risk
            + 0.10 * graph_risk,
            0.0, 1.0,
        ))
        is_anomaly = anomaly_score >= self._threshold

        # SHAP explanation — TreeExplainer on IsolationForest returns
        # shap_values as ndarray (n_samples, n_features)
        sv         = self._explainer.shap_values(X)
        shap_row   = sv[0] if sv.ndim == 2 else sv  # handle both shapes

        shap_list = [
            {
                "feature":     FEATURE_NAMES[i],
                "raw_value":   float(X_raw[0, i]),
                "shap_impact": float(shap_row[i]),
            }
            for i in range(N_FEATURES)
        ]
        # Sort by absolute impact (most influential first)
        shap_list.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)

        top_reason = self._build_reason(shap_list[0], amount)
        # Prepend rule explanations to top_reason when rules fire
        if rule_result.triggered_rules:
            rule_summary = "; ".join(rule_result.triggered_rules)
            top_reason = f"[Rules: {rule_summary}] {top_reason}"

        return AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            threshold=self._threshold,
            raw_decision_value=raw_score,
            shap_values=shap_list,
            top_reason=top_reason,
            ml_score=round(ml_score, 4),
            rule_score=round(rule_result.rule_score, 4),
            triggered_rules=rule_result.triggered_rules,
            rule_details=rule_result.rule_details,
            vendor_risk_score=round(vendor_risk, 4),
            graph_risk_score=round(graph_risk, 4),
        )

    # ── Human-readable reason ──────────────────────────────────────────────────

    @staticmethod
    def _build_reason(top_feature: Dict, amount: float) -> str:
        feat   = top_feature["feature"]
        impact = top_feature["shap_impact"]
        raw    = top_feature["raw_value"]
        dirn   = "elevated" if impact > 0 else "suppressed"

        messages: Dict[str, str] = {
            "amount_log":
                f"Transaction amount ${amount:,.2f} is unusually {dirn}",
            "vendor_30d_velocity":
                f"Vendor 30-day spend velocity is {dirn} (${raw:,.2f} in window)",
            "days_since_last_invoice":
                f"Gap since last invoice ({raw:.1f} days) is anomalous for this vendor",
            "is_weekend_payment":
                "Payment issued on a weekend — unusual in standard procurement cycles",
            "amount_vs_vendor_avg":
                f"Amount is {raw:.2f}× this vendor's historical average",
            "invoice_count_7d":
                f"Vendor submitted {int(raw)} invoice(s) in the last 7 days (velocity burst)",
            "invoice_count_30d":
                f"Vendor submitted {int(raw)} invoice(s) in the last 30 days (elevated volume)",
            "rolling_7d_sum":
                f"Vendor's 7-day rolling spend is ${raw:,.2f} (unusual short-window burst)",
            "max_invoices_7d_window":
                f"Historical burst rate ({int(raw)} invoices/7-day window) is anomalous",
            "consecutive_small_invoices":
                f"{int(raw)} consecutive invoices under the $10k split-billing threshold",
            "amount_std_30d":
                f"Amount variability in last 30 days (std=${raw:,.2f}) is abnormal",
            "is_round_amount":
                f"Transaction amount ${amount:,.2f} is a suspicious round number",
            "invoice_count_24h":
                f"{int(raw)} invoice(s) submitted within the last 24 hours — extreme velocity burst",
            "invoice_count_48h":
                f"{int(raw)} invoice(s) submitted within the last 48 hours — rush billing pattern",
            "invoice_sum_48h":
                f"48-hour spend total ${raw:,.2f} — concentrated short-window payment burst",
            "approval_limit_proximity":
                f"Amount ${amount:,.2f} is within 5% below a federal approval threshold (threshold manipulation)",
        }
        return messages.get(feat, f"Feature '{feat}' is anomalous (shap={impact:.4f})")


# ── Module-level singleton ────────────────────────────────────────────────────

_scorer_singleton: Optional[ProcurementScorer] = None


def get_scorer() -> ProcurementScorer:
    """FastAPI dependency — returns the loaded singleton scorer."""
    global _scorer_singleton
    if _scorer_singleton is None:
        _scorer_singleton = ProcurementScorer()
        _scorer_singleton.load()
    return _scorer_singleton
