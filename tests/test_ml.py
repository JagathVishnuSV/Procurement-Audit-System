"""
tests/test_ml.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the ML layer (Sprint 2).

Runs entirely in-process — no PostgreSQL, no model file, no network.
pandas and numpy are the only non-stdlib deps required.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from backend.ml.features import (
    FEATURE_NAMES,
    N_FEATURES,
    build_feature_matrix,
    build_single_feature_vector,
)


# ── FEATURE_NAMES contract ─────────────────────────────────────────────────────

class TestFeatureContract:
    def test_count(self):
        assert N_FEATURES == 22
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_required_names_present(self):
        for name in [
            "amount_log",
            "vendor_30d_velocity",
            "days_since_last_invoice",
            "is_weekend_payment",
            "amount_vs_vendor_avg",
        ]:
            assert name in FEATURE_NAMES

    def test_order_stable(self):
        """Order must be stable — scaler expects the same column order every time."""
        assert FEATURE_NAMES[0] == "amount_log"
        assert FEATURE_NAMES[3] == "is_weekend_payment"


# ── build_single_feature_vector ────────────────────────────────────────────────

class TestSingleVector:
    WED = datetime(2026, 1, 14, 12, 0, tzinfo=timezone.utc)   # Wednesday
    SAT = datetime(2026, 1, 10, 10, 0, tzinfo=timezone.utc)   # Saturday
    SUN = datetime(2026, 1, 11,  9, 0, tzinfo=timezone.utc)   # Sunday

    def test_output_shape(self):
        v = build_single_feature_vector(1000.0, self.WED)
        assert v.shape == (1, N_FEATURES)

    def test_dtype_float32(self):
        v = build_single_feature_vector(1000.0, self.WED)
        assert v.dtype == np.float32

    def test_amount_log_correct(self):
        v = build_single_feature_vector(1000.0, self.WED)
        assert abs(float(v[0, 0]) - math.log1p(1000.0)) < 1e-5

    def test_zero_amount_safe(self):
        v = build_single_feature_vector(0.0, self.WED)
        assert float(v[0, 0]) == pytest.approx(0.0)

    def test_negative_amount_clamped(self):
        """Negative amounts should not cause NaN in log — clamped to 0."""
        v = build_single_feature_vector(-500.0, self.WED)
        assert not np.isnan(v).any()
        assert float(v[0, 0]) == pytest.approx(0.0)  # log1p(max(-500,0)) = log1p(0) = 0

    def test_weekend_saturday(self):
        v = build_single_feature_vector(100.0, self.SAT)
        assert float(v[0, 3]) == 1.0

    def test_weekend_sunday(self):
        v = build_single_feature_vector(100.0, self.SUN)
        assert float(v[0, 3]) == 1.0

    def test_weekday_not_flagged(self):
        v = build_single_feature_vector(100.0, self.WED)
        assert float(v[0, 3]) == 0.0

    def test_vendor_features_passed_through(self):
        v = build_single_feature_vector(
            amount=500.0,
            date=self.WED,
            vendor_30d_spend=10_000.0,
            days_since_last_invoice=3.0,
            vendor_avg_amount=400.0,
        )
        assert float(v[0, 1]) == pytest.approx(10_000.0)  # vendor_30d_velocity
        assert float(v[0, 2]) == pytest.approx(3.0)       # days_since_last_invoice
        expected_ratio = 500.0 / (400.0 + 1.0)
        assert float(v[0, 4]) == pytest.approx(expected_ratio, rel=1e-5)


# ── build_feature_matrix ───────────────────────────────────────────────────────

def _make_df(
    vendor_ids=("v1", "v1", "v1", "v2", "v2"),
    amounts=(1000.0, 2000.0, 1500.0, 500.0, 600.0),
    dates=("2026-01-01", "2026-01-15", "2026-01-20", "2026-01-05", "2026-01-25"),
) -> pd.DataFrame:
    return pd.DataFrame({
        "vendor_id": list(vendor_ids),
        "amount":    list(amounts),
        "date":      pd.to_datetime(list(dates), utc=True),
    })


class TestBuildFeatureMatrix:
    def test_shape(self):
        X = build_feature_matrix(_make_df())
        assert X.shape == (5, N_FEATURES)

    def test_dtype(self):
        X = build_feature_matrix(_make_df())
        assert X.dtype == np.float32

    def test_no_nans(self):
        X = build_feature_matrix(_make_df())
        assert not np.isnan(X).any()

    def test_no_leakage_on_first_tx(self):
        """
        First transaction per vendor must have 0 30d-velocity and 0 dsli.
        If data were leaked, velocity would be non-zero.
        """
        df = pd.DataFrame({
            "vendor_id": ["v1", "v1"],
            "amount":    [1000.0, 2000.0],
            "date":      pd.to_datetime(["2026-01-01", "2026-01-10"], utc=True),
        })
        X = build_feature_matrix(df)
        assert X[0, 1] == pytest.approx(0.0)  # vendor_30d_velocity
        assert X[0, 2] == pytest.approx(0.0)  # days_since_last_invoice

    def test_amount_log_values(self):
        df = pd.DataFrame({
            "vendor_id": ["v1"],
            "amount":    [99.0],
            "date":      pd.to_datetime(["2026-01-01"], utc=True),
        })
        X = build_feature_matrix(df)
        assert abs(float(X[0, 0]) - math.log1p(99.0)) < 1e-5

    def test_weekend_flag_in_matrix(self):
        df = pd.DataFrame({
            "vendor_id": ["v1", "v1"],
            "amount":    [1000.0, 1000.0],
            # 2026-01-09 = Friday (0), 2026-01-10 = Saturday (1)
            "date":      pd.to_datetime(["2026-01-09", "2026-01-10"], utc=True),
        })
        X = build_feature_matrix(df)
        assert float(X[0, 3]) == 0.0   # Friday
        assert float(X[1, 3]) == 1.0   # Saturday

    def test_30d_velocity_accumulates(self):
        """Third transaction should include amounts from first two."""
        df = pd.DataFrame({
            "vendor_id": ["v1", "v1", "v1"],
            "amount":    [100.0, 200.0, 50.0],
            "date":      pd.to_datetime(
                ["2026-01-01", "2026-01-05", "2026-01-20"], utc=True
            ),
        })
        X = build_feature_matrix(df)
        # Third tx velocity = 100 + 200 = 300 (both within 30 days)
        assert float(X[2, 1]) == pytest.approx(300.0)

    def test_velocity_excludes_old_tx(self):
        """Transactions older than 30 days must not contribute to velocity."""
        df = pd.DataFrame({
            "vendor_id": ["v1", "v1"],
            "amount":    [500.0, 100.0],
            # Second tx is 40 days after first → first is outside the window
            "date":      pd.to_datetime(["2026-01-01", "2026-02-10"], utc=True),
        })
        X = build_feature_matrix(df)
        assert float(X[1, 1]) == pytest.approx(0.0)  # outside 30-day window

    def test_single_vendor_single_row(self):
        df = pd.DataFrame({
            "vendor_id": ["v1"],
            "amount":    [1000.0],
            "date":      pd.to_datetime(["2026-01-15"], utc=True),
        })
        X = build_feature_matrix(df)
        assert X.shape == (1, N_FEATURES)
        assert not np.isnan(X).any()


# ── ProcurementScorer static helpers (no model file needed) ───────────────────

class TestScorerHelpers:
    def test_build_reason_amount_elevated(self):
        from backend.ml.scorer import ProcurementScorer
        scorer  = ProcurementScorer()
        feature = {"feature": "amount_log", "shap_impact": 0.35, "raw_value": 9.21}
        reason  = scorer._build_reason(feature, 10_000.0)
        assert "10,000.00" in reason
        assert "elevated" in reason

    def test_build_reason_amount_suppressed(self):
        from backend.ml.scorer import ProcurementScorer
        scorer  = ProcurementScorer()
        feature = {"feature": "amount_log", "shap_impact": -0.35, "raw_value": 9.21}
        reason  = scorer._build_reason(feature, 10_000.0)
        assert "suppressed" in reason

    def test_build_reason_weekend(self):
        from backend.ml.scorer import ProcurementScorer
        scorer  = ProcurementScorer()
        feature = {"feature": "is_weekend_payment", "shap_impact": 0.2, "raw_value": 1.0}
        reason  = scorer._build_reason(feature, 500.0)
        assert "weekend" in reason.lower()

    def test_build_reason_velocity(self):
        from backend.ml.scorer import ProcurementScorer
        scorer  = ProcurementScorer()
        feature = {"feature": "vendor_30d_velocity", "shap_impact": 0.4, "raw_value": 50_000.0}
        reason  = scorer._build_reason(feature, 100.0)
        assert "50,000.00" in reason

    def test_score_normalisation_bounds(self):
        """Verify the [0,1] normalisation formula."""
        # decision_function=-0.5 → most anomalous → score=1.0
        assert float(np.clip(0.5 - (-0.5), 0.0, 1.0)) == pytest.approx(1.0)
        # decision_function=+0.5 → most normal → score=0.0
        assert float(np.clip(0.5 - 0.5, 0.0, 1.0)) == pytest.approx(0.0)
        # decision_function=0.0 → boundary → score=0.5
        assert float(np.clip(0.5 - 0.0, 0.0, 1.0)) == pytest.approx(0.5)

    def test_scorer_raises_if_not_loaded(self):
        from backend.ml.scorer import ProcurementScorer
        scorer = ProcurementScorer()  # not loaded
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer.score_transaction(
                amount=1000.0,
                transaction_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            )
