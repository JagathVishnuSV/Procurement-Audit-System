"""
backend/ml/features.py
──────────────────────────────────────────────────────────────────────────────
Feature engineering for procurement anomaly detection.

Feature vector – 22 dimensions (FEATURE_NAMES order):
  0.  amount_log                  – log1p(amount), normalises heavy right tail
  1.  vendor_30d_velocity         – vendor's total spend in the 30 days BEFORE this tx
  2.  days_since_last_invoice     – days since vendor's previous transaction
  3.  is_weekend_payment          – 1 if Saturday/Sunday, else 0
  4.  amount_vs_vendor_avg        – amount / vendor hist mean (expanding lag-1)
  5.  invoice_count_7d            – # vendor invoices in prior 7 days (velocity)
  6.  invoice_count_30d           – # vendor invoices in prior 30 days
  7.  rolling_7d_sum              – sum of vendor amounts in prior 7 days
  8.  max_invoices_7d_window      – max # invoices seen in any prior 7-day window
                                    (vendor's baseline burst rate)
  9.  consecutive_small_invoices  – # consecutive prior invoices < $10k (split billing)
  10. amount_std_30d              – std of vendor amounts in prior 30 days
  11. is_round_amount             – 1 if amount is exact multiple of $1k or $500≥$10k
  12. invoice_count_24h           – # vendor invoices in prior 24 hours (ultra-burst)
  13. invoice_count_48h           – # vendor invoices in prior 48 hours
  14. invoice_sum_48h             – vendor spend sum in prior 48 hours
  15. approval_limit_proximity    – 1 if amount is 0-5 pct below a federal approval
                                    threshold ($10k/$25k/$100k/$250k), else 0
  16. freq_change_rate            – BEHAVIORAL: invoice_count_7d / (invoice_count_30d/4 + ε)
                                    detects sudden velocity spikes vs vendor monthly baseline
  17. amount_zscore_30d          – BEHAVIORAL: (amount - vendor_30d_mean) / (vendor_30d_std + ε)
                                    standardised deviation from vendor's recent norm
  18. invoice_spacing_cv         – BEHAVIORAL: coeff of variation of inter-invoice gaps
                                    irregular spacing = unusual billing rhythm
  19. small_invoice_cluster_14d  – BEHAVIORAL: # invoices < $10k in prior 14 days
                                    persistent split-billing footprint
  20. amount_vs_category_avg     – PEER: amount / mean amount of all vendors sharing same
                                    awarding agency; outlier among agency peers
  21. invoice_freq_vs_category   – PEER: vendor 30d invoice count / agency-average 30d
                                    count per vendor; unusually active in the agency

All values are float32. NaN-safe: missing values filled with 0.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

# ── Public contract ────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    "amount_log",               # 0  – scalar, computed from amount
    "vendor_30d_velocity",      # 1  – 30-day prior spend
    "days_since_last_invoice",  # 2  – time gap
    "is_weekend_payment",       # 3  – flag
    "amount_vs_vendor_avg",     # 4  – ratio to expanding lag-1 mean
    "invoice_count_7d",         # 5  – short-burst velocity count
    "invoice_count_30d",        # 6  – monthly volume
    "rolling_7d_sum",           # 7  – short-burst spend sum
    "max_invoices_7d_window",   # 8  – vendor's historical max burst
    "consecutive_small_invoices", # 9 – split-billing signal
    "amount_std_30d",           # 10 – 30-day amount variability
    "is_round_amount",          # 11 – round-number fraud flag
    "invoice_count_24h",        # 12 – ultra-short burst (24h)
    "invoice_count_48h",        # 13 – 48-hour burst count
    "invoice_sum_48h",          # 14 – 48-hour burst spend
    "approval_limit_proximity", # 15 – near federal approval threshold
    "freq_change_rate",         # 16 BEHAVIORAL – weekly velocity vs monthly baseline
    "amount_zscore_30d",        # 17 BEHAVIORAL – amount deviation from vendor norm
    "invoice_spacing_cv",       # 18 BEHAVIORAL – irregularity in billing rhythm
    "small_invoice_cluster_14d",# 19 BEHAVIORAL – sub-$10k invoice density (split billing)
    "amount_vs_category_avg",   # 20 PEER – vendor amount vs agency-level avg
    "invoice_freq_vs_category", # 21 PEER – vendor invoice rate vs agency peers
]

N_FEATURES: int = len(FEATURE_NAMES)


# ── Batch feature matrix (used by trainer) ────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build a training feature matrix from a DataFrame of historical transactions.

    Required columns
    ----------------
    vendor_id : str or UUID-compatible
    amount    : float  (USD)
    date      : datetime (timezone-aware or naive; will be coerced to UTC)

    Returns
    -------
    X : np.ndarray  shape (n_samples, N_FEATURES)  dtype float32
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # ── scalar features (no vendor history needed) ────────────────────────────
    df["amount_log"]         = np.log1p(df["amount"].clip(lower=0).astype(float))
    df["is_weekend_payment"] = (df["date"].dt.dayofweek >= 5).astype(float)

    def _is_round(x: float) -> float:
        if x >= 10_000 and abs(x % 500) < 0.01:
            return 1.0
        if abs(x % 1_000) < 0.01:
            return 1.0
        return 0.0

    df["is_round_amount"] = df["amount"].apply(_is_round)

    # ── per-vendor temporal features ──────────────────────────────────────────
    n  = len(df)
    _24h = np.timedelta64(24, "h")
    _48h = np.timedelta64(48, "h")
    _7d  = np.timedelta64(7,  "D")
    _30d = np.timedelta64(30, "D")
    _10k = 10_000.0
    # US federal simplified-acquisition thresholds (avoid-oversight split points)
    _APPROVAL_LIMITS = [10_000.0, 25_000.0, 100_000.0, 250_000.0]

    rows_30d_vel      = np.zeros(n, dtype=float)
    rows_dsli         = np.zeros(n, dtype=float)
    rows_ava          = np.zeros(n, dtype=float)
    rows_inv7d        = np.zeros(n, dtype=float)
    rows_inv30d       = np.zeros(n, dtype=float)
    rows_sum7d        = np.zeros(n, dtype=float)
    rows_max7d_win    = np.zeros(n, dtype=float)
    rows_consec_sm    = np.zeros(n, dtype=float)
    rows_std30d       = np.zeros(n, dtype=float)
    rows_inv24h       = np.zeros(n, dtype=float)
    rows_inv48h       = np.zeros(n, dtype=float)
    rows_sum48h       = np.zeros(n, dtype=float)
    rows_freq_chg     = np.zeros(n, dtype=float)
    rows_amt_zscore   = np.zeros(n, dtype=float)
    rows_spacing_cv   = np.zeros(n, dtype=float)
    rows_sm_cluster14 = np.zeros(n, dtype=float)

    for _vid, grp in df.groupby("vendor_id", sort=False):
        grp  = grp.sort_values("date")
        idxs = grp.index.tolist()
        dts  = grp["date"].values           # datetime64[ns, UTC] – sorted
        amts = grp["amount"].values.astype(float)

        for pos, orig_idx in enumerate(idxs):
            cur_date  = dts[pos]
            prev_dts  = dts[:pos]            # strictly prior – no leakage
            prev_amts = amts[:pos]

            mask_30d  = prev_dts >= (cur_date - _30d)
            mask_7d   = prev_dts >= (cur_date - _7d)
            amts_30d  = prev_amts[mask_30d]
            amts_7d   = prev_amts[mask_7d]

            # 30-day spend velocity
            rows_30d_vel[orig_idx] = float(amts_30d.sum())

            # days since last invoice
            if pos > 0:
                delta_s = (cur_date - prev_dts[-1]).astype("timedelta64[s]").astype(float)
                rows_dsli[orig_idx] = delta_s / 86400.0

            # amount vs expanding lag-1 vendor average
            hist_mean = float(prev_amts.mean()) if pos > 0 else amts[0]
            rows_ava[orig_idx] = amts[pos] / (hist_mean + 1.0)

            # invoice counts in prior windows
            rows_inv7d[orig_idx]  = float(mask_7d.sum())
            rows_inv30d[orig_idx] = float(mask_30d.sum())

            # rolling 7-day spend sum
            rows_sum7d[orig_idx] = float(amts_7d.sum())

            # amount std in prior 30-day window (≥2 samples)
            if len(amts_30d) >= 2:
                rows_std30d[orig_idx] = float(amts_30d.std())

            # max invoice count in any prior 7-day rolling window
            # (captures vendor's historical burst baseline)
            if pos >= 2:
                max_ct = 0
                for d_j in prev_dts:
                    ct = int(((prev_dts >= d_j) & (prev_dts < d_j + _7d)).sum())
                    if ct > max_ct:
                        max_ct = ct
                rows_max7d_win[orig_idx] = float(max_ct)

            # consecutive prior invoices below $10k (split-billing detector)
            csm = 0
            for amt in reversed(prev_amts.tolist()):
                if amt < _10k:
                    csm += 1
                else:
                    break
            rows_consec_sm[orig_idx] = float(csm)

            # ultra-short burst windows (24h / 48h)
            mask_24h = prev_dts >= (cur_date - _24h)
            mask_48h = prev_dts >= (cur_date - _48h)
            rows_inv24h[orig_idx] = float(mask_24h.sum())
            rows_inv48h[orig_idx] = float(mask_48h.sum())
            rows_sum48h[orig_idx] = float(prev_amts[mask_48h].sum())

            # ── BEHAVIORAL FEATURES ──────────────────────────────────────────
            # 16. freq_change_rate: weekly rate vs monthly baseline
            monthly_baseline = float(mask_30d.sum()) / 4.0  # expected per week
            rows_freq_chg[orig_idx] = float(mask_7d.sum()) / (monthly_baseline + 0.1)

            # 17. amount_zscore_30d: deviation from recent vendor norm
            if len(amts_30d) >= 2:
                mean_30d = float(amts_30d.mean())
                std_30d  = float(amts_30d.std())
                rows_amt_zscore[orig_idx] = (amts[pos] - mean_30d) / (std_30d + 1.0)
            else:
                rows_amt_zscore[orig_idx] = 0.0

            # 18. invoice_spacing_cv: coeff of variation of inter-invoice gaps
            if pos >= 3:
                gaps = np.diff(
                    prev_dts[-min(pos, 10):].astype("datetime64[h]").astype(float)
                )
                if len(gaps) >= 2 and gaps.mean() > 0:
                    rows_spacing_cv[orig_idx] = float(gaps.std() / (gaps.mean() + 0.1))

            # 19. small_invoice_cluster_14d: sub-$10k count in last 14 days
            _14d = np.timedelta64(14, "D")
            mask_14d = prev_dts >= (cur_date - _14d)
            rows_sm_cluster14[orig_idx] = float(
                (prev_amts[mask_14d] < _10k).sum()
            )

    df["vendor_30d_velocity"]        = rows_30d_vel
    df["days_since_last_invoice"]    = rows_dsli
    df["amount_vs_vendor_avg"]       = rows_ava
    df["invoice_count_7d"]           = rows_inv7d
    df["invoice_count_30d"]          = rows_inv30d
    df["rolling_7d_sum"]             = rows_sum7d
    df["max_invoices_7d_window"]     = rows_max7d_win
    df["consecutive_small_invoices"] = rows_consec_sm
    df["amount_std_30d"]             = rows_std30d
    df["invoice_count_24h"]          = rows_inv24h
    df["invoice_count_48h"]          = rows_inv48h
    df["invoice_sum_48h"]            = rows_sum48h
    df["freq_change_rate"]           = rows_freq_chg
    df["amount_zscore_30d"]          = rows_amt_zscore
    df["invoice_spacing_cv"]         = rows_spacing_cv
    df["small_invoice_cluster_14d"]  = rows_sm_cluster14

    # ── Peer comparison features (vendor vs agency peers) ─────────────────────
    # Compares each vendor's amount and invoice frequency against the average
    # behaviour of all vendors billing the same awarding agency.  A vendor
    # that charges 10× the agency norm or invoices 20× as often as peers is a
    # strong outlier signal the self-history features cannot detect.
    if "awarding_agency" in df.columns:
        # Per-agency mean invoice amount (global, not time-windowed — stable baseline)
        _cat_mean = df.groupby("awarding_agency")["amount"].transform("mean")
        df["amount_vs_category_avg"] = (df["amount"] / (_cat_mean + 1.0)).fillna(1.0)

        # Per-agency average 30d invoice rate across all its vendors.
        # Step 1: mean invoice_count_30d per (agency, vendor) pair
        # Step 2: mean of those vendor means per agency → fair peer baseline
        _va_rate = (
            df.groupby(["awarding_agency", "vendor_id"])["invoice_count_30d"]
            .mean()
            .reset_index()
        )
        _agency_avg_rate = _va_rate.groupby("awarding_agency")["invoice_count_30d"].mean()
        _cat_rate = df["awarding_agency"].map(_agency_avg_rate).fillna(1.0)
        df["invoice_freq_vs_category"] = (
            df["invoice_count_30d"] / (_cat_rate + 0.1)
        ).fillna(1.0)
    else:
        df["amount_vs_category_avg"]   = 1.0
        df["invoice_freq_vs_category"] = 1.0

    # approval_limit_proximity: 1 if amount is within 5 pct BELOW a federal
    # approval threshold (classic avoid-oversight splitting pattern)
    def _approx_limit(x: float) -> float:
        for lim in [10_000.0, 25_000.0, 100_000.0, 250_000.0]:
            if lim * 0.95 <= x < lim:
                return 1.0
        return 0.0

    df["approval_limit_proximity"] = df["amount"].apply(_approx_limit)

    X = df[FEATURE_NAMES].fillna(0.0).values.astype(np.float32)
    return X


# ── Single-vector builder (used by live scorer) ───────────────────────────────

def build_single_feature_vector(
    amount: float,
    date: datetime,
    vendor_30d_spend: float = 0.0,
    days_since_last_invoice: float = 0.0,
    vendor_avg_amount: float = 0.0,
    # Feature-store lookups (populated by Redis / DB at API call time)
    invoice_count_7d: float = 0.0,
    invoice_count_30d: float = 0.0,
    rolling_7d_sum: float = 0.0,
    max_invoices_7d_window: float = 0.0,
    consecutive_small_invoices: float = 0.0,
    amount_std_30d: float = 0.0,
    invoice_count_24h: float = 0.0,
    invoice_count_48h: float = 0.0,
    invoice_sum_48h: float = 0.0,
    # Behavioral features
    freq_change_rate: float = 0.0,
    amount_zscore_30d: float = 0.0,
    invoice_spacing_cv: float = 0.0,
    small_invoice_cluster_14d: float = 0.0,
    # Peer comparison features (default 1.0 = vendor at agency average)
    amount_vs_category_avg: float = 1.0,
    invoice_freq_vs_category: float = 1.0,
) -> np.ndarray:
    """
    Build a (1, N_FEATURES) feature vector for a single live transaction.

    The first five parameters map to existing Redis keys.  The six new
    parameters should also be fetched from the Redis feature store (or a
    DB look-up) and default to 0 for backward compatibility.
    """
    is_weekend = float(date.weekday() >= 5)
    amount_log = math.log1p(max(amount, 0.0))
    ratio      = amount / (vendor_avg_amount + 1.0)

    def _is_round(x: float) -> float:
        if x >= 10_000 and abs(x % 500) < 0.01:
            return 1.0
        if abs(x % 1_000) < 0.01:
            return 1.0
        return 0.0

    # approval_limit_proximity: amount within 5% below a federal approval threshold
    def _approx_limit(x: float) -> float:
        for lim in [10_000.0, 25_000.0, 100_000.0, 250_000.0]:
            if lim * 0.95 <= x < lim:
                return 1.0
        return 0.0

    # freq_change_rate: derive from call-site values if not provided
    _freq_chg = freq_change_rate if freq_change_rate != 0.0 else (
        invoice_count_7d / ((invoice_count_30d / 4.0) + 0.1)
        if invoice_count_30d > 0 else 0.0
    )

    return np.array(
        [[
            amount_log,
            vendor_30d_spend,
            days_since_last_invoice,
            is_weekend,
            ratio,
            invoice_count_7d,
            invoice_count_30d,
            rolling_7d_sum,
            max_invoices_7d_window,
            consecutive_small_invoices,
            amount_std_30d,
            _is_round(amount),
            invoice_count_24h,
            invoice_count_48h,
            invoice_sum_48h,
            _approx_limit(amount),
            _freq_chg,
            amount_zscore_30d,
            invoice_spacing_cv,
            small_invoice_cluster_14d,
            amount_vs_category_avg,
            invoice_freq_vs_category,
        ]],
        dtype=np.float32,
    )
