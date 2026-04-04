"""
backend/ml/augment.py
──────────────────────────────────────────────────────────────────────────────
Synthetic data augmentation for IsolationForest training and evaluation.

WHY this is needed
──────────────────
With federal procurement data from USAspending.gov, almost every contract
award goes to a distinct vendor:  95 vendors, 99 transactions → 1 tx/vendor.
This makes the velocity and time-gap features useless (they're all 0).

WHAT we generate
────────────────
1. expand_vendor_history()
   Adds 15 synthetic *normal* historical transactions per real vendor,
   dated BEFORE the real transactions, so that:
   - vendor_30d_velocity  →  has realistic signal
   - days_since_last_invoice →  has realistic signal
   - amount_vs_vendor_avg →  uses a proper historical average

2. create_evaluation_set()
   Injects 5 types of labeled procurement fraud patterns AFTER the
   training window, so the model can be evaluated with true precision
   / recall / F1 scores.

   Fraud types injected (total: ~100 anomalies, 200 normal):
   A. WEEKEND_SPIKE      – large payment on Saturday/Sunday
   B. SPLIT_BILLING      – 3 invoices clustered just below $10k threshold
   C. VELOCITY_SPIKE     – sudden burst of 5 invoices in one week
   D. RATIO_OUTLIER      – single invoice 7-15× vendor historical average
   E. ROUND_NUMBER       – exact round-number large invoice

The evaluation set labels are:  0 = normal,  1 = anomaly
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

NORMAL_LABEL  = 0
ANOMALY_LABEL = 1

# Procurement approval threshold used by most US federal agencies
_SPLIT_THRESHOLD = 10_000.0
_ROUND_AMOUNTS   = [50_000, 100_000, 250_000, 500_000, 1_000_000]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic normal history
# ─────────────────────────────────────────────────────────────────────────────

def expand_vendor_history(
    real_df: pd.DataFrame,
    txns_per_vendor: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic *normal* historical transactions per vendor.

    Dates are placed 10–365 days BEFORE each vendor's earliest real transaction
    so the temporal lag features (velocity, gap, avg) become meaningful.

    Parameters
    ----------
    real_df           : DataFrame with columns [vendor_id, amount, date]
    txns_per_vendor   : number of synthetic transactions to add per vendor
    seed              : random seed for reproducibility

    Returns
    -------
    DataFrame of synthetic rows only (same schema as real_df, plus
    'is_synthetic' and 'anomaly_type' columns).
    """
    rng = np.random.default_rng(seed)
    real_df = real_df.copy()
    real_df["date"] = pd.to_datetime(real_df["date"], utc=True)

    synthetic_rows = []

    for vendor_id, grp in real_df.groupby("vendor_id"):
        vendor_median = float(grp["amount"].median())
        log_mean      = np.log(max(vendor_median, 1.0))
        log_std       = 0.25  # 25% coefficient of variation in log space

        earliest_date = grp["date"].min()

        # Generate dates SEQUENTIALLY walking backward from earliest_date.
        # Each invoice is spaced 14–90 days before the previous one, matching
        # typical government procurement cycles (net-30 to quarterly billing).
        # This prevents accidental short gaps that would corrupt the
        # days_since_last_invoice feature and make velocity spikes invisible.
        current_date = earliest_date
        for _ in range(txns_per_vendor):
            interval = int(rng.integers(14, 91))   # 2 weeks – 3 months
            current_date = current_date - pd.Timedelta(days=interval)
            date = current_date

            # 95% weekday — shift Saturday → Friday, Sunday → Monday
            if rng.random() < 0.95:
                if date.weekday() == 5:    # Saturday → Friday
                    date -= pd.Timedelta(days=1)
                elif date.weekday() == 6:  # Sunday → Monday
                    date += pd.Timedelta(days=1)

            amount = float(np.exp(rng.normal(log_mean, log_std)))
            amount = max(amount, 100.0)

            synthetic_rows.append({
                "vendor_id":    vendor_id,
                "amount":       amount,
                "date":         date,
                "is_synthetic": True,
                "anomaly_type": "NORMAL",
            })

    if not synthetic_rows:
        return pd.DataFrame(columns=["vendor_id", "amount", "date",
                                     "is_synthetic", "anomaly_type"])
    return pd.DataFrame(synthetic_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Labeled evaluation set
# ─────────────────────────────────────────────────────────────────────────────

def create_evaluation_set(
    real_df: pd.DataFrame,
    n_normal: int  = 200,
    n_per_type: int = 20,
    seed: int = 777,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build a fully-labeled evaluation dataset placed AFTER the real data window.

    All anomalies are constructed from real vendor statistics, so the model
    must generalise to new transaction patterns — not just memorise training
    distributions.

    Returns
    -------
    eval_df : pd.DataFrame  (vendor_id, amount, date, is_synthetic, anomaly_type)
    labels  : np.ndarray shape (len(eval_df),)  0=normal, 1=anomaly
    """
    rng     = np.random.default_rng(seed)
    real_df = real_df.copy()
    real_df["date"] = pd.to_datetime(real_df["date"], utc=True)

    # Per-vendor statistics from full history
    vendor_stats = (
        real_df.groupby("vendor_id")["amount"]
        .agg(median="median", std="std")
        .reset_index()
    )
    vendor_stats["std"] = vendor_stats["std"].fillna(vendor_stats["median"] * 0.3)
    vendor_stats["std"] = vendor_stats["std"].clip(lower=100.0)

    # Base date: 1 day after the most recent real transaction
    base_date = real_df["date"].max() + pd.Timedelta(days=1)

    rows:   list = []
    labels: list = []

    def _sample_vendor() -> pd.Series:
        return vendor_stats.iloc[rng.integers(len(vendor_stats))]

    # ── Normal transactions ────────────────────────────────────────────────────
    for i in range(n_normal):
        v      = _sample_vendor()
        offset = int(rng.integers(1, 180))
        date   = base_date + pd.Timedelta(days=offset)
        if rng.random() < 0.95:
            if date.weekday() == 5:    date -= pd.Timedelta(days=1)
            elif date.weekday() == 6:  date += pd.Timedelta(days=1)

        log_mean = np.log(max(float(v["median"]), 1.0))
        amount   = max(float(np.exp(rng.normal(log_mean, 0.25))), 100.0)
        rows.append({"vendor_id": v["vendor_id"], "amount": amount, "date": date,
                     "is_synthetic": True, "anomaly_type": "NORMAL"})
        labels.append(NORMAL_LABEL)

    # ── Anomaly A: WEEKEND_SPIKE ───────────────────────────────────────────────
    for _ in range(n_per_type):
        v      = _sample_vendor()
        offset = int(rng.integers(1, 90))
        date   = base_date + pd.Timedelta(days=offset)
        # Force Saturday (5) or Sunday (6)
        dow   = date.weekday()
        shift = (5 - dow) % 7
        date  = date + pd.Timedelta(days=shift if shift != 0 else 6)
        amount = float(v["median"]) * float(rng.uniform(3.0, 8.0))
        rows.append({"vendor_id": v["vendor_id"], "amount": amount, "date": date,
                     "is_synthetic": True, "anomaly_type": "WEEKEND_SPIKE"})
        labels.append(ANOMALY_LABEL)

    # ── Anomaly B: SPLIT_BILLING (3 invoices per set) ─────────────────────────
    # Split billing = breaking a large contract into sub-threshold invoices.
    # Only pick vendors with median >> $10k so the small invoices look anomalous.
    large_vendors = vendor_stats[vendor_stats["median"] > 50_000]
    if len(large_vendors) == 0:
        large_vendors = vendor_stats  # fallback if no large vendors
    n_sets = max(1, n_per_type // 3)
    for _ in range(n_sets):
        v      = large_vendors.iloc[rng.integers(len(large_vendors))]
        offset = int(rng.integers(1, 80))
        base   = base_date + pd.Timedelta(days=offset)
        for day_offset in [0, 1, 2]:
            date   = base + pd.Timedelta(days=day_offset)
            amount = float(rng.uniform(_SPLIT_THRESHOLD * 0.92, _SPLIT_THRESHOLD * 0.998))
            rows.append({"vendor_id": v["vendor_id"], "amount": amount, "date": date,
                         "is_synthetic": True, "anomaly_type": "SPLIT_BILLING"})
            labels.append(ANOMALY_LABEL)

    # ── Anomaly C: VELOCITY_SPIKE (5 invoices in 7 days) ─────────────────────
    n_bursts = max(1, n_per_type // 5)
    for _ in range(n_bursts):
        v      = _sample_vendor()
        offset = int(rng.integers(1, 80))
        base   = base_date + pd.Timedelta(days=offset)
        for day_offset in range(5):
            date   = base + pd.Timedelta(days=day_offset)
            amount = float(v["median"]) * float(rng.uniform(0.85, 1.15))
            rows.append({"vendor_id": v["vendor_id"], "amount": amount, "date": date,
                         "is_synthetic": True, "anomaly_type": "VELOCITY_SPIKE"})
            labels.append(ANOMALY_LABEL)

    # ── Anomaly D: RATIO_OUTLIER (20-50× vendor typical amount) ─────────────
    # 7-15× is too small for federal data (real contracts vary by 50×+).
    # Using 20-50× ensures the ratio is well outside the training distribution.
    for _ in range(n_per_type):
        v      = _sample_vendor()
        offset = int(rng.integers(1, 90))
        date   = base_date + pd.Timedelta(days=offset)
        if rng.random() < 0.95:
            if date.weekday() == 5:    date -= pd.Timedelta(days=1)
            elif date.weekday() == 6:  date += pd.Timedelta(days=1)
        amount = float(v["median"]) * float(rng.uniform(20.0, 50.0))
        rows.append({"vendor_id": v["vendor_id"], "amount": amount, "date": date,
                     "is_synthetic": True, "anomaly_type": "RATIO_OUTLIER"})
        labels.append(ANOMALY_LABEL)

    # ── Anomaly E: ROUND_NUMBER ────────────────────────────────────────────────
    for _ in range(n_per_type):
        v      = _sample_vendor()
        offset = int(rng.integers(1, 90))
        date   = base_date + pd.Timedelta(days=offset)
        if rng.random() < 0.95:
            if date.weekday() == 5:    date -= pd.Timedelta(days=1)
            elif date.weekday() == 6:  date += pd.Timedelta(days=1)
        amount = float(rng.choice(_ROUND_AMOUNTS))
        rows.append({"vendor_id": v["vendor_id"], "amount": amount, "date": date,
                     "is_synthetic": True, "anomaly_type": "ROUND_NUMBER"})
        labels.append(ANOMALY_LABEL)

    # ── Anomaly F: BURST_24H (>4 invoices within 24 hours) ────────────────────
    # Fraudsters rush to submit many invoices before detection cutoff.
    # 4 invoices in a single day is nearly impossible in normal procurement.
    n_24h_bursts = max(1, n_per_type // 4)
    for _ in range(n_24h_bursts):
        v      = _sample_vendor()
        offset = int(rng.integers(1, 80))
        base   = base_date + pd.Timedelta(days=offset)
        # Inject 5 invoices within the same 24-hour window
        for h_offset in range(5):
            date   = base + pd.Timedelta(hours=h_offset * 3)  # every 3 hours
            amount = float(v["median"]) * float(rng.uniform(0.80, 1.20))
            rows.append({"vendor_id": v["vendor_id"], "amount": amount, "date": date,
                         "is_synthetic": True, "anomaly_type": "BURST_24H"})
            labels.append(ANOMALY_LABEL)

    # ── Anomaly G: APPROVAL_LIMIT (amount just below federal approval threshold) ─
    # Classic split: amount is 0-5% below $10k, $25k, $100k, or $250k threshold
    # to avoid competitive bidding or additional oversight.
    _fed_limits = [10_000.0, 25_000.0, 100_000.0, 250_000.0]
    for _ in range(n_per_type):
        v      = _sample_vendor()
        offset = int(rng.integers(1, 90))
        date   = base_date + pd.Timedelta(days=offset)
        if rng.random() < 0.95:
            if date.weekday() == 5:    date -= pd.Timedelta(days=1)
            elif date.weekday() == 6:  date += pd.Timedelta(days=1)
        # Pick a random threshold and land just below it (95–99.9% of limit)
        lim    = float(rng.choice(_fed_limits))
        amount = lim * float(rng.uniform(0.950, 0.999))
        rows.append({"vendor_id": v["vendor_id"], "amount": amount, "date": date,
                     "is_synthetic": True, "anomaly_type": "APPROVAL_LIMIT"})
        labels.append(ANOMALY_LABEL)

    eval_df = pd.DataFrame(rows)
    eval_df["date"] = pd.to_datetime(eval_df["date"], utc=True)

    return eval_df, np.array(labels, dtype=np.int32)
