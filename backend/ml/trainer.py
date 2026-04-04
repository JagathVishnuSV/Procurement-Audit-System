"""
backend/ml/trainer.py
──────────────────────────────────────────────────────────────────────────────
Train IsolationForest on procurement transaction data.

What makes this better than a naïve fit
────────────────────────────────────────
1.  Synthetic vendor history expansion
    With federal open-data (USAspending), most vendors appear only once.
    That makes velocity / gap / avg features near-zero for all samples.
    We inject 15 synthetic *normal* transactions per vendor (dated before the
    real ones) to give the model realistic per-vendor baselines, expanding
    the training corpus from ~100 → ~1,500 rows with full feature signal.

2.  Contamination auto-tuning  (--contamination auto)
    Instead of guessing, we sweep [0.01, 0.03, 0.05, 0.08, 0.10] and pick
    the value that maximises F1 on a held-out labeled synthetic test set.
    The winning value is saved to model_metadata.json.

3.  Post-training evaluation
    After training, the full evaluation pipeline runs automatically and saves
    the optimal_threshold — the score cutoff (0-1) that maximises F1 on
    labeled anomalies.  scorer.py reads this value at startup so the API
    never uses a hardcoded threshold.

Saved artefacts
───────────────
  models/isolation_forest.joblib        – trained model
  models/scaler.joblib                  – fitted StandardScaler
  models/model_metadata.json            – training + evaluation stats
  models/evaluation_report.json         – full precision/recall/F1 table
  models/shap_feature_importance.json   – mean |SHAP| per feature

Usage
─────
    python -m backend.ml.trainer                          # auto-tune contamination
    python -m backend.ml.trainer --contamination 0.05    # fixed contamination
    python -m backend.ml.trainer --n-estimators 300      # more trees
    python -m backend.ml.trainer --no-augment            # skip synthetic history
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sqlalchemy import text

from backend.config import get_settings
from backend.database import get_sync_session
from backend.ml.augment import create_evaluation_set, expand_vendor_history
from backend.ml.features import FEATURE_NAMES, N_FEATURES, build_feature_matrix
from backend.ml.graph_fraud import GraphFraudDetector
from backend.ml.vendor_risk import VendorRiskProfiler

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR     = Path("models")
MODEL_PATH    = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH   = MODEL_DIR / "scaler.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SECOND_STAGE_PATH = MODEL_DIR / "second_stage_clf.joblib"

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_N_ESTIMATORS = 300
# Denser grid – 0.005 for very clean data, 0.02 fills the 0.01–0.03 gap
CONTAMINATION_CANDIDATES: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """Pull all transactions with vendor_id from PostgreSQL."""
    query = text(
        "SELECT t.vendor_id::text AS vendor_id, "
        "       t.amount::float   AS amount, "
        "       t.date, "
        "       t.awarding_agency "
        "FROM transactions t "
        "ORDER BY t.date"
    )
    with get_sync_session() as session:
        rows = session.execute(query).fetchall()

    if not rows:
        logger.error(
            "No transactions in DB. "
            "Run: python -m backend.ingestion.seed --limit 200 --pages 10 --no-kafka"
        )
        sys.exit(1)

    df = pd.DataFrame(rows, columns=["vendor_id", "amount", "date", "awarding_agency"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    logger.info(
        "Loaded {} real transactions from {} unique vendors",
        len(df), df["vendor_id"].nunique(),
    )
    return df


# ── Contamination auto-tune ────────────────────────────────────────────────────

def _normalize(raw: np.ndarray) -> np.ndarray:
    """Map IsolationForest decision values → [0,1] anomaly score."""
    return np.clip(0.5 - raw, 0.0, 1.0)


def _score_contamination(
    contamination: float,
    X_train: np.ndarray,
    X_eval:  np.ndarray,
    labels:  np.ndarray,
    n_estimators: int,
) -> dict:
    """Train a candidate model and return its F1 on the labeled evaluation set."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    raw   = model.decision_function(X_eval)
    norms = _normalize(raw)

    # Sweep thresholds to find best F1 for this contamination candidate
    best_f1  = 0.0
    best_thr = 0.5
    for thr in np.arange(0.1, 1.0, 0.05):
        preds = (norms >= thr).astype(int)
        f1    = float(f1_score(labels, preds, zero_division=0))
        if f1 > best_f1:
            best_f1  = f1
            best_thr = float(thr)

    preds = (norms >= best_thr).astype(int)
    return {
        "contamination": contamination,
        "best_thr":      round(best_thr, 2),
        "f1":            round(best_f1, 4),
        "precision":     round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":        round(float(recall_score(labels, preds, zero_division=0)), 4),
    }


def auto_tune_contamination(
    X_train:  np.ndarray,
    X_eval:   np.ndarray,
    labels:   np.ndarray,
    n_estimators: int,
) -> dict:
    """Sweep contamination candidates; return the best-F1 result dict."""
    logger.info(
        "Auto-tuning contamination over {} | eval_set_size={}",
        CONTAMINATION_CANDIDATES, len(labels),
    )
    results = []
    for c in CONTAMINATION_CANDIDATES:
        r = _score_contamination(c, X_train, X_eval, labels, n_estimators)
        results.append(r)
        logger.info(
            "  contamination={:.2f}  →  F1={:.4f}  Prec={:.4f}  Rec={:.4f}  thr={}",
            r["contamination"], r["f1"], r["precision"], r["recall"], r["best_thr"],
        )

    best = max(results, key=lambda x: x["f1"])
    logger.success(
        "Best contamination: {} (F1={:.4f}, threshold={})",
        best["contamination"], best["f1"], best["best_thr"],
    )
    return best


# ── Main training pipeline ─────────────────────────────────────────────────────

def train(
    contamination: Union[float, str] = "auto",
    n_estimators:  int  = DEFAULT_N_ESTIMATORS,
    use_augment:   bool = True,
) -> None:
    """
    Full training pipeline: load → augment → build features → tune → fit → save.

    Parameters
    ----------
    contamination : float or "auto"
        If "auto", sweep CONTAMINATION_CANDIDATES and pick best by F1.
    n_estimators  : number of isolation trees
    use_augment   : if True, expand vendor history with synthetic normal txns
    """
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Load real data ──────────────────────────────────────────────────────────
    logger.info("Loading training data from PostgreSQL…")
    real_df = load_training_data()

    # 2. Expand with synthetic vendor history ────────────────────────────────────
    # Cap synthetic rows at 5% of real data so real procurement patterns dominate.
    # This avoids the model learning synthetic distributions instead of real fraud.
    if use_augment:
        max_synth_rows = max(int(len(real_df) * 0.05), 100)
        avg_tx_per_vendor = len(real_df) / max(real_df["vendor_id"].nunique(), 1)

        if avg_tx_per_vendor >= 15:
            # Dense history: real data already has the behavioral signal we need.
            # Only add a tiny synthetic pad to bootstrap brand-new vendors.
            txns_per_vendor = 0  # skip expansion entirely
        else:
            # Sparse: inflate but still respect the 5% ceiling
            txns_per_vendor = max(1, max_synth_rows // max(real_df["vendor_id"].nunique(), 1))

        if txns_per_vendor > 0:
            synth_df = expand_vendor_history(real_df, txns_per_vendor=txns_per_vendor)
            # Hard cap
            if len(synth_df) > max_synth_rows:
                synth_df = synth_df.sample(max_synth_rows, random_state=42)
            logger.info(
                "avg_tx/vendor={:.1f} → synthetic rows={} ({:.1f}% of real data)",
                avg_tx_per_vendor, len(synth_df),
                100.0 * len(synth_df) / len(real_df),
            )
            train_base_df = pd.concat(
                [real_df[["vendor_id", "amount", "date"]],
                 synth_df[["vendor_id", "amount", "date"]]],
                ignore_index=True,
            )
        else:
            logger.info(
                "avg_tx/vendor={:.1f} → skipping synthetic expansion (dense real history)",
                avg_tx_per_vendor,
            )
            train_base_df = real_df[["vendor_id", "amount", "date"]].copy()
    else:
        logger.warning(
            "Augmentation disabled – training on {} real rows only", len(real_df)
        )
        train_base_df = real_df[["vendor_id", "amount", "date"]].copy()

    train_base_df["date"] = pd.to_datetime(train_base_df["date"], utc=True)

    # 3. Build labeled evaluation set (eval dates are AFTER the training window) ─
    eval_df, eval_labels = create_evaluation_set(
        real_df, n_normal=300, n_per_type=50
    )
    logger.info(
        "Evaluation set: {} rows  ({} normal  /  {} anomaly)",
        len(eval_labels),
        int((eval_labels == 0).sum()),
        int(eval_labels.sum()),
    )

    # 4. Feature matrix on full combined dataset ──────────────────────────────────
    #    Eval rows are appended last; because they are dated AFTER training rows,
    #    build_feature_matrix (which sorts by date internally) will place them at
    #    the end, so slicing X_raw[-len(eval_df):] reliably isolates eval rows.
    full_df = pd.concat(
        [train_base_df,
         eval_df[["vendor_id", "amount", "date"]]],
        ignore_index=True,
    )
    full_df["date"] = pd.to_datetime(full_df["date"], utc=True)
    full_df = full_df.sort_values("date").reset_index(drop=True)

    logger.info("Building feature matrix on {} rows…", len(full_df))
    X_raw_all = build_feature_matrix(full_df)

    # Eval rows occupy the last len(eval_df) positions after the date-sort
    X_raw_eval  = X_raw_all[-len(eval_df):]
    X_raw_train = X_raw_all[: len(X_raw_all) - len(eval_df)]

    # 5. Scale ────────────────────────────────────────────────────────────────────
    # RobustScaler uses median + IQR instead of mean + std.
    # Federal contract amounts have extreme right-tails ($313M max vs $5.76M avg).
    # RobustScaler prevents those extreme real transactions from dominating the
    # scale, making injected anomalies (e.g. 10× vendor avg) visibly isolated.
    logger.info("Fitting RobustScaler on {} training rows…", len(X_raw_train))
    scaler  = RobustScaler(quantile_range=(10.0, 90.0))
    X_train = scaler.fit_transform(X_raw_train)
    X_eval  = scaler.transform(X_raw_eval)

    # 6. Tune / set contamination ─────────────────────────────────────────────────
    if contamination == "auto":
        best_result        = auto_tune_contamination(
            X_train, X_eval, eval_labels, n_estimators
        )
        best_contamination = best_result["contamination"]
        optimal_threshold  = best_result["best_thr"]
    else:
        best_contamination = float(contamination)
        optimal_threshold  = 0.5
        logger.info("Using fixed contamination={}", best_contamination)

    # 7. Final model fit ───────────────────────────────────────────────────────────
    logger.info(
        "Training final IsolationForest | contamination={} n_estimators={}",
        best_contamination, n_estimators,
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=best_contamination,
        max_samples="auto",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # 8. Training score distribution ───────────────────────────────────────────────
    train_scores  = model.decision_function(X_train)
    anomaly_preds = model.predict(X_train)
    anomaly_count = int((anomaly_preds == -1).sum())
    logger.info(
        "Training anomaly rate: {}/{} ({:.1f}%)  score=[{:.4f}, {:.4f}]",
        anomaly_count, len(X_train),
        100.0 * anomaly_count / len(X_train),
        float(train_scores.min()), float(train_scores.max()),
    )

    # 9. Save model + scaler ────────────────────────────────────────────────────────
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.success("Model  → {}", MODEL_PATH)
    logger.success("Scaler → {}", SCALER_PATH)
    # 9b. Second-stage supervised classifier ─────────────────────────────────
    # XGBoost trained on [IF_score | raw_features] using labeled anomalies.
    # XGBoost outperforms RF here because gradient boosting corrects residual
    # errors from IF triage and handles feature interactions better.
    # scale_pos_weight handles class imbalance natively.
    # Uses seed=111 eval set so evaluate.py's seed=777 set is truly held out.
    logger.info("Training second-stage XGBoost classifier…")
    second_stage_f1 = None
    try:
        from sklearn.metrics import f1_score as _f1
        from xgboost import XGBClassifier

        rf_eval_df, rf_eval_labels = create_evaluation_set(
            real_df, n_normal=300, n_per_type=50, seed=111
        )
        rf_full_df = pd.concat(
            [train_base_df, rf_eval_df[["vendor_id", "amount", "date"]]],
            ignore_index=True,
        )
        rf_full_df["date"] = pd.to_datetime(rf_full_df["date"], utc=True)
        rf_full_df = rf_full_df.sort_values("date").reset_index(drop=True)
        X_rf_raw_all  = build_feature_matrix(rf_full_df)
        X_rf_raw_eval = X_rf_raw_all[-len(rf_eval_df):]

        # Stack: [normalised_IF_score | raw_features]
        if_rf_scores = _normalize(
            model.decision_function(scaler.transform(X_rf_raw_eval))
        ).reshape(-1, 1)
        X_2nd = np.hstack([if_rf_scores, X_rf_raw_eval]).astype(np.float32)

        n_neg = int((rf_eval_labels == 0).sum())
        n_pos = int((rf_eval_labels == 1).sum())
        scale_pw = n_neg / max(n_pos, 1)   # class imbalance weight

        # ── Split labels → XGBoost never sees its own training data at eval ──
        from sklearn.model_selection import train_test_split as _tts
        X_2nd_tr, X_2nd_val, y_tr, y_val = _tts(
            X_2nd, rf_eval_labels,
            test_size=0.20, stratify=rf_eval_labels, random_state=42,
        )

        # max_depth=4: shallower trees prevent memorising 500-sample eval set.
        # min_child_weight=5: leaf must contain ≥5 samples → no singleton leaves.
        # reg_alpha/lambda: L1+L2 regularisation penalises large leaf weights.
        clf2 = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=2.0,
            scale_pos_weight=scale_pw,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        clf2.fit(X_2nd_tr, y_tr)
        joblib.dump(clf2, SECOND_STAGE_PATH)

        # Report validation F1 – reflects generalisation, not memorisation
        probs_val = clf2.predict_proba(X_2nd_val)[:, 1]
        preds_val = (probs_val >= 0.5).astype(int)
        second_stage_f1 = round(float(_f1(y_val, preds_val, zero_division=0)), 4)
        logger.success(
            "Second-stage XGBoost → {} | val_F1={}", SECOND_STAGE_PATH, second_stage_f1
        )
    except Exception as exc:
        logger.warning("Second-stage training failed (non-fatal): {}", exc)
    # 10. Run full evaluation to get a reliable optimal_threshold ───────────────────
    logger.info("Running full evaluation pipeline…")
    eval_report: dict = {}
    try:
        from backend.ml.evaluate import print_report, run_evaluation
        eval_report = run_evaluation(n_normal=300, n_per_type=50)
        # Prefer the eval-pipeline threshold (wider F1 sweep → more reliable)
        optimal_threshold = eval_report.get("optimal_threshold", optimal_threshold)
        print_report(eval_report)
    except Exception as exc:
        logger.warning("Evaluation failed (non-fatal): {}", exc)

    # 11. Save metadata ─────────────────────────────────────────────────────────────
    metadata = {
        "trained_at":          datetime.now(timezone.utc).isoformat(),
        "n_real_samples":      len(real_df),
        "n_train_samples":     len(X_train),
        "n_features":          N_FEATURES,
        "feature_names":       FEATURE_NAMES,
        "contamination":       best_contamination,
        "n_estimators":        n_estimators,
        "use_augment":         use_augment,
        "train_anomaly_count": anomaly_count,
        "train_anomaly_rate":  round(float(anomaly_count / len(X_train)), 4),
        "score_min":           round(float(train_scores.min()), 4),
        "score_max":           round(float(train_scores.max()), 4),
        "score_mean":          round(float(train_scores.mean()), 4),
        "optimal_threshold":   optimal_threshold,
        "second_stage_train_f1": second_stage_f1,
        "eval_roc_auc":        eval_report.get("roc_auc"),
        "eval_pr_auc":         eval_report.get("pr_auc"),
        "eval_best_f1":        eval_report.get("best_f1"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.success("Metadata → {}", METADATA_PATH)

    # 12. Build vendor risk profiles + graph risk scores ──────────────────────────
    logger.info("Building vendor risk profiles…")
    try:
        vendor_profiler = VendorRiskProfiler()
        vendor_profiler.build_and_save(real_df)
    except Exception as exc:
        logger.warning("Vendor risk profiling failed (non-fatal): {}", exc)

    logger.info("Building graph fraud risk scores…")
    try:
        graph_detector = GraphFraudDetector()
        graph_detector.build_and_save(real_df)
    except Exception as exc:
        logger.warning("Graph fraud detection failed (non-fatal): {}", exc)

    logger.success(
        "\n✓ Sprint 2 model ready.\n"
        "  Optimal anomaly threshold : {}\n"
        "  ROC-AUC : {}  │  PR-AUC : {}  │  F1 : {}\n"
        "  Start API  : uvicorn backend.api.main:app --reload\n"
        "  Test score : POST http://localhost:8000/api/v1/score",
        optimal_threshold,
        eval_report.get("roc_auc", "?"),
        eval_report.get("pr_auc",  "?"),
        eval_report.get("best_f1", "?"),
    )


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train IsolationForest on procurement transaction data"
    )
    parser.add_argument(
        "--contamination", default="auto",
        help=(
            "Float 0.0-0.5 or 'auto' to sweep and pick best by F1 "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS,
        help=f"Number of isolation trees (default: {DEFAULT_N_ESTIMATORS})",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable synthetic history expansion (not recommended for sparse data)",
    )
    args = parser.parse_args()

    settings = get_settings()
    logger.info(
        "DB: {} | contamination={} | n_estimators={}",
        settings.POSTGRES_DB, args.contamination, args.n_estimators,
    )

    contamination_arg: Union[float, str]
    try:
        contamination_arg = float(args.contamination)
    except ValueError:
        contamination_arg = "auto"

    train(
        contamination=contamination_arg,
        n_estimators=args.n_estimators,
        use_augment=not args.no_augment,
    )
