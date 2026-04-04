"""
backend/ml/evaluate.py
──────────────────────────────────────────────────────────────────────────────
Standalone evaluation pipeline for the trained IsolationForest model.

Outputs
───────
• Console: formatted precision / recall / F1 table at 10 thresholds
• Console: per-anomaly-type breakdown (which fraud patterns are caught)
• models/evaluation_report.json  – full metrics for CI/CD gates
• models/shap_feature_importance.json  – mean |SHAP| per feature

Usage
─────
    python -m backend.ml.evaluate
    python -m backend.ml.evaluate --n-normal 400 --n-per-type 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import text

from backend.database import get_sync_session
from backend.ml.augment import create_evaluation_set, expand_vendor_history
from backend.ml.features import FEATURE_NAMES, build_feature_matrix

MODEL_PATH        = Path("models/isolation_forest.joblib")
SCALER_PATH       = Path("models/scaler.joblib")
METADATA_PATH     = Path("models/model_metadata.json")
SECOND_STAGE_PATH = Path("models/second_stage_clf.joblib")
EVAL_REPORT_PATH  = Path("models/evaluation_report.json")
SHAP_PATH         = Path("models/shap_feature_importance.json")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (shared with trainer)
# ─────────────────────────────────────────────────────────────────────────────

def _load_real_data() -> pd.DataFrame:
    query = text(
        "SELECT t.vendor_id::text AS vendor_id, t.amount::float AS amount, t.date, "
        "       t.awarding_agency "
        "FROM transactions t ORDER BY t.date"
    )
    with get_sync_session() as session:
        rows = session.execute(query).fetchall()
    df = pd.DataFrame(rows, columns=["vendor_id", "amount", "date", "awarding_agency"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_score(raw: np.ndarray) -> np.ndarray:
    """IsolationForest raw → [0,1] where 1 = most anomalous."""
    return np.clip(0.5 - raw, 0.0, 1.0)


def run_evaluation(
    n_normal:   int = 300,
    n_per_type: int = 50,
) -> dict:
    """
    Full evaluation pipeline.

    Steps
    -----
    1. Load trained model + scaler
    2. Load real DB transactions
    3. Build synthetic vendor history (same as training)
    4. Build labeled evaluation set (normals + fraud types)
    5. Compute feature matrix on combined data (preserves lag features)
    6. Score the evaluation rows
    7. Sweep thresholds, compute metrics, find optimal F1 operating point
    8. Compute SHAP feature importance
    9. Save & return report dict
    """
    # ── Load artefacts ─────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run: python -m backend.ml.trainer first."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Loaded model from {}", MODEL_PATH)

    # Load second-stage RF if available
    clf2 = None
    if SECOND_STAGE_PATH.exists():
        clf2 = joblib.load(SECOND_STAGE_PATH)
        logger.info("Loaded second-stage RF from {}", SECOND_STAGE_PATH)
    else:
        logger.warning("No second-stage classifier found — evaluating IF only")

    # ── Build combined dataset in temporal order ────────────────────────────────
    logger.info("Loading real transactions from database…")
    real_df  = _load_real_data()
    synth_df = expand_vendor_history(real_df)

    logger.info("Generating evaluation set (n_normal={}, n_per_type={})…",
                n_normal, n_per_type)
    eval_df, labels = create_evaluation_set(
        real_df, n_normal=n_normal, n_per_type=n_per_type
    )

    # Combine: real + synthetic history + evaluation rows (in date order)
    training_rows_df      = pd.concat([real_df, synth_df], ignore_index=True)
    training_rows_df["is_synthetic"]   = training_rows_df.get("is_synthetic", False)
    training_rows_df["anomaly_type"]   = "NORMAL"

    full_df = pd.concat(
        [training_rows_df[["vendor_id","amount","date","anomaly_type","awarding_agency"]],
         eval_df[["vendor_id","amount","date","anomaly_type"]]],
        ignore_index=True,
    )
    full_df["date"] = pd.to_datetime(full_df["date"], utc=True)
    full_df = full_df.sort_values("date").reset_index(drop=True)

    # Track which rows belong to the evaluation set
    n_eval         = len(eval_df)
    eval_start_idx = len(full_df) - n_eval

    # ── Feature matrix ─────────────────────────────────────────────────────────
    logger.info("Building feature matrix ({} rows)…", len(full_df))
    X_raw = build_feature_matrix(full_df)
    X     = scaler.transform(X_raw)

    # Evaluation portion
    X_eval     = X[eval_start_idx:]
    eval_types = eval_df["anomaly_type"].tolist()

    # ── Score ──────────────────────────────────────────────────────────────────
    logger.info("Scoring {} evaluation rows…", len(X_eval))
    raw_scores    = model.decision_function(X_eval)
    norm_scores   = _normalize_score(raw_scores)

    # Two-stage blended score: 40% IF + 60% RF probability (mirrors scorer.py)
    if clf2 is not None:
        X_raw_eval = X_raw[eval_start_idx:]
        if_col     = norm_scores.reshape(-1, 1)
        X_2nd      = np.hstack([if_col, X_raw_eval]).astype(np.float32)
        rf_probs   = clf2.predict_proba(X_2nd)[:, 1]
        final_scores = np.clip(0.4 * norm_scores + 0.6 * rf_probs, 0.0, 1.0)
        logger.info("Using two-stage blended score (IF+RF)")
    else:
        final_scores = norm_scores

    # ── Per-type summary ───────────────────────────────────────────────────────
    type_summary: dict = {}
    for atype in dict.fromkeys(eval_types):  # preserve insertion order
        mask    = np.array([t == atype for t in eval_types])
        scores  = final_scores[mask]
        is_anom = np.array(labels)[mask].astype(bool)
        if is_anom.any():
            detected_at_05 = int((scores[is_anom] >= 0.5).sum())
            type_summary[atype] = {
                "count":             int(mask.sum()),
                "detected_at_0.5":   detected_at_05,
                "recall_at_0.5":     round(detected_at_05 / int(mask.sum()), 3),
                "mean_score":        round(float(scores[is_anom].mean()), 4),
                "max_score":         round(float(scores[is_anom].max()), 4),
            }
        else:
            type_summary[atype] = {
                "count":         int(mask.sum()),
                "mean_score":    round(float(scores.mean()), 4),
            }

    # ── Threshold sweep ────────────────────────────────────────────────────────
    thresholds = np.arange(0.1, 1.0, 0.05)
    sweep = []
    for thr in thresholds:
        preds = (final_scores >= thr).astype(int)
        if preds.sum() == 0:   # nothing flagged
            p, r, f1 = 1.0, 0.0, 0.0
        else:
            p  = float(precision_score(labels, preds, zero_division=0))
            r  = float(recall_score(labels, preds, zero_division=0))
            f1 = float(f1_score(labels, preds, zero_division=0))
        sweep.append({"threshold": round(float(thr), 2), "precision": round(p,3),
                      "recall": round(r,3), "f1": round(f1,3)})

    # Prefer precision ≥ 0.60 so auditors are not overwhelmed by false alerts.
    # Fall back to plain best-F1 if no threshold meets that floor.
    _PREC_FLOOR = 0.60
    candidates = [s for s in sweep if s["precision"] >= _PREC_FLOOR]
    best = (
        max(candidates, key=lambda x: x["f1"])
        if candidates
        else max(sweep, key=lambda x: x["f1"])
    )
    optimal_threshold = best["threshold"]

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    try:
        roc_auc = float(roc_auc_score(labels, final_scores))
    except Exception:
        roc_auc = 0.0

    # ── PR-AUC ────────────────────────────────────────────────────────────────
    prec_curve, rec_curve, _ = precision_recall_curve(labels, final_scores)
    pr_auc = float(auc(rec_curve, prec_curve))

    # ── SHAP feature importance ────────────────────────────────────────────────
    logger.info("Computing SHAP feature importance…")
    explainer = shap.TreeExplainer(model)
    # Use a sample of eval rows for speed
    n_shap    = min(200, len(X_eval))
    sv        = explainer.shap_values(X_eval[:n_shap])
    mean_abs  = np.abs(sv).mean(axis=0).tolist()
    shap_importance = [
        {"feature": FEATURE_NAMES[i], "mean_abs_shap": round(float(mean_abs[i]), 5)}
        for i in range(len(FEATURE_NAMES))
    ]
    shap_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    # ── Assemble report ───────────────────────────────────────────────────────
    n_anomalies = int(labels.sum())
    report = {
        "n_eval":           len(labels),
        "n_normals":        int((labels == 0).sum()),
        "n_anomalies":      n_anomalies,
        "roc_auc":          round(roc_auc, 4),
        "pr_auc":           round(pr_auc, 4),
        "optimal_threshold": optimal_threshold,
        "best_f1":          best["f1"],
        "best_precision":   best["precision"],
        "best_recall":      best["recall"],
        "threshold_sweep":  sweep,
        "per_type_summary": type_summary,
        "shap_importance":  shap_importance,
    }

    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))
    SHAP_PATH.write_text(json.dumps(shap_importance, indent=2))
    logger.success("Evaluation report → {}", EVAL_REPORT_PATH)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    n_a = report["n_anomalies"]
    n_n = report["n_normals"]
    print("\n" + "═" * 60)
    print("  PROCUREMENT AUDIT ML – EVALUATION REPORT")
    print("═" * 60)
    print(f"  Eval set:   {report['n_eval']} rows  "
          f"({n_n} normal  /  {n_a} anomaly)")
    print(f"  ROC-AUC:    {report['roc_auc']:.4f}   "
          f"(>0.85 = excellent)")
    print(f"  PR-AUC:     {report['pr_auc']:.4f}   "
          f"(>0.70 = good for imbalanced)")
    print(f"\n  ── Optimal operating point ──")
    print(f"  Threshold:  {report['optimal_threshold']}")
    print(f"  F1:         {report['best_f1']:.3f}")
    print(f"  Precision:  {report['best_precision']:.3f}")
    print(f"  Recall:     {report['best_recall']:.3f}")

    print(f"\n  ── Per-Anomaly-Type Recall @ threshold=0.50 ──")
    print(f"  {'Type':<22} {'Count':>6}  {'Detected':>9}  {'Recall':>7}  {'Mean Score':>11}")
    print(f"  {'-'*22} {'-'*6}  {'-'*9}  {'-'*7}  {'-'*11}")
    for atype, stats in report["per_type_summary"].items():
        if atype == "NORMAL":
            continue
        print(f"  {atype:<22} {stats['count']:>6}  "
              f"{stats.get('detected_at_0.5', '--'):>9}  "
              f"{stats.get('recall_at_0.5', '--'):>7}  "
              f"{stats.get('mean_score', '--'):>11}")

    print(f"\n  ── SHAP Feature Importance ──")
    for entry in report["shap_importance"]:
        bar = "█" * int(entry["mean_abs_shap"] * 500)
        print(f"  {entry['feature']:<30} {entry['mean_abs_shap']:.5f}  {bar}")

    print(f"\n  ── Threshold Sweep ──")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for row in report["threshold_sweep"]:
        marker = " ← optimal" if row["threshold"] == report["optimal_threshold"] else ""
        print(f"  {row['threshold']:>10.2f}  {row['precision']:>10.3f}  "
              f"{row['recall']:>8.3f}  {row['f1']:>8.3f}{marker}")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained IsolationForest on a labeled test set"
    )
    parser.add_argument("--n-normal",   type=int, default=200,
                        help="Normal samples to include in eval set (default: 200)")
    parser.add_argument("--n-per-type", type=int, default=20,
                        help="Anomaly samples per fraud type (default: 20)")
    args   = parser.parse_args()
    report = run_evaluation(n_normal=args.n_normal, n_per_type=args.n_per_type)
    print_report(report)
