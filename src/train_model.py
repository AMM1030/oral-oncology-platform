"""
Train and persist the oral cancer risk model.

Run:
    python src/train_model.py

Outputs (under data/processed/):
    risk_model.joblib       - trained XGBoost classifier
    risk_scaler.joblib      - fitted StandardScaler (for parity with notebook)
    feature_columns.json    - exact column order the model expects
    model_metrics.json      - test-set metrics for display in the app
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "processed" / "oral_oncology.db"
OUT = ROOT / "data" / "processed"


def load_features() -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    patients = pd.read_sql_query("SELECT * FROM patients", conn)
    visits   = pd.read_sql_query("SELECT * FROM dental_visits", conn)
    appts    = pd.read_sql_query("SELECT * FROM appointments", conn)
    refs     = pd.read_sql_query("SELECT * FROM referrals", conn)

    visit_agg = visits.groupby("patient_id").agg(
        n_visits=("visit_id", "count"),
        n_lesion_visits=("lesion_present", "sum"),
        any_lesion_ever=("lesion_present", "max"),
        screening_rate=("screening_completed", "mean"),
    ).reset_index()
    appt_agg = appts.groupby("patient_id").agg(
        n_appts=("appointment_id", "count"),
        no_show_rate=("no_show_flag", "mean"),
    ).reset_index()
    ref_agg = refs.groupby("patient_id").agg(
        n_referrals=("referral_id", "count"),
    ).reset_index()

    df = (patients
          .merge(visit_agg, on="patient_id", how="left")
          .merge(appt_agg,  on="patient_id", how="left")
          .merge(ref_agg,   on="patient_id", how="left"))
    df[["n_referrals"]] = df[["n_referrals"]].fillna(0)
    return df


def main() -> None:
    df = load_features()
    cat_cols = ["sex", "race_ethnicity", "insurance_type", "smoking_status",
                "alcohol_use", "hpv_status"]
    num_cols = ["age", "socioeconomic_risk_score", "n_visits", "n_lesion_visits",
                "any_lesion_ever", "screening_rate", "n_appts", "no_show_rate",
                "n_referrals"]

    X = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols, drop_first=True)
    y = df["oral_cancer_positive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=neg / pos,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Metrics
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = {
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc":  float(average_precision_score(y_test, y_proba)),
        "recall_cancer":    float(((y_pred == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())),
        "precision_cancer": float(((y_pred == 1) & (y_test == 1)).sum() / max(1, (y_pred == 1).sum())),
        "feature_count": int(X.shape[1]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  OUT / "risk_model.joblib")
    joblib.dump(scaler, OUT / "risk_scaler.joblib")
    (OUT / "feature_columns.json").write_text(json.dumps(list(X.columns)))
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("--- Trained and saved ---")
    print(f"  ROC-AUC : {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC  : {metrics['pr_auc']:.3f}")
    print(f"  Recall  : {metrics['recall_cancer']:.3f}")
    print(f"  Saved to: {OUT.resolve()}")


if __name__ == "__main__":
    main()