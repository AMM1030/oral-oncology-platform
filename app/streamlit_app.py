"""
Oral Oncology Risk & Care Triage App
-------------------------------------
Streamlit demo combining ML risk prediction + NLP note analysis +
CDSS rule flags + FHIR bundle output.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from cdss_rules import apply_rules           # noqa: E402
from fhir_mapper import build_fhir_bundle    # noqa: E402
from nlp_module import analyze_note          # noqa: E402

DB        = ROOT / "data" / "processed" / "oral_oncology.db"
MODEL     = ROOT / "data" / "processed" / "risk_model.joblib"
SCALER    = ROOT / "data" / "processed" / "risk_scaler.joblib"
FEAT_COLS = ROOT / "data" / "processed" / "feature_columns.json"
METRICS   = ROOT / "data" / "processed" / "model_metrics.json"


@st.cache_resource
def load_model_artifacts():
    return (joblib.load(MODEL),
            joblib.load(SCALER),
            json.loads(FEAT_COLS.read_text()),
            json.loads(METRICS.read_text()))


@st.cache_data
def load_patient_table() -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    return pd.read_sql_query("SELECT * FROM patients ORDER BY patient_id", conn)


@st.cache_data
def load_aggregates(patient_id: str) -> dict:
    conn = sqlite3.connect(DB)
    visits = pd.read_sql_query(
        "SELECT * FROM dental_visits WHERE patient_id = ?", conn, params=[patient_id])
    appts = pd.read_sql_query(
        "SELECT * FROM appointments WHERE patient_id = ?", conn, params=[patient_id])
    refs = pd.read_sql_query(
        "SELECT * FROM referrals WHERE patient_id = ?", conn, params=[patient_id])
    paths = pd.read_sql_query(
        "SELECT * FROM pathology WHERE patient_id = ?", conn, params=[patient_id])
    notes = pd.read_sql_query(
        "SELECT * FROM clinical_notes WHERE patient_id = ? ORDER BY note_date DESC",
        conn, params=[patient_id])
    return {"visits": visits, "appts": appts, "refs": refs,
            "paths": paths, "notes": notes}


def build_feature_row(p: dict, visits: pd.DataFrame, appts: pd.DataFrame,
                      refs: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    row = {
        "age":                       p["age"],
        "socioeconomic_risk_score":  p["socioeconomic_risk_score"],
        "n_visits":                  len(visits),
        "n_lesion_visits":           int(visits["lesion_present"].sum()) if len(visits) else 0,
        "any_lesion_ever":           int(visits["lesion_present"].max()) if len(visits) else 0,
        "screening_rate":            float(visits["screening_completed"].mean()) if len(visits) else 0,
        "n_appts":                   len(appts),
        "no_show_rate":              float(appts["no_show_flag"].mean()) if len(appts) else 0,
        "n_referrals":               len(refs),
    }
    for col in feature_cols:
        if col in row:
            continue
        if "_" in col:
            base, val = col.split("_", 1)
            row[col] = 1 if str(p.get(base)) == val else 0
        else:
            row[col] = 0
    return pd.DataFrame([row])[feature_cols]


st.set_page_config(page_title="Oral Oncology Triage", layout="wide", page_icon="🦷")
st.title("Oral Oncology Risk & Care Triage")
st.caption("ML + NLP + Clinical Decision Support, with FHIR-compatible output.")

model, scaler, feature_cols, metrics = load_model_artifacts()

with st.sidebar:
    st.header("Model info")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.metric("PR-AUC",  f"{metrics['pr_auc']:.3f}")
    st.metric("Recall (cancer)", f"{metrics['recall_cancer']:.2%}")
    st.caption(f"Trained on {metrics['n_train']:,} synthetic patients · {metrics['feature_count']} features")
    st.divider()
    st.markdown("**Author:** Ammulakshmi M.S")
    st.markdown("[GitHub repo](https://github.com/AMM1030/oral-oncology-platform)")

patients = load_patient_table()
top_choice = st.radio("Choose mode:",
                       ["Existing patient", "Manual entry"],
                       horizontal=True)

if top_choice == "Existing patient":
    pid = st.selectbox("Patient ID", patients["patient_id"].tolist(), index=41)
    p = patients.set_index("patient_id").loc[pid].to_dict()
    p["patient_id"] = pid
    aggs = load_aggregates(pid)
    notes_df = aggs["notes"]
    note_text = notes_df["note_text"].iloc[0] if len(notes_df) else ""
    user_note = st.text_area("Most recent clinical note (editable):",
                             value=note_text, height=140)
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 92, 65)
        sex = st.selectbox("Sex", ["F", "M"])
        race = st.selectbox("Race / Ethnicity",
                            ["White", "Black", "Hispanic", "Asian", "Other"])
    with col2:
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol use",
                               ["No use", "Light", "Moderate", "Heavy"])
        hpv = st.selectbox("HPV status", ["Negative", "Positive", "Unknown"])
    with col3:
        insurance = st.selectbox("Insurance",
                                 ["Private", "Medicare", "Medicaid", "Uninsured"])
        ses = st.slider("Socioeconomic risk", 0.0, 1.0, 0.4, 0.05)
    p = {
        "patient_id": "MANUAL",
        "age": age, "sex": sex, "race_ethnicity": race,
        "smoking_status": smoking, "alcohol_use": alcohol, "hpv_status": hpv,
        "insurance_type": insurance, "socioeconomic_risk_score": ses,
    }
    aggs = {"visits": pd.DataFrame(), "appts": pd.DataFrame(),
            "refs": pd.DataFrame(), "paths": pd.DataFrame(),
            "notes": pd.DataFrame()}
    user_note = st.text_area(
        "Clinical note (paste any free text):",
        value="Pt presents with non-healing ulcer on lateral tongue, ~1cm. "
              "Indurated borders. Concerning for malignancy. Smoker. Refer.",
        height=140,
    )

st.divider()

if st.button("Analyze patient", type="primary"):
    feature_row = build_feature_row(p, aggs["visits"], aggs["appts"],
                                    aggs["refs"], feature_cols)
    p_risk = float(model.predict_proba(feature_row.values)[0, 1])

    nlp = analyze_note(user_note)

    today = date(2026, 5, 1)
    last_screening = None
    if len(aggs["visits"]) and aggs["visits"]["screening_completed"].any():
        last_screening = aggs["visits"][aggs["visits"]["screening_completed"] == 1
                                        ]["visit_date"].max()
    last_referral = aggs["refs"]["referral_date"].max() if len(aggs["refs"]) else None
    has_pathology = len(aggs["paths"]) > 0
    has_recent_completed = (
        len(aggs["refs"]) > 0
        and (aggs["refs"]["referral_status"] == "Completed").any()
    )
    patient_full = {
        **p,
        "ml_risk_probability": p_risk,
        "nlp_max_urgency": nlp["urgency_score"],
        "note_risk_category": nlp["note_risk_category"],
        "nlp_extracted_terms": nlp["extracted_terms"],
        "has_recent_completed_referral": has_recent_completed,
        "last_screening_date": last_screening,
        "last_referral_date": last_referral,
        "has_pathology": has_pathology,
        "n_appts": len(aggs["appts"]),
        "no_show_rate": float(aggs["appts"]["no_show_flag"].mean()) if len(aggs["appts"]) else 0,
    }
    flags = apply_rules(patient_full, today=today)
    bundle = build_fhir_bundle(patient_full, flags)

    c1, c2, c3 = st.columns(3)
    c1.metric("ML risk probability", f"{p_risk:.1%}")
    c2.metric("NLP urgency score", nlp["urgency_score"],
              delta=nlp["note_risk_category"])
    c3.metric("CDSS flags raised", len(flags))

    st.subheader("CDSS Flags")
    if not flags:
        st.success("No flags raised. Routine surveillance.")
    else:
        sev_color = {"High": "🔴", "Medium": "🟡", "Low": "🔵", "Info": "⚪"}
        for f in flags:
            st.markdown(f"**{sev_color.get(f.severity, '⚪')} {f.label}**  ·  "
                        f"`{f.code}`  ·  *{f.severity}*")
            st.caption(f.rationale)

    st.subheader("NLP Findings")
    if nlp["extracted_terms"]:
        st.write("Extracted suspicious terms:")
        st.write(nlp["extracted_terms"])
    else:
        st.write("_No suspicious terms detected._")
    if nlp["benign_terms"]:
        st.caption(f"Benign mentions noted (not scored): {nlp['benign_terms']}")

    with st.expander("FHIR Bundle (R4)"):
        st.json(bundle)

    with st.expander("Feature row sent to ML model"):
        st.dataframe(feature_row.T.rename(columns={0: "value"}))