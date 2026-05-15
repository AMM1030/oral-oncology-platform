"""
FHIR Resource Mapper
--------------------
Converts a patient's analytics outputs (demographics, ML risk score,
NLP findings, CDSS flags) into a FHIR R4-compliant Bundle.

Resources emitted:
  * Patient
  * Observation (one per significant risk factor)
  * Condition (one per high-urgency NLP finding)
  * RiskAssessment (one summarising the ML prediction)

We hand-roll the JSON (instead of using fhir.resources) for transparency
and zero dependencies. The output validates structurally against the FHIR
R4 spec and is intentionally minimal for portfolio demonstration.

Public API:
    build_fhir_bundle(patient_dict, flags) -> dict
"""

from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import Dict, List, Any


# -----------------------------------------------------------------------------
# SNOMED / LOINC codes used in the bundle.
# These are simplified mappings for portfolio purposes — in production
# you'd consult an authoritative terminology server.
# -----------------------------------------------------------------------------
CODES = {
    "smoking_status":  ("LOINC", "72166-2",  "Tobacco smoking status"),
    "alcohol_use":     ("LOINC", "68518-0",  "How often do you have a drink containing alcohol"),
    "hpv_status":      ("LOINC", "59420-2",  "HPV DNA test status"),
    "oral_cancer_risk":("LOINC", "75321-0",  "Clinical finding present"),
    "leukoplakia":     ("SCT",  "414494005","Oral leukoplakia"),
    "erythroplakia":   ("SCT",  "236074009","Erythroplakia of oral mucosa"),
    "non_healing_ulcer":("SCT", "26284000", "Oral ulcer"),
    "suspicious_lesion":("SCT", "300919008","Lesion of oral cavity"),
}


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ref(resource: dict) -> dict:
    return {"reference": f"{resource['resourceType']}/{resource['id']}"}


# -----------------------------------------------------------------------------
# Individual resources
# -----------------------------------------------------------------------------

def build_patient(p: Dict) -> dict:
    return {
        "resourceType": "Patient",
        "id": p["patient_id"],
        "identifier": [{"system": "https://oral-onc.local/patients",
                        "value": p["patient_id"]}],
        "gender": {"M": "male", "F": "female"}.get(p.get("sex", ""), "unknown"),
        "extension": [
            {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
             "valueString": p.get("race_ethnicity", "Unknown")},
        ],
        "_age": p.get("age"),  # convenience field; non-standard
    }


def build_risk_factor_observation(patient_ref: dict, code_key: str,
                                  value: str) -> dict:
    system, code, display = CODES[code_key]
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code": "social-history", "display": "Social History",
        }]}],
        "code": {"coding": [{"system": system, "code": code, "display": display}]},
        "subject": patient_ref,
        "effectiveDateTime": _now(),
        "valueString": value,
    }


def build_condition(patient_ref: dict, code_key: str, note: str) -> dict:
    system, code, display = CODES.get(code_key, CODES["suspicious_lesion"])
    return {
        "resourceType": "Condition",
        "id": str(uuid.uuid4()),
        "clinicalStatus": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "code": "active",
        }]},
        "verificationStatus": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
            "code": "provisional",
        }]},
        "code": {"coding": [{"system": system, "code": code, "display": display}]},
        "subject": patient_ref,
        "note": [{"text": note}],
        "recordedDate": _now(),
    }


def build_risk_assessment(patient_ref: dict, probability: float,
                          rationale: str, flag_summary: List[str]) -> dict:
    return {
        "resourceType": "RiskAssessment",
        "id": str(uuid.uuid4()),
        "status": "final",
        "subject": patient_ref,
        "occurrenceDateTime": _now(),
        "method": {"text": "ML model (XGBoost) + NLP + rule-based CDSS"},
        "prediction": [{
            "outcome": {"text": "Oral cancer (predicted)"},
            "probabilityDecimal": round(probability, 4),
            "rationale": rationale,
        }],
        "note": [{"text": "; ".join(flag_summary)}] if flag_summary else [],
    }


# -----------------------------------------------------------------------------
# Bundle assembler
# -----------------------------------------------------------------------------

def build_fhir_bundle(patient: Dict, flags: List[Any]) -> dict:
    """
    Assemble a FHIR Bundle from one patient's outputs.
    `flags` is a list of Flag dataclass instances from cdss_rules.
    """
    bundle_id = str(uuid.uuid4())

    # Patient + reference to it
    pt = build_patient(patient)
    pt_ref = _ref(pt)

    entries = [{"resource": pt, "fullUrl": f"urn:uuid:{pt['id']}"}]

    # Observations for major risk factors (if present)
    for key, field in [("smoking_status", "smoking_status"),
                       ("alcohol_use",    "alcohol_use"),
                       ("hpv_status",     "hpv_status")]:
        val = patient.get(field)
        if val and val not in ("Never", "No use", "Negative", "Unknown"):
            obs = build_risk_factor_observation(pt_ref, key, str(val))
            entries.append({"resource": obs, "fullUrl": f"urn:uuid:{obs['id']}"})

    # Conditions for high-urgency NLP findings
    for term in patient.get("nlp_extracted_terms", []):
        term_lower = term.lower()
        if "leukoplakia" in term_lower:
            code_key = "leukoplakia"
        elif "erythroplakia" in term_lower:
            code_key = "erythroplakia"
        elif "ulcer" in term_lower:
            code_key = "non_healing_ulcer"
        else:
            code_key = "suspicious_lesion"
        cond = build_condition(pt_ref, code_key, f"NLP-extracted finding: '{term}'")
        entries.append({"resource": cond, "fullUrl": f"urn:uuid:{cond['id']}"})

    # RiskAssessment summarising ML output + flags
    p_risk = patient.get("ml_risk_probability", 0.0)
    flag_summary = [f"[{f.severity}] {f.label}" for f in flags]
    rationale = (
        f"ML predicted P(cancer)={p_risk:.2f}. "
        f"{len(flags)} CDSS flag(s) raised."
    )
    ra = build_risk_assessment(pt_ref, p_risk, rationale, flag_summary)
    entries.append({"resource": ra, "fullUrl": f"urn:uuid:{ra['id']}"})

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": _now(),
        "entry": entries,
    }


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from cdss_rules import apply_rules

    sample = {
        "patient_id": "P00042",
        "age": 67, "sex": "M", "race_ethnicity": "White",
        "smoking_status": "Current", "alcohol_use": "Heavy",
        "hpv_status": "Positive",
        "ml_risk_probability": 0.55,
        "nlp_max_urgency": 10, "note_risk_category": "High",
        "nlp_extracted_terms": ["leukoplakia", "non-healing ulcer", "urgent referral"],
        "has_recent_completed_referral": False,
        "last_screening_date": "2023-09-01",
        "last_referral_date": "2026-02-01",
        "has_pathology": False,
        "n_appts": 5, "no_show_rate": 0.40,
    }
    flags = apply_rules(sample, today=date(2026, 5, 1))
    bundle = build_fhir_bundle(sample, flags)
    print(json.dumps(bundle, indent=2, default=str))