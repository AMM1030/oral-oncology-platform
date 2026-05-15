"""
Clinical Decision Support System (CDSS) — Rule Engine
------------------------------------------------------
Deterministic rules that combine:
  * Structured risk factors (age, smoking, alcohol, HPV)
  * NLP findings from clinical notes
  * Workflow signals (referral status, last screening date, biopsy follow-up)

into actionable clinical flags. Rules are intentionally hand-coded (not ML)
because clinical decision support has to be explainable, auditable, and
agreed on with clinicians.

Public API:
    apply_rules(patient_dict) -> list[Flag]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class Flag:
    """A single CDSS flag raised for a patient."""
    code: str              # short machine code, e.g. 'HIGH_PRIORITY_REFERRAL'
    label: str             # human-readable name
    severity: str          # 'High' | 'Medium' | 'Low' | 'Info'
    rationale: str         # short one-line explanation
    fired_by: List[str] = field(default_factory=list)   # which inputs triggered it


# -----------------------------------------------------------------------------
# Helper predicates
# -----------------------------------------------------------------------------

def _is_high_risk_demographics(p: Dict) -> bool:
    """Smoking + alcohol + age threshold combination."""
    return (
        p.get("smoking_status") in ("Current", "Former")
        and p.get("alcohol_use") in ("Moderate", "Heavy")
    ) or p.get("hpv_status") == "Positive" or (p.get("age", 0) >= 65)


def _parse_date(s) -> Optional[date]:
    if not s:
        return None
    if isinstance(s, date):
        return s
    try:
        return datetime.fromisoformat(str(s)).date()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Rules
# -----------------------------------------------------------------------------

def rule_high_priority_referral(p: Dict) -> Optional[Flag]:
    """
    HIGH PRIORITY REFERRAL:
    Patient has a high-urgency clinical note + high-risk demographics +
    no completed referral in the last 30 days.
    """
    high_risk = _is_high_risk_demographics(p)
    nlp_urgent = p.get("nlp_max_urgency", 0) >= 8 or p.get("note_risk_category") == "High"
    no_recent_referral = not p.get("has_recent_completed_referral", False)

    if nlp_urgent and high_risk and no_recent_referral:
        return Flag(
            code="HIGH_PRIORITY_REFERRAL",
            label="High-Priority Oral Oncology Referral",
            severity="High",
            rationale=(
                "Patient has suspicious clinical findings AND established risk factors "
                "AND no recent completed specialist referral."
            ),
            fired_by=["nlp_urgent_finding", "high_risk_demographics", "no_recent_referral"],
        )
    return None


def rule_screening_overdue(p: Dict, today: date = None) -> Optional[Flag]:
    """
    SCREENING OVERDUE:
    High-risk patient with no oral cancer screening completed in past 365 days.
    """
    today = today or date.today()
    last_screening = _parse_date(p.get("last_screening_date"))
    high_risk = _is_high_risk_demographics(p)

    overdue = (last_screening is None) or ((today - last_screening).days > 365)

    if high_risk and overdue:
        return Flag(
            code="SCREENING_OVERDUE",
            label="Oral Cancer Screening Overdue",
            severity="Medium",
            rationale=(
                "High-risk patient has no documented oral cancer screening in the "
                "past 12 months."
            ),
            fired_by=["high_risk_demographics", "screening_gap"],
        )
    return None


def rule_biopsy_follow_up_gap(p: Dict, today: date = None) -> Optional[Flag]:
    """
    BIOPSY FOLLOW-UP GAP:
    A referral was placed more than 30 days ago and no pathology record exists.
    """
    today = today or date.today()
    last_ref = _parse_date(p.get("last_referral_date"))
    has_pathology = p.get("has_pathology", False)

    if last_ref and (today - last_ref).days > 30 and not has_pathology:
        return Flag(
            code="BIOPSY_FOLLOW_UP_GAP",
            label="Biopsy Follow-Up Gap",
            severity="High",
            rationale=(
                "Referral was placed >30 days ago but no pathology record exists. "
                "Patient may have been lost to follow-up."
            ),
            fired_by=["stale_referral", "no_pathology"],
        )
    return None


def rule_ml_high_risk(p: Dict) -> Optional[Flag]:
    """
    ML HIGH RISK:
    Predicted probability from the ML model exceeds threshold.
    """
    p_risk = p.get("ml_risk_probability", 0)
    if p_risk >= 0.40:
        return Flag(
            code="ML_HIGH_RISK",
            label="High Predicted Oral Cancer Risk (ML)",
            severity="High" if p_risk >= 0.60 else "Medium",
            rationale=f"ML model predicted oral cancer probability of {p_risk:.2f}.",
            fired_by=["ml_model"],
        )
    return None


def rule_no_show_pattern(p: Dict) -> Optional[Flag]:
    """
    REPEATED NO-SHOW PATTERN:
    Patient has a no-show rate above 30% across at least 3 appointments.
    """
    n_appts = p.get("n_appts", 0)
    no_show_rate = p.get("no_show_rate", 0)
    if n_appts >= 3 and no_show_rate >= 0.30:
        return Flag(
            code="NO_SHOW_PATTERN",
            label="High No-Show Risk",
            severity="Low",
            rationale=(
                f"Patient missed {no_show_rate*100:.0f}% of {n_appts} appointments. "
                "Outreach may need extra effort."
            ),
            fired_by=["appointment_history"],
        )
    return None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

ALL_RULES = [
    rule_high_priority_referral,
    rule_ml_high_risk,
    rule_biopsy_follow_up_gap,
    rule_screening_overdue,
    rule_no_show_pattern,
]


def apply_rules(patient: Dict, today: date = None) -> List[Flag]:
    """Run every rule against a patient dict and return all flags raised."""
    today = today or date.today()
    flags: List[Flag] = []
    for rule in ALL_RULES:
        try:
            result = rule(patient, today=today) if rule.__code__.co_argcount == 2 else rule(patient)
        except TypeError:
            result = rule(patient)
        if result:
            flags.append(result)
    return flags


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample_patient = {
        "patient_id": "P00042",
        "age": 67,
        "smoking_status": "Current",
        "alcohol_use": "Heavy",
        "hpv_status": "Positive",
        "ml_risk_probability": 0.55,
        "nlp_max_urgency": 10,
        "note_risk_category": "High",
        "has_recent_completed_referral": False,
        "last_screening_date": "2023-09-01",
        "last_referral_date": "2026-02-01",
        "has_pathology": False,
        "n_appts": 5,
        "no_show_rate": 0.40,
    }
    today = date(2026, 5, 1)
    flags = apply_rules(sample_patient, today=today)
    print(f"Patient {sample_patient['patient_id']} → {len(flags)} flags raised:\n")
    for f in flags:
        print(f"  [{f.severity}] {f.label} ({f.code})")
        print(f"     {f.rationale}\n")