"""
Synthetic Dental Oncology Data Generator
-----------------------------------------
Generates a clinically plausible synthetic dataset for the Oral Oncology
Risk & Care Intelligence Platform.

Run:
    python src/generate_data.py --n 5000 --seed 42 --out data/raw

Output: 7 CSVs in data/raw/
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# =============================================================================
# SECTION 1: Reference distributions
# Anchored loosely to US population stats so the synthetic data is plausible.
# =============================================================================

SEX_DIST = {"F": 0.51, "M": 0.49}
RACE_DIST = {
    "White": 0.60, "Black": 0.13, "Hispanic": 0.18,
    "Asian": 0.06, "Other": 0.03,
}
INSURANCE_DIST = {
    "Private": 0.50, "Medicare": 0.25, "Medicaid": 0.18, "Uninsured": 0.07,
}

# Pittsburgh-area / Allegheny County ZIPs tagged with a baseline SES risk
# (0=lowest socioeconomic risk, 1=highest). Used to influence insurance,
# screening compliance, and referral delays.
ZIP_RISK = {
    "15213": 0.30, "15217": 0.25, "15232": 0.20, "15238": 0.15,
    "15206": 0.45, "15208": 0.55, "15210": 0.60, "15212": 0.55,
    "15219": 0.70, "15221": 0.50, "15224": 0.45, "15235": 0.50,
    "15236": 0.40, "15237": 0.20, "15243": 0.20, "15146": 0.40,
}

SMOKING_BASE = {"Never": 0.55, "Former": 0.25, "Current": 0.20}
ALCOHOL_BASE = {"No use": 0.40, "Light": 0.35, "Moderate": 0.18, "Heavy": 0.07}
HPV_BASE = {"Negative": 0.85, "Positive": 0.10, "Unknown": 0.05}

LESION_LOCATIONS = [
    "lateral tongue", "ventral tongue", "floor of mouth",
    "buccal mucosa", "soft palate", "hard palate",
    "lower lip", "retromolar trigone", "gingiva", "tonsillar pillar",
]

VISIT_REASONS = [
    "routine cleaning", "routine exam", "patient-reported lesion",
    "follow-up", "pain evaluation", "denture adjustment",
    "screening exam", "second opinion",
]

REFERRAL_TYPES = [
    "Oral Surgery", "Oral Medicine", "ENT / Head & Neck Surgery",
    "Medical Oncology", "Radiation Oncology",
]

# =============================================================================
# SECTION 2: Helper functions
# =============================================================================

def weighted_choice(d: dict, rng: np.random.Generator):
    """Pick a key from a dict whose values are weights."""
    keys = list(d.keys())
    probs = np.array(list(d.values()))
    probs = probs / probs.sum()
    return rng.choice(keys, p=probs)


def sample_age(rng: np.random.Generator) -> int:
    """Beta-shaped distribution skewed older (oral cancer risk rises with age)."""
    a = rng.beta(5, 2) * 74 + 18  # range: 18 to 92
    return int(round(a))


def adjusted_smoking(age: int, ses_risk: float, rng: np.random.Generator) -> str:
    """Lower SES + older age -> higher smoking rates."""
    p_current = 0.10 + 0.20 * ses_risk + 0.001 * max(0, age - 40)
    p_former = 0.20 + 0.15 * ses_risk + 0.003 * max(0, age - 40)
    p_never = max(0.05, 1 - p_current - p_former)
    total = p_current + p_former + p_never
    p = np.array([p_never, p_former, p_current]) / total
    return rng.choice(["Never", "Former", "Current"], p=p)


def adjusted_alcohol(smoking: str, ses_risk: float, rng: np.random.Generator) -> str:
    """Smokers drink more; drinking also rises with SES risk."""
    boost = 0.15 if smoking == "Current" else (0.05 if smoking == "Former" else 0)
    p_heavy = 0.05 + 0.05 * ses_risk + boost * 0.6
    p_mod = 0.15 + 0.05 * ses_risk + boost * 0.4
    p_light = 0.35
    p_none = max(0.05, 1 - p_heavy - p_mod - p_light)
    p = np.array([p_none, p_light, p_mod, p_heavy])
    p = p / p.sum()
    return rng.choice(["No use", "Light", "Moderate", "Heavy"], p=p)


def cancer_probability(row: dict) -> float:
    """
    THE HEART OF THE GENERATOR.
    Logistic combination of real risk factors -> P(oral cancer).
    The coefficients are loosely calibrated so cancer prevalence in the
    overall sample lands around 5-8% (enriched vs general population since
    this is meant to be a screening cohort).
    """
    log_odds = -5.5  # baseline ~ 0.4%
    log_odds += 0.04 * max(0, row["age"] - 40)
    log_odds += 0.5 if row["sex"] == "M" else 0
    log_odds += {"Never": 0, "Former": 0.7, "Current": 1.6}[row["smoking_status"]]
    log_odds += {"No use": 0, "Light": 0.1, "Moderate": 0.5, "Heavy": 1.1}[row["alcohol_use"]]
    log_odds += 1.2 if row["hpv_status"] == "Positive" else 0
    log_odds += 0.6 * row["socioeconomic_risk_score"]
    return 1.0 / (1.0 + np.exp(-log_odds))


# =============================================================================
# SECTION 3: Note templates
# Three flavors of dental notes. Suspicious ones use placeholders that
# get filled in with actual lesion locations and durations so NLP isn't
# trivial (multiple phrasings of "non-healing ulcer").
# =============================================================================

ROUTINE_NOTES = [
    "Pt presents for routine prophylaxis. No complaints. Soft tissue exam unremarkable. OHI reinforced. RTC 6 months.",
    "Recall visit. Patient reports no oral pain or lesions. Intraoral exam WNL. Calculus removed UR/LL quadrants. Polished. Floss demo.",
    "Adult prophy completed. Mild gingival inflammation #18-19 area, otherwise tissues healthy. Pt counseled on smoking cessation.",
    "Routine 6mo recall. Pt denies any concerns. Oral cancer screening performed - negative. Plaque score 12%.",
    "Pt here for cleaning. No symptoms reported. Tongue, FOM, palate, buccal mucosa visualized and palpated - no abnormalities noted.",
    "Recall exam. Patient asymptomatic. Soft tissue WNL. Recommended fluoride varnish given caries risk.",
]

SUSPICIOUS_NOTES = [
    "Pt reports {duration} hx of non-healing ulcer on {loc}. Approx 8mm, indurated borders, no obvious cause. Will refer to oral medicine for biopsy consideration. Pt is a current smoker.",
    "Noted persistent white patch (leukoplakia) on {loc} during exam today. Pt unaware of lesion. Cannot be wiped off. Referring for further eval.",
    "Patient c/o {duration} of persistent ulcer with bleeding when brushing. Examined {loc} - lesion ~1cm, irregular margins, no resolution despite removal of suspected trauma source. Referral placed.",
    "Erythroplakia noted on {loc} - bright red, velvety, well-demarcated. No pain reported. Given pt's smoking and alcohol hx, urgent referral to oral surgeon.",
    "Pt presents with lump in mouth that has been present approximately {duration}. Firm, non-tender mass on {loc}. Concerning for malignancy - referring out.",
    "Patient reports difficulty swallowing and unexplained oral pain x {duration}. Exam reveals indurated lesion on {loc} with rolled borders. High suspicion - urgent referral.",
    "Lesion present on {loc} for more than two weeks per pt report. Mixed red and white appearance. Does not appear traumatic. Will biopsy or refer.",
    "Noted neck swelling and ulcerated area on {loc}. Pt is a former heavy smoker. Concerning findings - same-day phone consult with oral surgery placed.",
    "Recall pt. Today noted new lesion {loc} not present at prior visit 6mo ago. Indurated, ~5mm. Photo taken. Referring for evaluation.",
    "Patient self-referred for {duration}-old sore that hasn't healed. Appears as ulcerated lesion on {loc}, ~1.5cm. Bleeds easily on probing. Urgent referral made.",
]

EQUIVOCAL_NOTES = [
    "Small white area noted on {loc}, likely frictional keratosis from {loc} occlusal trauma. Will monitor at next recall. Pt advised to return sooner if changes.",
    "Aphthous ulcer on {loc} per pt - reports recurs intermittently. Resolved at exam today. No referral needed.",
    "Trauma-related lesion {loc} from cheek bite. Will reassess at follow-up in 2 weeks.",
    "Mild erythema {loc}, possibly from new medication. Monitor.",
    "Geographic tongue noted - benign finding, pt reassured.",
]

DURATIONS = ["3 weeks", "1 month", "6 weeks", "2 months", "3 months", "over 4 weeks"]

# =============================================================================
# SECTION 4: Generator functions
# =============================================================================

def gen_patients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        zip_code = weighted_choice({z: 1 for z in ZIP_RISK}, rng)
        ses = ZIP_RISK[zip_code] + rng.normal(0, 0.05)
        ses = float(np.clip(ses, 0, 1))
        age = sample_age(rng)
        sex = weighted_choice(SEX_DIST, rng)
        smoking = adjusted_smoking(age, ses, rng)
        alcohol = adjusted_alcohol(smoking, ses, rng)

        # Insurance: Medicare for 65+, otherwise SES-influenced.
        if age >= 65 and rng.random() < 0.75:
            insurance = "Medicare"
        else:
            ins_w = {"Private": 0.55 - 0.4 * ses, "Medicaid": 0.15 + 0.35 * ses,
                     "Uninsured": 0.05 + 0.15 * ses, "Medicare": 0.05}
            insurance = weighted_choice(ins_w, rng)

        rows.append({
            "patient_id": f"P{i:05d}",
            "age": age,
            "sex": sex,
            "race_ethnicity": weighted_choice(RACE_DIST, rng),
            "insurance_type": insurance,
            "zip_code": zip_code,
            "smoking_status": smoking,
            "alcohol_use": alcohol,
            "hpv_status": weighted_choice(HPV_BASE, rng),
            "socioeconomic_risk_score": round(ses, 3),
        })

    df = pd.DataFrame(rows)
    # Generate the cancer label using the logistic risk model.
    probs = df.apply(lambda r: cancer_probability(r), axis=1)
    df["_cancer_prob"] = probs
    df["oral_cancer_positive"] = (rng.random(len(df)) < probs).astype(int)
    return df


def gen_visits_and_notes(patients: pd.DataFrame, rng: np.random.Generator
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    visit_rows, note_rows = [], []
    today = date(2026, 5, 1)
    visit_id = 1
    note_id = 1
    for _, p in patients.iterrows():
        # Each patient has 1-6 visits over the past ~3 years.
        n_visits = rng.integers(1, 7)
        for v in range(int(n_visits)):
            days_back = int(rng.integers(15, 1100))
            visit_date = today - timedelta(days=days_back)
            reason = rng.choice(VISIT_REASONS)
            screening_done = rng.random() < (0.85 - 0.4 * p["socioeconomic_risk_score"])

            base_lesion_p = 0.05 + 0.6 * p["_cancer_prob"]
            if reason == "patient-reported lesion":
                base_lesion_p += 0.6
            lesion_present = int(rng.random() < min(base_lesion_p, 0.95))
            lesion_loc = rng.choice(LESION_LOCATIONS) if lesion_present else None
            symptoms = ""
            if lesion_present and rng.random() < 0.7:
                symptoms = rng.choice([
                    "non-healing ulcer", "white patch", "lump", "pain on swallowing",
                    "bleeding lesion", "burning sensation", "numbness",
                ])

            if lesion_present and p["_cancer_prob"] > 0.05 and rng.random() < 0.7:
                template = rng.choice(SUSPICIOUS_NOTES)
                note_text = template.format(loc=lesion_loc, duration=rng.choice(DURATIONS))
            elif lesion_present and rng.random() < 0.5:
                template = rng.choice(EQUIVOCAL_NOTES)
                note_text = template.format(loc=lesion_loc or "buccal mucosa")
            else:
                note_text = rng.choice(ROUTINE_NOTES)

            visit_rows.append({
                "visit_id": f"V{visit_id:06d}",
                "patient_id": p["patient_id"],
                "visit_date": visit_date.isoformat(),
                "provider_id": f"PRV{int(rng.integers(1, 21)):03d}",
                "reason_for_visit": reason,
                "lesion_present": lesion_present,
                "lesion_location": lesion_loc,
                "symptoms_reported": symptoms,
                "screening_completed": int(screening_done),
            })
            note_rows.append({
                "note_id": f"N{note_id:06d}",
                "patient_id": p["patient_id"],
                "visit_id": f"V{visit_id:06d}",
                "note_date": visit_date.isoformat(),
                "note_text": note_text,
            })
            visit_id += 1
            note_id += 1

    return pd.DataFrame(visit_rows), pd.DataFrame(note_rows)


def gen_referrals_pathology_outcomes(
    patients: pd.DataFrame, visits: pd.DataFrame, rng: np.random.Generator
):
    referrals, pathology, outcomes, appointments = [], [], [], []
    ref_id = path_id = appt_id = 1
    visits_with_lesion = visits[visits["lesion_present"] == 1]

    for _, p in patients.iterrows():
        pid = p["patient_id"]
        ses = p["socioeconomic_risk_score"]
        cancer_pos = p["oral_cancer_positive"] == 1

        pt_lesion_visits = visits_with_lesion[visits_with_lesion["patient_id"] == pid]

        # ----- Referrals: triggered by lesion visits in higher-risk patients
        if not pt_lesion_visits.empty and (cancer_pos or rng.random() < 0.4):
            ref_visit = pt_lesion_visits.sample(1, random_state=int(rng.integers(0, 1e6))).iloc[0]
            ref_date = pd.to_datetime(ref_visit["visit_date"]).date()
            base_delay = max(1, int(rng.normal(14 + 25 * ses, 10)))
            ref_status = rng.choice(
                ["Completed", "Pending", "No-show", "Cancelled"],
                p=[0.65 - 0.3 * ses, 0.20 + 0.15 * ses,
                   0.10 + 0.10 * ses, 0.05 + 0.05 * ses],
            )
            referrals.append({
                "referral_id": f"R{ref_id:05d}",
                "patient_id": pid,
                "referral_date": ref_date.isoformat(),
                "referral_type": rng.choice(REFERRAL_TYPES, p=[0.40, 0.20, 0.20, 0.10, 0.10]),
                "referral_status": ref_status,
                "days_to_specialist_visit": base_delay if ref_status == "Completed" else None,
            })
            ref_id += 1

            # Pathology only if referral completed AND lesion was suspicious
            if ref_status == "Completed" and (cancer_pos or rng.random() < 0.3):
                biopsy_date = ref_date + timedelta(days=base_delay + int(rng.integers(1, 14)))
                if cancer_pos:
                    dx = rng.choice(
                        ["Squamous Cell Carcinoma", "Verrucous Carcinoma",
                         "Carcinoma in situ", "Severe Dysplasia"],
                        p=[0.70, 0.05, 0.10, 0.15],
                    )
                    stage = rng.choice(["I", "II", "III", "IV"],
                                       p=[0.25 - 0.1 * ses, 0.30,
                                          0.25 + 0.05 * ses, 0.20 + 0.05 * ses])
                    dysplasia = "N/A"
                else:
                    dx = rng.choice(
                        ["Benign hyperkeratosis", "Mild dysplasia",
                         "Lichen planus", "Inflammatory only"],
                        p=[0.40, 0.30, 0.15, 0.15],
                    )
                    stage = None
                    dysplasia = rng.choice(["None", "Mild", "Moderate"], p=[0.5, 0.4, 0.1])
                pathology.append({
                    "pathology_id": f"PA{path_id:05d}",
                    "patient_id": pid,
                    "biopsy_date": biopsy_date.isoformat(),
                    "diagnosis_result": dx,
                    "tumor_site": ref_visit["lesion_location"],
                    "tumor_stage": stage,
                    "dysplasia_grade": dysplasia,
                })
                path_id += 1

        # ----- Appointments (scheduling/no-show signal)
        n_appts = int(rng.integers(2, 8))
        for _ in range(n_appts):
            ad = date(2026, 5, 1) - timedelta(days=int(rng.integers(15, 900)))
            no_show_p = 0.05 + 0.20 * ses
            status = rng.choice(["Completed", "No-show", "Cancelled"],
                                p=[1 - no_show_p - 0.05, no_show_p, 0.05])
            appointments.append({
                "appointment_id": f"A{appt_id:06d}",
                "patient_id": pid,
                "appointment_date": ad.isoformat(),
                "appointment_type": rng.choice(["Cleaning", "Exam", "Treatment",
                                                "Consultation", "Follow-up"]),
                "appointment_status": status,
                "no_show_flag": int(status == "No-show"),
            })
            appt_id += 1

        # ----- Outcomes
        if cancer_pos:
            tx_started = rng.random() < (0.85 - 0.3 * ses)
            outcomes.append({
                "patient_id": pid,
                "diagnosis_status": "Confirmed",
                "diagnosis_stage": rng.choice(["I", "II", "III", "IV"],
                                              p=[0.20, 0.30, 0.25, 0.25]),
                "treatment_started": int(tx_started),
                "treatment_start_date": (date(2026, 5, 1) - timedelta(
                    days=int(rng.integers(30, 800)))).isoformat() if tx_started else None,
                "recurrence_status": int(tx_started and rng.random() < 0.18),
                "survival_status": rng.choice(["Alive", "Deceased"], p=[0.78, 0.22]),
            })
        else:
            outcomes.append({
                "patient_id": pid,
                "diagnosis_status": "Negative",
                "diagnosis_stage": None,
                "treatment_started": 0,
                "treatment_start_date": None,
                "recurrence_status": 0,
                "survival_status": "Alive",
            })

    return (pd.DataFrame(referrals), pd.DataFrame(pathology),
            pd.DataFrame(appointments), pd.DataFrame(outcomes))


# =============================================================================
# SECTION 5: Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of patients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} patients...")
    patients = gen_patients(args.n, rng)

    print("Generating visits and clinical notes...")
    visits, notes = gen_visits_and_notes(patients, rng)

    print("Generating referrals, pathology, appointments, outcomes...")
    referrals, pathology, appointments, outcomes = gen_referrals_pathology_outcomes(
        patients, visits, rng
    )

    # Drop the leakage column before saving.
    patients_save = patients.drop(columns=["_cancer_prob"])

    patients_save.to_csv(out / "patients.csv", index=False)
    visits.to_csv(out / "dental_visits.csv", index=False)
    notes.to_csv(out / "clinical_notes.csv", index=False)
    referrals.to_csv(out / "referrals.csv", index=False)
    pathology.to_csv(out / "pathology.csv", index=False)
    appointments.to_csv(out / "appointments.csv", index=False)
    outcomes.to_csv(out / "outcomes.csv", index=False)

    print("\n--- Generation summary ---")
    print(f"  patients         : {len(patients_save):>6,}")
    print(f"  dental_visits    : {len(visits):>6,}")
    print(f"  clinical_notes   : {len(notes):>6,}")
    print(f"  referrals        : {len(referrals):>6,}")
    print(f"  pathology        : {len(pathology):>6,}")
    print(f"  appointments     : {len(appointments):>6,}")
    print(f"  outcomes         : {len(outcomes):>6,}")
    print(f"  cancer prevalence: {patients_save['oral_cancer_positive'].mean()*100:.1f}%")
    print(f"\nFiles written to: {out.resolve()}")


if __name__ == "__main__":
    main()