-- =============================================================================
-- Oral Oncology Risk & Care Intelligence Platform — Database Schema
-- =============================================================================
-- Design notes:
--   * patient_id is the central foreign key — every other table joins back here.
--   * Indexes are added on join columns and date columns for query speed.
--   * Written in portable SQL (works on SQLite, PostgreSQL, MySQL with minor tweaks).
-- =============================================================================

-- Drop existing tables (so this script is rerunnable)
DROP TABLE IF EXISTS outcomes;
DROP TABLE IF EXISTS pathology;
DROP TABLE IF EXISTS referrals;
DROP TABLE IF EXISTS appointments;
DROP TABLE IF EXISTS clinical_notes;
DROP TABLE IF EXISTS dental_visits;
DROP TABLE IF EXISTS patients;

-- -----------------------------------------------------------------------------
-- patients : core demographics + risk factors
-- -----------------------------------------------------------------------------
CREATE TABLE patients (
    patient_id                 TEXT PRIMARY KEY,
    age                        INTEGER NOT NULL,
    sex                        TEXT    NOT NULL,
    race_ethnicity             TEXT,
    insurance_type             TEXT,
    zip_code                   TEXT,
    smoking_status             TEXT,
    alcohol_use                TEXT,
    hpv_status                 TEXT,
    socioeconomic_risk_score   REAL,
    oral_cancer_positive       INTEGER NOT NULL  -- 0/1 ground-truth label
);

CREATE INDEX idx_patients_smoking   ON patients(smoking_status);
CREATE INDEX idx_patients_insurance ON patients(insurance_type);
CREATE INDEX idx_patients_zip       ON patients(zip_code);

-- -----------------------------------------------------------------------------
-- dental_visits : every dental encounter
-- -----------------------------------------------------------------------------
CREATE TABLE dental_visits (
    visit_id              TEXT PRIMARY KEY,
    patient_id            TEXT NOT NULL,
    visit_date            DATE NOT NULL,
    provider_id           TEXT,
    reason_for_visit      TEXT,
    lesion_present        INTEGER,
    lesion_location       TEXT,
    symptoms_reported     TEXT,
    screening_completed   INTEGER,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_visits_patient ON dental_visits(patient_id);
CREATE INDEX idx_visits_date    ON dental_visits(visit_date);
CREATE INDEX idx_visits_lesion  ON dental_visits(lesion_present);

-- -----------------------------------------------------------------------------
-- clinical_notes : free-text dental notes (one per visit)
-- -----------------------------------------------------------------------------
CREATE TABLE clinical_notes (
    note_id      TEXT PRIMARY KEY,
    patient_id   TEXT NOT NULL,
    visit_id     TEXT,
    note_date    DATE,
    note_text    TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (visit_id)   REFERENCES dental_visits(visit_id)
);

CREATE INDEX idx_notes_patient ON clinical_notes(patient_id);
CREATE INDEX idx_notes_visit   ON clinical_notes(visit_id);

-- -----------------------------------------------------------------------------
-- referrals : referrals to specialists
-- -----------------------------------------------------------------------------
CREATE TABLE referrals (
    referral_id                TEXT PRIMARY KEY,
    patient_id                 TEXT NOT NULL,
    referral_date              DATE,
    referral_type              TEXT,
    referral_status            TEXT,    -- Completed / Pending / No-show / Cancelled
    days_to_specialist_visit   INTEGER, -- NULL if not completed
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_referrals_patient ON referrals(patient_id);
CREATE INDEX idx_referrals_status  ON referrals(referral_status);
CREATE INDEX idx_referrals_date    ON referrals(referral_date);

-- -----------------------------------------------------------------------------
-- pathology : biopsy results
-- -----------------------------------------------------------------------------
CREATE TABLE pathology (
    pathology_id        TEXT PRIMARY KEY,
    patient_id          TEXT NOT NULL,
    biopsy_date         DATE,
    diagnosis_result    TEXT,
    tumor_site          TEXT,
    tumor_stage         TEXT,
    dysplasia_grade     TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_pathology_patient ON pathology(patient_id);
CREATE INDEX idx_pathology_dx      ON pathology(diagnosis_result);

-- -----------------------------------------------------------------------------
-- appointments : scheduled appointments + no-show tracking
-- -----------------------------------------------------------------------------
CREATE TABLE appointments (
    appointment_id       TEXT PRIMARY KEY,
    patient_id           TEXT NOT NULL,
    appointment_date     DATE,
    appointment_type     TEXT,
    appointment_status   TEXT,
    no_show_flag         INTEGER,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_appts_patient ON appointments(patient_id);
CREATE INDEX idx_appts_date    ON appointments(appointment_date);

-- -----------------------------------------------------------------------------
-- outcomes : final clinical outcomes per patient
-- -----------------------------------------------------------------------------
CREATE TABLE outcomes (
    patient_id            TEXT PRIMARY KEY,
    diagnosis_status      TEXT,
    diagnosis_stage       TEXT,
    treatment_started     INTEGER,
    treatment_start_date  DATE,
    recurrence_status     INTEGER,
    survival_status       TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_outcomes_status ON outcomes(diagnosis_status);