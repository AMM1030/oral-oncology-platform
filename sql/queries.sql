-- =============================================================================
-- Oral Oncology Risk & Care Intelligence Platform — Analytics Queries
-- =============================================================================
-- Each query maps to one clinical or operational question. Run individually
-- against data/processed/oral_oncology.db using SQLite Viewer or sqlite3 CLI.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Q1. POPULATION OVERVIEW KPIs
-- One-row dashboard summary used by the executive view.
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*)                                                    AS total_patients,
    SUM(oral_cancer_positive)                                   AS cancer_positive_count,
    ROUND(100.0 * AVG(oral_cancer_positive), 2)                 AS cancer_prevalence_pct,
    SUM(CASE WHEN smoking_status = 'Current' THEN 1 ELSE 0 END) AS current_smokers,
    SUM(CASE WHEN alcohol_use   = 'Heavy'    THEN 1 ELSE 0 END) AS heavy_drinkers,
    SUM(CASE WHEN hpv_status    = 'Positive' THEN 1 ELSE 0 END) AS hpv_positive
FROM patients;


-- -----------------------------------------------------------------------------
-- Q2. CANCER PREVALENCE BY RISK FACTOR
-- Stratifies cancer rate by smoking, alcohol, HPV. Drives the population
-- risk dashboard. Demonstrates GROUP BY + conditional aggregation.
-- -----------------------------------------------------------------------------
SELECT
    'Smoking' AS factor,
    smoking_status AS group_value,
    COUNT(*) AS n_patients,
    SUM(oral_cancer_positive) AS n_cancer,
    ROUND(100.0 * AVG(oral_cancer_positive), 2) AS cancer_rate_pct
FROM patients
GROUP BY smoking_status

UNION ALL

SELECT
    'Alcohol' AS factor,
    alcohol_use AS group_value,
    COUNT(*),
    SUM(oral_cancer_positive),
    ROUND(100.0 * AVG(oral_cancer_positive), 2)
FROM patients
GROUP BY alcohol_use

UNION ALL

SELECT
    'HPV' AS factor,
    hpv_status AS group_value,
    COUNT(*),
    SUM(oral_cancer_positive),
    ROUND(100.0 * AVG(oral_cancer_positive), 2)
FROM patients
GROUP BY hpv_status

ORDER BY factor, cancer_rate_pct DESC;


-- -----------------------------------------------------------------------------
-- Q3. REFERRAL FUNNEL by STATUS and TYPE
-- How many referrals are placed vs completed vs lost? Drives the
-- clinical workflow dashboard.
-- -----------------------------------------------------------------------------
SELECT
    referral_type,
    referral_status,
    COUNT(*) AS n_referrals,
    ROUND(AVG(days_to_specialist_visit), 1) AS avg_days_to_visit
FROM referrals
GROUP BY referral_type, referral_status
ORDER BY referral_type, n_referrals DESC;


-- -----------------------------------------------------------------------------
-- Q4. REFERRAL DELAY BY INSURANCE TYPE (equity / disparities lens)
-- Joins patients -> referrals to expose differential access by insurance.
-- This is the type of finding clinical leadership cares about.
-- -----------------------------------------------------------------------------
SELECT
    p.insurance_type,
    COUNT(r.referral_id) AS n_completed_referrals,
    ROUND(AVG(r.days_to_specialist_visit), 1) AS avg_days_to_specialist,
    MIN(r.days_to_specialist_visit) AS min_days,
    MAX(r.days_to_specialist_visit) AS max_days
FROM patients p
JOIN referrals r ON p.patient_id = r.patient_id
WHERE r.referral_status = 'Completed'
GROUP BY p.insurance_type
ORDER BY avg_days_to_specialist DESC;


-- -----------------------------------------------------------------------------
-- Q5. SCREENING GAPS — high-risk patients with no screening in last 12 months
-- The "outreach list" — who needs to be called for an appointment.
-- Uses LEFT JOIN to find patients whose most-recent screening is too old
-- (or doesn't exist).
-- -----------------------------------------------------------------------------
WITH last_screening AS (
    SELECT
        patient_id,
        MAX(visit_date) AS last_screening_date
    FROM dental_visits
    WHERE screening_completed = 1
    GROUP BY patient_id
),
high_risk AS (
    SELECT patient_id, age, smoking_status, alcohol_use, hpv_status
    FROM patients
    WHERE smoking_status IN ('Current', 'Former')
       OR alcohol_use   IN ('Moderate', 'Heavy')
       OR hpv_status    = 'Positive'
       OR age >= 60
)
SELECT
    hr.patient_id,
    hr.age,
    hr.smoking_status,
    hr.alcohol_use,
    hr.hpv_status,
    ls.last_screening_date,
    CASE
        WHEN ls.last_screening_date IS NULL THEN 'Never screened'
        ELSE 'Overdue'
    END AS screening_status
FROM high_risk hr
LEFT JOIN last_screening ls ON hr.patient_id = ls.patient_id
WHERE ls.last_screening_date IS NULL
   OR ls.last_screening_date < DATE('2026-05-01', '-365 days')
ORDER BY ls.last_screening_date NULLS LAST;


-- -----------------------------------------------------------------------------
-- Q6. CANCER STAGE AT DIAGNOSIS by INSURANCE TYPE
-- Late-stage diagnosis is a key quality metric. Combined with insurance
-- type it surfaces equity-of-care issues.
-- -----------------------------------------------------------------------------
SELECT
    p.insurance_type,
    o.diagnosis_stage,
    COUNT(*) AS n_patients,
    ROUND(100.0 * COUNT(*) * 1.0 /
        SUM(COUNT(*)) OVER (PARTITION BY p.insurance_type), 1) AS pct_within_insurance
FROM patients p
JOIN outcomes o ON p.patient_id = o.patient_id
WHERE o.diagnosis_status = 'Confirmed'
GROUP BY p.insurance_type, o.diagnosis_stage
ORDER BY p.insurance_type, o.diagnosis_stage;


-- -----------------------------------------------------------------------------
-- Q7. PROVIDER PERFORMANCE
-- Visits per provider, screening rate per provider, lesion-detection rate.
-- For provider-level QI dashboards.
-- -----------------------------------------------------------------------------
SELECT
    provider_id,
    COUNT(*) AS n_visits,
    ROUND(100.0 * AVG(screening_completed), 1) AS screening_rate_pct,
    ROUND(100.0 * AVG(lesion_present), 2) AS lesion_detection_rate_pct
FROM dental_visits
GROUP BY provider_id
ORDER BY n_visits DESC;


-- -----------------------------------------------------------------------------
-- Q8. PRIORITY OUTREACH LIST — Top 50 highest-risk patients to contact
-- Composite risk score combining multiple risk factors and care gaps.
-- This is the "actionable" output the project ultimately exists to produce.
-- -----------------------------------------------------------------------------
WITH risk_score AS (
    SELECT
        p.patient_id,
        p.age,
        p.smoking_status,
        p.alcohol_use,
        p.hpv_status,
        p.insurance_type,
        (CASE WHEN p.smoking_status = 'Current' THEN 3
              WHEN p.smoking_status = 'Former'  THEN 1 ELSE 0 END
       + CASE WHEN p.alcohol_use   = 'Heavy'    THEN 2
              WHEN p.alcohol_use   = 'Moderate' THEN 1 ELSE 0 END
       + CASE WHEN p.hpv_status    = 'Positive' THEN 2 ELSE 0 END
       + CASE WHEN p.age >= 60                  THEN 1 ELSE 0 END
       + CAST(p.socioeconomic_risk_score * 2 AS INTEGER)
        ) AS composite_risk_score
    FROM patients p
),
last_visit AS (
    SELECT patient_id, MAX(visit_date) AS most_recent_visit
    FROM dental_visits
    GROUP BY patient_id
)
SELECT
    rs.patient_id,
    rs.age,
    rs.smoking_status,
    rs.alcohol_use,
    rs.hpv_status,
    rs.insurance_type,
    rs.composite_risk_score,
    lv.most_recent_visit,
    CAST((JULIANDAY('2026-05-01') - JULIANDAY(lv.most_recent_visit)) AS INTEGER) AS days_since_last_visit
FROM risk_score rs
LEFT JOIN last_visit lv ON rs.patient_id = lv.patient_id
WHERE rs.composite_risk_score >= 6
ORDER BY rs.composite_risk_score DESC, days_since_last_visit DESC
LIMIT 50;