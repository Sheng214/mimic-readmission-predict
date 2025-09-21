# Readmission Cohort, Feature Dictionary & Sampling (UPDATED)

**Version:** 2025‑08‑21  
**Scope:** Defines cohort logic, feature dictionary, and *patient‑level* deterministic sampling for the 30‑day readmission project using MIMIC‑IV in BigQuery.

---

## What changed in this update
Insert the following two sentences **verbatim** in the indicated sections (see “Where to insert” below):

1) **Earliest‑admission prefilter:**  
“**We restrict the cohort to each patient’s *earliest* eligible admission *before any sampling* by selecting the minimum `admittime` per `subject_id` in SQL.**”

2) **Patient‑level 10k guarantee:**  
“**Sampling is applied at the *patient* level (by `subject_id`), and we cap the cohort to **exactly 10,000 patients** *before* any joins to ICU tables to avoid row fan‑out affecting `LIMIT`.**”

---

## Where to insert the two sentences
- **Section: Cohort Definition → Final Cohort Rules (end of the list):** add sentence (1) as a new bullet.  
- **Section: Sampling Strategy (first paragraph):** append sentence (2) to the end of the first paragraph.

> These placements preserve the logical flow: define the cohort once (earliest admission rule belongs to the cohort), then describe how we sample (10k patient guarantee belongs to sampling).

---

## Full document (with updates already merged)

### 1) Cohort Definition

**Goal:** Build a cohort of index admissions and a target indicating an unplanned readmission within 30 days after discharge, excluding deaths up to 30 days post‑discharge.

**Data sources (BigQuery):**
- `physionet-data.mimiciv_3_1_hosp.patients`
- `physionet-data.mimiciv_3_1_hosp.admissions`
- `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`, `procedures_icd`
- (Optional) `physionet-data.mimiciv_3_1_icu.icustays` (for ICU vitals/flags — *joined only after patient sampling*)
- Local copies/derived tables as noted in your project.

**Inclusion criteria:**
- Valid `admittime` and `dischtime` (drop rows with missing/invalid times).
- A non‑empty discharge summary/note for the index admission.
- Patient alive at discharge **and** no death within 30 days post‑discharge.

**Final cohort rules (applied in SQL order):**
- Define eligible admissions per patient.  
- Compute readmission label using the *next* admission within 30 days; if focusing on *unplanned* readmissions, apply `admission_type != 'ELECTIVE'` on the **next** admission check.  
- **We restrict the cohort to each patient’s *earliest* eligible admission *before any sampling* by selecting the minimum `admittime` per `subject_id` in SQL.**

**Rationale:** Fixing one index admission per patient avoids leakage from later admissions and simplifies downstream modeling.

---

### 2) Target Variable

- **Target:** `readmit_30d_unplanned` ∈ {0,1}.  
- **Positive rule:** Next admission for the same `subject_id` starts within 30 days of the index `dischtime` **and** that next admission’s `admission_type != 'ELECTIVE'`.  
- **Negative rule:** No qualifying next admission, or only elective next admissions within the window.

---

### 3) Deterministic Patient‑Level Sampling

We use a *deterministic hash bucket* of `subject_id` to produce stable subsamples across runs (e.g., 10%, 20%, …) for rapid iteration and reproducibility. **Sampling is applied at the *patient* level (by `subject_id`), and we cap the cohort to exactly 10,000 patients *before* any joins to ICU tables to avoid row fan‑out affecting `LIMIT`.**

**Key principles**
- **Deterministic:** `FARM_FINGERPRINT(CAST(subject_id AS STRING))` → stable hash.  
- **Bucketed:** Keep patients where `MOD(ABS(hash), sample_mod) < sample_bucket`.  
- **Cardinality control:** Enforce the final **10,000 patients** with `LIMIT` **on a distinct patient list**, *then* join to features.

---

### 4) Recommended SQL structure (drop‑in CTEs)

Below is a template to implement the updated rules. Names can be adapted to your project’s dataset IDs.

```sql
DECLARE cohort_name STRING;
DECLARE sample_mod INT64 DEFAULT 1000;    -- denominator for hash-bucket sampling
DECLARE sample_bucket INT64 DEFAULT 500;  -- keep first N buckets (0..N-1); adjust to exceed 10k patients
DECLARE target_n INT64 DEFAULT 10000;     -- fixed patient count

SET cohort_name = FORMAT(
  'cohort_full_features_%s',
  FORMAT_TIMESTAMP('%Y_%m_%d_%H%M', CURRENT_TIMESTAMP())
);

EXECUTE IMMEDIATE FORMAT("""
CREATE OR REPLACE TABLE `your-project.your_dataset.%s` AS
WITH
eligible_admissions AS (
  SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.admission_type
  FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
  WHERE a.admittime IS NOT NULL
    AND a.dischtime IS NOT NULL
    AND a.deathtime IS NULL                          -- alive at discharge
),
-- (A) Earliest eligible admission per patient (prefilter BEFORE sampling)
earliest_adm AS (
  SELECT AS VALUE
    ARRAY_AGG(e ORDER BY e.admittime ASC LIMIT 1)[OFFSET(0)]
  FROM eligible_admissions e
  GROUP BY e.subject_id
),
-- (B) Readmission label using the *next* admission; if unplanned, exclude ELECTIVE on the *next* one
next_adm AS (
  SELECT
    ea.subject_id,
    ea.hadm_id AS index_hadm_id,
    MIN_BY(a.hadm_id, a.admittime) AS next_hadm_id,
    MIN(a.admittime) AS next_admittime
  FROM earliest_adm ea
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON a.subject_id = ea.subject_id
    AND a.admittime > ea.dischtime
    AND TIMESTAMP_DIFF(a.admittime, ea.dischtime, DAY) <= 30
    AND a.admission_type != 'ELECTIVE'               -- unplanned criterion on NEXT admission
  GROUP BY ea.subject_id, ea.hadm_id
),
labels AS (
  SELECT
    ea.subject_id,
    ea.hadm_id,
    IF(na.next_hadm_id IS NULL, 0, 1) AS readmit_30d_unplanned
  FROM earliest_adm ea
  LEFT JOIN next_adm na
    ON na.subject_id = ea.subject_id
    AND na.index_hadm_id = ea.hadm_id
),
-- (C) PATIENT-LEVEL deterministic sampling BEFORE any ICU joins
sampled_patients AS (
  SELECT subject_id
  FROM earliest_adm
  WHERE MOD(ABS(FARM_FINGERPRINT(CAST(subject_id AS STRING))), sample_mod) < sample_bucket
),
-- (D) Enforce EXACTLY 10k patients (cardinality control)
final_patients AS (
  SELECT subject_id
  FROM sampled_patients
  LIMIT target_n
),
-- (E) Join features only AFTER patient list is fixed (prevents ICU fan-out from skewing LIMIT)
dx AS (
  SELECT hadm_id, ARRAY_AGG(STRUCT(icd_code, icd_version)) AS diag_codes
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
  GROUP BY hadm_id
),
px AS (
  SELECT hadm_id, ARRAY_AGG(STRUCT(icd_code, icd_version)) AS proc_codes
  FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
  GROUP BY hadm_id
),
-- (Optional) ICU-derived aggregates should be computed at hadm_id or subject_id level FIRST,
-- then joined to avoid row multiplication.
icu_agg AS (
  SELECT
    s.subject_id,
    s.hadm_id,
    AVG(s.heart_rate) AS icu_mean_hr
  FROM `physionet-data.mimiciv_3_1_icu.vitalsign` s
  GROUP BY s.subject_id, s.hadm_id
),
features AS (
  SELECT
    ea.subject_id,
    ea.hadm_id,
    -- add engineered features here, e.g., durations, code counts, last/min/max labs, etc.
    SAFE.TIMESTAMP_DIFF(ea.dischtime, ea.admittime, HOUR) AS los_hours,
    -- examples of code-based features:
    ARRAY_LENGTH(dx.diag_codes) AS n_dx_codes,
    ARRAY_LENGTH(px.proc_codes) AS n_px_codes,
    icu.icu_mean_hr
  FROM earliest_adm ea
  LEFT JOIN dx ON dx.hadm_id = ea.hadm_id
  LEFT JOIN px ON px.hadm_id = ea.hadm_id
  LEFT JOIN icu_agg icu ON icu.hadm_id = ea.hadm_id
  WHERE ea.subject_id IN (SELECT subject_id FROM final_patients)
)
SELECT
  f.*,
  l.readmit_30d_unplanned
FROM features f
JOIN labels l
  ON l.subject_id = f.subject_id
  AND l.hadm_id = f.hadm_id
""", cohort_name);
```

**Notes:**
- **Why prefilter earliest admission?** Guarantees exactly one index per patient, simplifies labels and avoids leakage.  
- **Why patient‑level sampling?** Ensures stable and fair selection; avoids per‑admission bias.  
- **Why limit on distinct patients?** `LIMIT` after joins to ICU can shrink/expand arbitrarily due to row multiplication; lock the patient list first.

---

### 5) Feature Dictionary (high level)

- **Lengths of stay:** `los_hours` = `TIMESTAMP_DIFF(dischtime, admittime, HOUR)`; provide both hours and days if useful.
- **Comorbidity proxies:** counts/flags derived from ICD blocks (Charlson/Elixhauser style). Keep regex maps in a reference CTE.
- **Procedural burden:** `n_px_codes`; add service‑line flags if needed.
- **Labs (per index admission):** last/min/max of key labs (e.g., creatinine, sodium). Compute in a labs CTE at `hadm_id` granularity before joining.
- **ICU aggregates (optional):** mean HR, min/max MAP, etc., aggregated to `hadm_id` *before* join.
- **Demographics:** age at admission, sex.
- **Text availability:** boolean flags for presence of discharge note; do **not** use note text unless explicitly modeling NLP features separately.

---

### 6) Quality & Reproducibility Checklist

- [ ] Earliest eligible admission per patient selected **before** sampling.  
- [ ] Deterministic patient‑level sampling using hash of `subject_id`.  
- [ ] `LIMIT 10000` applied to **patients** (distinct `subject_id`) prior to any ICU joins.  
- [ ] ICU/lab features aggregated at `hadm_id` (or `subject_id`) before joining to avoid fan‑out.  
- [ ] Versioned table name via timestamped `cohort_name`.  
- [ ] Seed parameters (`sample_mod`, `sample_bucket`) recorded in table metadata or a run log.

---

### 7) Troubleshooting

- **I’m getting fewer than 10k patients.** Increase `sample_bucket` (e.g., 500 → 700 out of 1000) to widen the bucket range, or reduce exclusions (e.g., remove optional filters) until the sampled pool exceeds 10k before the `LIMIT`.  
- **My `LIMIT` still seems unstable.** Confirm you apply it on a distinct patient subquery *before* any joins to high‑cardinality tables (ICU events, notes).  
- **Hash seems to change across runs.** Ensure `subject_id` is cast to `STRING` and the same hashing function (`FARM_FINGERPRINT`) is used consistently.

---

## Changelog

- **2025‑08‑21:** Added earliest‑admission prefilter **before sampling**; enforced patient‑level `LIMIT 10000` **before** ICU joins; updated SQL template and checklist.
