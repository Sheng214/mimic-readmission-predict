# Cohort Table Feature Explanation and Selection Summary

## 1. Overview
This document explains the features in the newly generated **cohort table** and summarizes the process used to select the cohort from the original **MIMIC-IV** dataset.

---

## 2. How the Cohort Was Selected

The cohort was built from the **MIMIC-IV** database (including `hosp`, `icu`, and `note` modules) using the following inclusion and exclusion criteria:

1. **Include all hospital admissions** (no filtering for ICU stay times).
2. **Exclude**:
   - Patients who died during hospitalization.
   - Patients who died within 30 days after discharge.
   - Admissions with missing admission or discharge times.
3. **Require**:
   - At least one discharge note (non-empty text) for each admission.
   - Only one merged discharge note per admission, created by concatenating all available discharge notes in chronological order.
4. **Feature Enrichment**:
   - Diagnostic and procedural information for **Charlson-style comorbidity** calculation.
   - Laboratory values: last, minimum, and maximum measurements.
   - ICU vital signs: averaged over the stay (if available).
5. **Stable Sampling**:
   - A reproducible subset (~20k rows) selected using:
     ```sql
     WHERE MOD(ABS(FARM_FINGERPRINT(CAST(hadm_id AS STRING))), sample_mod) = sample_bucket
     ```
     This ensures consistent sampling across query runs, avoiding the cost of `ORDER BY RAND()`.

---

## 3. Feature Explanation

### **Identifiers**
- `subject_id` — Unique patient identifier.
- `hadm_id` — Unique hospital admission ID.
- `stay_id` — ICU stay identifier (if applicable).
- `note_id` — Identifier for the merged discharge note.

### **Admission Information**
- `admittime` — Timestamp of hospital admission.
- `dischtime` — Timestamp of hospital discharge.
- `deathtime` — If applicable, time of death (filtered out for this cohort).
- `hospital_expire_flag` — Flag indicating in-hospital death (always 0 in this cohort).

### **Discharge Note Content**
- `note_text` — Full merged discharge note text for the admission.

### **Demographics**
- `gender` — Patient gender.
- `anchor_age` — Approximate age at admission (calculated from anchor date and DOB).
- `ethnicity` — Recorded ethnicity.

### **Comorbidity Features**
From the Charlson Comorbidity Index mapping (`charlson_ref` table):
- `mi` — Myocardial infarction.
- `chf` — Congestive heart failure.
- `pvd` — Peripheral vascular disease.
- *(Additional Charlson categories can be included depending on mapping table completeness.)*

From the Elixhauser Comorbidity Index mapping (`elix_ref` table):
- `cardiac_arrhythmias`
- `peripheral_vascular_disorders`
- *(More categories as defined in the mapping references.)*

These are usually binary flags (0/1) indicating presence of the condition.

### **Laboratory Features**
For each lab test of interest:
- `<lab_name>_last` — Most recent lab result before discharge.
- `<lab_name>_min` — Minimum recorded value during admission.
- `<lab_name>_max` — Maximum recorded value during admission.

### **ICU Vital Signs**
For admissions with ICU stays:
- `heart_rate_avg` — Average heart rate during ICU stay.
- `mean_bp_avg` — Average mean arterial blood pressure.
- `resp_rate_avg` — Average respiratory rate.
- *(More vitals if available from the `vitalsign` table.)*

### **Outcome Variable**
- `readmit_30d` — Binary indicator: 1 if patient was readmitted within 30 days post-discharge, 0 otherwise.

---

## 4. Summary of Selection Logic

**Step-by-step**:
1. Start with all admissions in `mimiciv_hosp.admissions`.
2. Join with patient demographics in `patients` table.
3. Filter out deaths during hospitalization or within 30 days post-discharge.
4. Join with discharge notes from `mimiciv_note.discharge`.
5. Merge multiple notes per admission into a single `note_text`.
6. Join with diagnoses and procedures to map comorbidities (Charlson and Elixhauser).
7. Join with laboratory results and ICU vital signs for predictive features.
8. Apply **stable sampling** to select approximately 20,000 rows reproducibly.
9. Save to a timestamped table for version tracking.

---

## 5. Notes on Reproducibility
- The **stable sampling** method ensures the exact same set of admissions is selected each time, as long as the underlying dataset does not change.
- The cohort table includes a **timestamped suffix** in its name to allow version comparison between runs.
