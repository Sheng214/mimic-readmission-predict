-- Timestamped table name so each run is versioned + stable sampling bucket
DECLARE cohort_name STRING;
DECLARE sample_mod INT64 DEFAULT 100;   -- denominator for hash-bucket sampling
DECLARE sample_bucket INT64 DEFAULT 0;  -- pick 0..sample_mod-1; keep fixed for stability

SET cohort_name = FORMAT(
  "cohort_full_features_%s",
  FORMAT_TIMESTAMP('%Y_%m_%d_%H%M', CURRENT_TIMESTAMP())
);

EXECUTE IMMEDIATE FORMAT("""
CREATE OR REPLACE TABLE `striking-canyon-461218-i6.readmission.10k_%s` AS
WITH
/*───────────────────────────────────────────────────────────────────────────────
  0) FULL comorbidity mapping refs (Charlson & Elixhauser)
───────────────────────────────────────────────────────────────────────────────*/
charlson_ref AS (
  -- Myocardial infarction
  SELECT 'mi' AS group_name, 1 AS weight, 9 AS icd_version, r'^(410|412)' AS pattern UNION ALL
  SELECT 'mi', 1, 10, r'^(I21|I22|I252)' UNION ALL

  -- Congestive heart failure
  SELECT 'chf', 1, 9, r'^(428)' UNION ALL
  SELECT 'chf', 1, 10, r'^(I50|I110)' UNION ALL

  -- Peripheral vascular disease
  SELECT 'pvd', 1, 9, r'^(440|441|4439|7854|V434)' UNION ALL
  SELECT 'pvd', 1, 10, r'^(I70|I71|I731|I738|I739|I771|K551|K558|K559|Z958|Z959)' UNION ALL

  -- Cerebrovascular disease
  SELECT 'cevd', 1, 9, r'^(430|431|432|433|434|435|436|437|438)' UNION ALL
  SELECT 'cevd', 1, 10, r'^(I60|I61|I62|I63|I64|G45|I67|I69)' UNION ALL

  -- Dementia
  SELECT 'dementia', 1, 9, r'^(290|3310)' UNION ALL
  SELECT 'dementia', 1, 10, r'^(F00|F01|F02|F03|G30)' UNION ALL

  -- Chronic pulmonary disease (COPD)
  SELECT 'copd', 1, 9, r'^(490|491|492|494|496)' UNION ALL
  SELECT 'copd', 1, 10, r'^(J40|J41|J42|J43|J44|J47)' UNION ALL

  -- Rheumatic disease
  SELECT 'rheum', 1, 9, r'^(7100|7101|7104|7140|7141|7142|71481|725)' UNION ALL
  SELECT 'rheum', 1, 10, r'^(M05|M06|M32|M33|M34)' UNION ALL

  -- Peptic ulcer disease
  SELECT 'pud', 1, 9, r'^(531|532|533|534)' UNION ALL
  SELECT 'pud', 1, 10, r'^(K25|K26|K27|K28)' UNION ALL

  -- Mild liver disease
  SELECT 'mld', 1, 9, r'^(5712|5714|5715|5716|5718|5719)' UNION ALL
  SELECT 'mld', 1, 10, r'^(B18|K70[3-9]|K71[3-9]|K73|K74|K76[0-2])' UNION ALL

  -- Diabetes without chronic complication
  SELECT 'diab_wo', 1, 9, r'^(2500|2501|2502|2503|2507)' UNION ALL
  SELECT 'diab_wo', 1, 10, r'^(E10[0-59]|E11[0-59]|E13[0-59])' UNION ALL

  -- Diabetes with chronic complication
  SELECT 'diab_w', 2, 9, r'^(2504|2505|2506|2508|2509)' UNION ALL
  SELECT 'diab_w', 2, 10, r'^(E10[6-9]|E11[6-9]|E13[6-9])' UNION ALL

  -- Hemiplegia or paraplegia
  SELECT 'hpleg', 2, 9, r'^(342|343|3441)' UNION ALL
  SELECT 'hpleg', 2, 10, r'^(G81|G82)' UNION ALL

  -- Renal disease
  SELECT 'renal', 2, 9, r'^(582|583[0-7,9]|585|586|5880)' UNION ALL
  SELECT 'renal', 2, 10, r'^(N18|N19|N052|N053|N054|N055|N056|N057|N250)' UNION ALL

  -- Any malignancy (no mets)
  SELECT 'cancer', 2, 9, r'^(140|141|142|143|144|145|146|147|148|149|150|151|152|153|154|155|156|157|158|159|160|161|162|163|164|165|166|167|168|169|170|171|172|174|175|176|177|178|179|180|181|182|183|184|185|186|187|188|189|190|191|192|193|194|195|200|201|202|203|204|205|206|207|208)' UNION ALL
  SELECT 'cancer', 2, 10, r'^(C0[0-9]|C1[0-4]|C3[0-4]|C37|C38|C39|C40|C41|C43|C45|C46|C47|C48|C49|C5[0-8]|C6[0-9]|C7[0-6]|C81|C82|C83|C84|C85|C88|C90|C91|C92|C93|C94|C95|C96|C97)$' UNION ALL

  -- Metastatic solid tumor
  SELECT 'metastatic', 6, 9, r'^(196|197|198|199)' UNION ALL
  SELECT 'metastatic', 6, 10, r'^(C77|C78|C79|C80)' UNION ALL

  -- Severe liver disease
  SELECT 'sev_liver', 3, 9, r'^(4560|4561|4562|5722|5723|5724|5728)' UNION ALL
  SELECT 'sev_liver', 3, 10, r'^(I850|I859|I864|I982|K704|K711|K721|K729|K765|K766)$' UNION ALL

  -- AIDS/HIV
  SELECT 'aids', 6, 9, r'^(042|043|044)' UNION ALL
  SELECT 'aids', 6, 10, r'^(B20|B21|B22|B24)'
),

elix_weights AS (
  SELECT 'congestive_heart_failure' AS group_name, 7 AS weight UNION ALL
  SELECT 'cardiac_arrhythmias', 5 UNION ALL
  SELECT 'valvular_disease', 4 UNION ALL
  SELECT 'pulmonary_circulation', 6 UNION ALL
  SELECT 'peripheral_vascular_disorders', 2 UNION ALL
  SELECT 'hypertension_uncomplicated', -1 UNION ALL
  SELECT 'hypertension_complicated', 0 UNION ALL
  SELECT 'paralysis', 5 UNION ALL
  SELECT 'other_neurological', 6 UNION ALL
  SELECT 'chronic_pulmonary', 3 UNION ALL
  SELECT 'diabetes_uncomplicated', 0 UNION ALL
  SELECT 'diabetes_complicated', 2 UNION ALL
  SELECT 'hypothyroidism', -2 UNION ALL
  SELECT 'renal_failure', 5 UNION ALL
  SELECT 'liver_disease', 11 UNION ALL
  SELECT 'peptic_ulcer', 0 UNION ALL
  SELECT 'aids', 0 UNION ALL
  SELECT 'lymphoma', 9 UNION ALL
  SELECT 'metastatic_cancer', 12 UNION ALL
  SELECT 'solid_tumor_without_metastasis', 4 UNION ALL
  SELECT 'rheumatoid_arthritis_collagen', 0 UNION ALL
  SELECT 'coagulopathy', 3 UNION ALL
  SELECT 'obesity', -5 UNION ALL
  SELECT 'weight_loss', 6 UNION ALL
  SELECT 'fluid_electrolyte', 5 UNION ALL
  SELECT 'blood_loss_anemia', -3 UNION ALL
  SELECT 'deficiency_anemias', -2 UNION ALL
  SELECT 'alcohol_abuse', 0 UNION ALL
  SELECT 'drug_abuse', 0 UNION ALL
  SELECT 'psychoses', 0 UNION ALL
  SELECT 'depression', -3
),

-- Elixhauser ICD-9/10 mappings (dotted codes escaped as \\. )
elix_ref AS (
  -- Congestive heart failure
  SELECT 'congestive_heart_failure' AS group_name, 9 AS icd_version,  r'^(39891|402[019]|404[019]|425[4-9]|428)' AS pattern UNION ALL
  SELECT 'congestive_heart_failure', 10, r'^(I09\\.[0-9]|I11\\.0|I13\\.[02]|I50|I97\\.1)' UNION ALL

  -- Cardiac arrhythmias
  SELECT 'cardiac_arrhythmias', 9,  r'^(426|427)' UNION ALL
  SELECT 'cardiac_arrhythmias', 10, r'^(I47|I48|I49|R000|R001|R002)' UNION ALL

  -- Valvular disease
  SELECT 'valvular_disease', 9,  r'^(0932|394|395|396|397|3989|424|7463|7464|7465|7466)' UNION ALL
  SELECT 'valvular_disease', 10, r'^(I05|I06|I07|I08|I09\\.[1]|I34|I35|I36|I37|I38|I39|Q23)' UNION ALL

  -- Pulmonary circulation disorders
  SELECT 'pulmonary_circulation', 9,  r'^(4151|416)' UNION ALL
  SELECT 'pulmonary_circulation', 10, r'^(I26|I27)' UNION ALL

  -- Peripheral vascular disorders
  SELECT 'peripheral_vascular_disorders', 9,  r'^(440|441|4431|4432|4438|4439|4471|5571|5579|V434)' UNION ALL
  SELECT 'peripheral_vascular_disorders', 10, r'^(I70|I71|I73\\.1|I73\\.9|I79\\.0|I79\\.2|K55\\.1|K55\\.8|K55\\.9|Z95\\.8|Z95\\.9)' UNION ALL

  -- Hypertension (uncomplicated / complicated)
  SELECT 'hypertension_uncomplicated', 9,  r'^(401)' UNION ALL
  SELECT 'hypertension_uncomplicated', 10, r'^(I10)$' UNION ALL
  SELECT 'hypertension_complicated', 9,  r'^(402|403|404|405)' UNION ALL
  SELECT 'hypertension_complicated', 10, r'^(I11|I12|I13|I15)' UNION ALL

  -- Paralysis
  SELECT 'paralysis', 9,  r'^(342|343|344[0-6])' UNION ALL
  SELECT 'paralysis', 10, r'^(G81|G82|G83[0-4])' UNION ALL

  -- Other neurological disorders
  SELECT 'other_neurological', 9,  r'^(331|332|333|334|335|336|340|341|345|3481|3483)' UNION ALL
  SELECT 'other_neurological', 10, r'^(G10|G11|G12|G13|G20|G21|G22|G23|G24|G25|G26|G31|G35|G36|G37|G60|G61|G62|G63|G64)' UNION ALL

  -- Chronic pulmonary disease
  SELECT 'chronic_pulmonary', 9,  r'^(490|491|492|494|496)' UNION ALL
  SELECT 'chronic_pulmonary', 10, r'^(J40|J41|J42|J43|J44|J47)' UNION ALL

  -- Diabetes (uncomplicated / complicated)
  SELECT 'diabetes_uncomplicated', 9,  r'^(2500|2501|2502|2503|2507)' UNION ALL
  SELECT 'diabetes_uncomplicated', 10, r'^(E10[0-59]|E11[0-59]|E13[0-59])' UNION ALL
  SELECT 'diabetes_complicated', 9,  r'^(2504|2505|2506|2508|2509)' UNION ALL
  SELECT 'diabetes_complicated', 10, r'^(E10[6-9]|E11[6-9]|E13[6-9])' UNION ALL

  -- Hypothyroidism
  SELECT 'hypothyroidism', 9,  r'^(243|244)' UNION ALL
  SELECT 'hypothyroidism', 10, r'^(E02|E03)' UNION ALL

  -- Renal failure
  SELECT 'renal_failure', 9,  r'^(582|583[0-7,9]|584|585|586|588)' UNION ALL
  SELECT 'renal_failure', 10, r'^(N17|N18|N19)' UNION ALL

  -- Liver disease (any)
  SELECT 'liver_disease', 9,  r'^(0702|0703|0704|0705|0706|0709|570|571|5723|5728)' UNION ALL
  SELECT 'liver_disease', 10, r'^(B18|K70|K71|K72|K73|K74|K75|K76|K77)' UNION ALL

  -- Peptic ulcer disease
  SELECT 'peptic_ulcer', 9,  r'^(531|532|533|534)' UNION ALL
  SELECT 'peptic_ulcer', 10, r'^(K25|K26|K27|K28)' UNION ALL

  -- AIDS/HIV
  SELECT 'aids', 9,  r'^(042|043|044)' UNION ALL
  SELECT 'aids', 10, r'^(B20|B21|B22|B24)' UNION ALL

  -- Lymphoma
  SELECT 'lymphoma', 9,  r'^(200|201|202)' UNION ALL
  SELECT 'lymphoma', 10, r'^(C81|C82|C83|C84|C85|C88)' UNION ALL

  -- Metastatic cancer
  SELECT 'metastatic_cancer', 9,  r'^(196|197|198|199)' UNION ALL
  SELECT 'metastatic_cancer', 10, r'^(C77|C78|C79|C80)' UNION ALL

  -- Solid tumor without metastasis
  SELECT 'solid_tumor_without_metastasis', 9,  r'^(140|141|142|143|144|145|146|147|148|149|150|151|152|153|154|155|156|157|158|159|160|161|162|163|164|165|166|167|168|169|170|171|172|174|175|176|177|178|179|180|181|182|183|184|185|186|187|188|189|190|191|192|193|194|195)' UNION ALL
  SELECT 'solid_tumor_without_metastasis', 10, r'^(C0[0-9]|C1[0-4]|C3[0-4]|C37|C38|C39|C40|C41|C43|C45|C46|C47|C48|C49|C5[0-8]|C6[0-9]|C7[0-6])' UNION ALL

  -- Rheumatoid arthritis / collagen vascular
  SELECT 'rheumatoid_arthritis_collagen', 9,  r'^(446|7010|7100|7101|7104|714|725)' UNION ALL
  SELECT 'rheumatoid_arthritis_collagen', 10, r'^(M05|M06|M32|M33|M34)' UNION ALL

  -- Coagulopathy
  SELECT 'coagulopathy', 9,  r'^(286)' UNION ALL
  SELECT 'coagulopathy', 10, r'^(D65|D66|D67|D68)' UNION ALL

  -- Obesity
  SELECT 'obesity', 9,  r'^(2780)' UNION ALL
  SELECT 'obesity', 10, r'^(E66)' UNION ALL

  -- Weight loss
  SELECT 'weight_loss', 9,  r'^(260|261|262|263|78321)' UNION ALL
  SELECT 'weight_loss', 10, r'^(E40|E41|E42|E43|E44|E45|E46|R634)' UNION ALL

  -- Fluid and electrolyte disorders
  SELECT 'fluid_electrolyte', 9,  r'^(276)' UNION ALL
  SELECT 'fluid_electrolyte', 10, r'^(E86|E87)' UNION ALL

  -- Blood loss anemia
  SELECT 'blood_loss_anemia', 9,  r'^(2851)' UNION ALL
  SELECT 'blood_loss_anemia', 10, r'^(D500)' UNION ALL

  -- Deficiency anemias
  SELECT 'deficiency_anemias', 9,  r'^(280|281|283|2859)' UNION ALL
  SELECT 'deficiency_anemias', 10, r'^(D51|D52|D53|D508|D509|D649)' UNION ALL

  -- Alcohol abuse
  SELECT 'alcohol_abuse', 9,  r'^(291|303|3050|V113)' UNION ALL
  SELECT 'alcohol_abuse', 10, r'^(F10|Y90|Y91|Z714|Z721)' UNION ALL

  -- Drug abuse
  SELECT 'drug_abuse', 9,  r'^(292|304|305[2-9])' UNION ALL
  SELECT 'drug_abuse', 10, r'^(F11|F12|F13|F14|F15|F16|F18|F19|Z715|Z722)' UNION ALL

  -- Psychoses
  SELECT 'psychoses', 9,  r'^(295|29604|29614|29644|29654|297|298)' UNION ALL
  SELECT 'psychoses', 10, r'^(F20|F21|F22|F23|F24|F25|F28|F29)' UNION ALL

  -- Depression
  SELECT 'depression', 9,  r'^(2962|2963|3004|311)' UNION ALL
  SELECT 'depression', 10, r'^(F32|F33)'
),

/*───────────────────────────────────────────────────────────────────────────────
  1) Base admissions, excluding in-hospital and 30-day post-discharge deaths
───────────────────────────────────────────────────────────────────────────────*/
base_adm AS (
  SELECT
    a.subject_id, a.hadm_id,
    a.admittime, a.dischtime,
    a.admission_type, a.admission_location, a.discharge_location,
    a.insurance, a.language, a.marital_status, a.race
  FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
    ON p.subject_id = a.subject_id
  WHERE a.admittime IS NOT NULL
    AND a.dischtime IS NOT NULL
    AND a.deathtime IS NULL
    AND NOT (
      p.dod IS NOT NULL
      AND p.dod BETWEEN DATE(a.dischtime) AND DATE_ADD(DATE(a.dischtime), INTERVAL 30 DAY)
    )
),

/*───────────────────────────────────────────────────────────────────────────────
  2) Require a non-empty discharge note; merge all texts per admission
───────────────────────────────────────────────────────────────────────────────*/
adm_with_disc_note AS (
  SELECT
    n.subject_id,
    n.hadm_id,
    MIN(n.note_id) AS note_id,
    STRING_AGG(
      n.text,
      '\\n\\n-----\\n\\n'
      ORDER BY COALESCE(n.charttime, n.storetime), n.note_id
    ) AS note_text
  FROM `physionet-data.mimiciv_note.discharge` n
  WHERE n.text IS NOT NULL
    AND REGEXP_CONTAINS(TRIM(n.text), r'\\S')
  GROUP BY n.subject_id, n.hadm_id
),

/*───────────────────────────────────────────────────────────────────────────────
  3) Diagnosis / Procedure features (counts, uniqueness)
───────────────────────────────────────────────────────────────────────────────*/
dx_feats AS (
  SELECT
    d.hadm_id,
    COUNT(*) AS dx_count,
    COUNT(DISTINCT d.icd_code) AS dx_code_uniq,
    COUNT(DISTINCT SUBSTR(d.icd_code, 1, 3)) AS dx_root3_uniq
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
  GROUP BY d.hadm_id
),
proc_feats AS (
  SELECT
    p.hadm_id,
    COUNT(*) AS proc_count,
    COUNT(DISTINCT p.icd_code) AS proc_code_uniq,
    COUNT(DISTINCT SUBSTR(p.icd_code, 1, 3)) AS proc_root3_uniq
  FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd` p
  GROUP BY p.hadm_id
),

/*───────────────────────────────────────────────────────────────────────────────
  4) Charlson flags & score; Elixhauser flags & van Walraven score
───────────────────────────────────────────────────────────────────────────────*/
dx_for_maps AS (
  SELECT hadm_id, icd_version, UPPER(REPLACE(icd_code, '.', '')) AS icd_nodot
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
),
charlson_flags_raw AS (
  SELECT d.hadm_id, c.group_name, c.weight
  FROM dx_for_maps d
  JOIN charlson_ref c
    ON d.icd_version = c.icd_version
   AND REGEXP_CONTAINS(d.icd_nodot, c.pattern)
  GROUP BY d.hadm_id, c.group_name, c.weight
),
charlson_flags AS (
  SELECT
    hadm_id,
    MAX(IF(group_name = 'mi', 1, 0))           AS cci_mi,
    MAX(IF(group_name = 'chf', 1, 0))          AS cci_chf,
    MAX(IF(group_name = 'pvd', 1, 0))          AS cci_pvd,
    MAX(IF(group_name = 'cevd', 1, 0))         AS cci_cevd,
    MAX(IF(group_name = 'dementia', 1, 0))     AS cci_dementia,
    MAX(IF(group_name = 'copd', 1, 0))         AS cci_copd,
    MAX(IF(group_name = 'rheum', 1, 0))        AS cci_rheum,
    MAX(IF(group_name = 'pud', 1, 0))          AS cci_pud,
    MAX(IF(group_name = 'mld', 1, 0))          AS cci_mild_liver,
    MAX(IF(group_name = 'diab_wo', 1, 0))      AS cci_diabetes_wo_comp,
    MAX(IF(group_name = 'diab_w', 1, 0))       AS cci_diabetes_w_comp,
    MAX(IF(group_name = 'hpleg', 1, 0))        AS cci_hemiplegia,
    MAX(IF(group_name = 'renal', 1, 0))        AS cci_renal,
    MAX(IF(group_name = 'cancer', 1, 0))       AS cci_any_cancer,
    MAX(IF(group_name = 'metastatic', 1, 0))   AS cci_metastatic_solid,
    MAX(IF(group_name = 'sev_liver', 1, 0))    AS cci_severe_liver,
    MAX(IF(group_name = 'aids', 1, 0))         AS cci_aids
  FROM charlson_flags_raw
  GROUP BY hadm_id
),
charlson_index AS (
  SELECT hadm_id, SUM(DISTINCT weight) AS cci_score
  FROM charlson_flags_raw
  GROUP BY hadm_id
),
elix_flags_raw AS (
  SELECT d.hadm_id, e.group_name
  FROM dx_for_maps d
  JOIN elix_ref e
    ON d.icd_version = e.icd_version
   AND REGEXP_CONTAINS(d.icd_nodot, e.pattern)
  GROUP BY d.hadm_id, e.group_name
),
elix_flags AS (
  SELECT
    hadm_id,
    MAX(IF(group_name = 'congestive_heart_failure', 1, 0)) AS ex_congestive_heart_failure,
    MAX(IF(group_name = 'cardiac_arrhythmias', 1, 0))       AS ex_cardiac_arrhythmias,
    MAX(IF(group_name = 'valvular_disease', 1, 0))          AS ex_valvular_disease,
    MAX(IF(group_name = 'pulmonary_circulation', 1, 0))     AS ex_pulmonary_circulation,
    MAX(IF(group_name = 'peripheral_vascular_disorders', 1, 0)) AS ex_peripheral_vascular_disorders,
    MAX(IF(group_name = 'hypertension_uncomplicated', 1, 0)) AS ex_hypertension_uncomplicated,
    MAX(IF(group_name = 'hypertension_complicated', 1, 0))   AS ex_hypertension_complicated,
    MAX(IF(group_name = 'paralysis', 1, 0))                 AS ex_paralysis,
    MAX(IF(group_name = 'other_neurological', 1, 0))        AS ex_other_neurological,
    MAX(IF(group_name = 'chronic_pulmonary', 1, 0))         AS ex_chronic_pulmonary,
    MAX(IF(group_name = 'diabetes_uncomplicated', 1, 0))    AS ex_diabetes_uncomplicated,
    MAX(IF(group_name = 'diabetes_complicated', 1, 0))      AS ex_diabetes_complicated,
    MAX(IF(group_name = 'hypothyroidism', 1, 0))            AS ex_hypothyroidism,
    MAX(IF(group_name = 'renal_failure', 1, 0))             AS ex_renal_failure,
    MAX(IF(group_name = 'liver_disease', 1, 0))             AS ex_liver_disease,
    MAX(IF(group_name = 'peptic_ulcer', 1, 0))              AS ex_peptic_ulcer,
    MAX(IF(group_name = 'aids', 1, 0))                      AS ex_aids,
    MAX(IF(group_name = 'lymphoma', 1, 0))                  AS ex_lymphoma,
    MAX(IF(group_name = 'metastatic_cancer', 1, 0))         AS ex_metastatic_cancer,
    MAX(IF(group_name = 'solid_tumor_without_metastasis', 1, 0)) AS ex_solid_tumor_without_metastasis,
    MAX(IF(group_name = 'rheumatoid_arthritis_collagen', 1, 0))  AS ex_rheumatoid_arthritis_collagen,
    MAX(IF(group_name = 'coagulopathy', 1, 0))              AS ex_coagulopathy,
    MAX(IF(group_name = 'obesity', 1, 0))                   AS ex_obesity,
    MAX(IF(group_name = 'weight_loss', 1, 0))               AS ex_weight_loss,
    MAX(IF(group_name = 'fluid_electrolyte', 1, 0))         AS ex_fluid_electrolyte,
    MAX(IF(group_name = 'blood_loss_anemia', 1, 0))         AS ex_blood_loss_anemia,
    MAX(IF(group_name = 'deficiency_anemias', 1, 0))        AS ex_deficiency_anemias,
    MAX(IF(group_name = 'alcohol_abuse', 1, 0))             AS ex_alcohol_abuse,
    MAX(IF(group_name = 'drug_abuse', 1, 0))                AS ex_drug_abuse,
    MAX(IF(group_name = 'psychoses', 1, 0))                 AS ex_psychoses,
    MAX(IF(group_name = 'depression', 1, 0))                AS ex_depression
  FROM elix_flags_raw
  GROUP BY hadm_id
),
elix_score AS (
  SELECT
    f.hadm_id,
    SUM(COALESCE(w.weight, 0)) AS elix_vw_score
  FROM (SELECT hadm_id, group_name FROM elix_flags_raw GROUP BY hadm_id, group_name) f
  LEFT JOIN elix_weights w ON w.group_name = f.group_name
  GROUP BY f.hadm_id
),

/*───────────────────────────────────────────────────────────────────────────────
  5) Lab features: min/max per HADM; last value per HADM
───────────────────────────────────────────────────────────────────────────────*/
labevents_filtered AS (
  SELECT l.subject_id, l.hadm_id, l.charttime, l.valuenum, dl.label
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
  JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` dl
    ON dl.itemid = l.itemid
  WHERE l.hadm_id IS NOT NULL
    AND l.valuenum IS NOT NULL
    AND dl.label IN (
      'WBC','White Blood Cells','Platelet Count','Hemoglobin',
      'Sodium','Potassium','Chloride','Bicarbonate',
      'Glucose','Creatinine','Urea Nitrogen','Blood Urea Nitrogen','Lactate'
    )
),
lab_minmax AS (
  SELECT
    hadm_id,
    MIN(IF(label IN ('WBC','White Blood Cells'), valuenum, NULL)) AS wbc_min,
    MAX(IF(label IN ('WBC','White Blood Cells'), valuenum, NULL)) AS wbc_max,
    MIN(IF(label = 'Platelet Count', valuenum, NULL)) AS plt_min,
    MAX(IF(label = 'Platelet Count', valuenum, NULL)) AS plt_max,
    MIN(IF(label = 'Hemoglobin', valuenum, NULL)) AS hgb_min,
    MAX(IF(label = 'Hemoglobin', valuenum, NULL)) AS hgb_max,
    MIN(IF(label = 'Sodium', valuenum, NULL)) AS sodium_min,
    MAX(IF(label = 'Sodium', valuenum, NULL)) AS sodium_max,
    MIN(IF(label = 'Potassium', valuenum, NULL)) AS potassium_min,
    MAX(IF(label = 'Potassium', valuenum, NULL)) AS potassium_max,
    MIN(IF(label = 'Chloride', valuenum, NULL)) AS chloride_min,
    MAX(IF(label = 'Chloride', valuenum, NULL)) AS chloride_max,
    MIN(IF(label = 'Bicarbonate', valuenum, NULL)) AS bicarb_min,
    MAX(IF(label = 'Bicarbonate', valuenum, NULL)) AS bicarb_max,
    MIN(IF(label = 'Glucose', valuenum, NULL)) AS glucose_min,
    MAX(IF(label = 'Glucose', valuenum, NULL)) AS glucose_max,
    MIN(IF(label = 'Creatinine', valuenum, NULL)) AS creat_min,
    MAX(IF(label = 'Creatinine', valuenum, NULL)) AS creat_max,
    MIN(IF(label IN ('Urea Nitrogen','Blood Urea Nitrogen'), valuenum, NULL)) AS bun_min,
    MAX(IF(label IN ('Urea Nitrogen','Blood Urea Nitrogen'), valuenum, NULL)) AS bun_max,
    MIN(IF(label = 'Lactate', valuenum, NULL)) AS lactate_min,
    MAX(IF(label = 'Lactate', valuenum, NULL)) AS lactate_max
  FROM labevents_filtered
  GROUP BY hadm_id
),
lab_last_rows AS (
  SELECT
    hadm_id, label, valuenum,
    ROW_NUMBER() OVER (PARTITION BY hadm_id, label ORDER BY charttime DESC) AS rn
  FROM labevents_filtered
),
lab_last AS (
  SELECT
    hadm_id,
    MAX(IF(label IN ('WBC','White Blood Cells'), valuenum, NULL)) AS wbc_last,
    MAX(IF(label = 'Platelet Count', valuenum, NULL)) AS plt_last,
    MAX(IF(label = 'Hemoglobin', valuenum, NULL)) AS hgb_last,
    MAX(IF(label = 'Sodium', valuenum, NULL)) AS sodium_last,
    MAX(IF(label = 'Potassium', valuenum, NULL)) AS potassium_last,
    MAX(IF(label = 'Chloride', valuenum, NULL)) AS chloride_last,
    MAX(IF(label = 'Bicarbonate', valuenum, NULL)) AS bicarb_last,
    MAX(IF(label = 'Glucose', valuenum, NULL)) AS glucose_last,
    MAX(IF(label = 'Creatinine', valuenum, NULL)) AS creat_last,
    MAX(IF(label IN ('Urea Nitrogen','Blood Urea Nitrogen'), valuenum, NULL)) AS bun_last,
    MAX(IF(label = 'Lactate', valuenum, NULL)) AS lactate_last
  FROM lab_last_rows
  WHERE rn = 1
  GROUP BY hadm_id
),
labs_wide AS (
  SELECT
    hadm_id,
    -- min/max
    m.wbc_min, m.wbc_max, m.plt_min, m.plt_max, m.hgb_min, m.hgb_max,
    m.sodium_min, m.sodium_max, m.potassium_min, m.potassium_max,
    m.chloride_min, m.chloride_max, m.bicarb_min, m.bicarb_max,
    m.glucose_min, m.glucose_max, m.creat_min, m.creat_max,
    m.bun_min, m.bun_max, m.lactate_min, m.lactate_max,
    -- last
    l.wbc_last, l.plt_last, l.hgb_last, l.sodium_last, l.potassium_last,
    l.chloride_last, l.bicarb_last, l.glucose_last, l.creat_last, l.bun_last, l.lactate_last
  FROM lab_minmax m
  FULL OUTER JOIN lab_last l USING (hadm_id)
),

/*───────────────────────────────────────────────────────────────────────────────
  6) ICU stays (all stays per admission) + ICU vitals per stay + ICU counts
───────────────────────────────────────────────────────────────────────────────*/
icu_stays AS (
  SELECT
    i.hadm_id,
    i.stay_id,
    i.intime  AS icu_intime,
    i.outtime AS icu_outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
),
icu_counts AS (
  SELECT hadm_id, COUNT(*) AS icu_stay_count
  FROM icu_stays
  GROUP BY hadm_id
),
ce_filtered AS (
  SELECT
    i.stay_id,
    di.label,
    ce.valuenum,
    ce.valueuom
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  JOIN `physionet-data.mimiciv_3_1_icu.d_items` di
    ON di.itemid = ce.itemid
  JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
    ON i.stay_id = ce.stay_id
  WHERE ce.valuenum IS NOT NULL
    AND di.linksto = 'chartevents'
    AND di.label IN (
      'Heart Rate','Respiratory Rate','SpO2','O2 saturation pulseoxymetry',
      'Temperature Celsius','Temperature Fahrenheit',
      'Arterial Blood Pressure mean','Arterial Blood Pressure systolic','Arterial Blood Pressure diastolic',
      'Non Invasive Blood Pressure mean','Non Invasive Blood Pressure systolic','Non Invasive Blood Pressure diastolic'
    )
),
vitals_per_stay AS (
  SELECT
    stay_id,
    AVG(IF(label = 'Heart Rate',               valuenum, NULL)) AS hr_mean,
    AVG(IF(label = 'Respiratory Rate',         valuenum, NULL)) AS rr_mean,
    AVG(IF(label IN ('SpO2','O2 saturation pulseoxymetry'), valuenum, NULL)) AS spo2_mean,
    AVG(CASE
          WHEN label = 'Temperature Fahrenheit' THEN (valuenum - 32.0) * 5.0/9.0
          WHEN label = 'Temperature Celsius'     THEN valuenum
        END) AS temp_c_mean,
    AVG(IF(label = 'Arterial Blood Pressure mean',      valuenum, NULL)) AS map_art_mean,
    AVG(IF(label = 'Non Invasive Blood Pressure mean',  valuenum, NULL)) AS map_ni_mean,
    AVG(IF(label = 'Arterial Blood Pressure systolic',  valuenum, NULL)) AS sbp_art_mean,
    AVG(IF(label = 'Arterial Blood Pressure diastolic', valuenum, NULL)) AS dbp_art_mean,
    AVG(IF(label = 'Non Invasive Blood Pressure systolic',  valuenum, NULL)) AS sbp_ni_mean,
    AVG(IF(label = 'Non Invasive Blood Pressure diastolic', valuenum, NULL)) AS dbp_ni_mean
  FROM ce_filtered
  GROUP BY stay_id
)

/*───────────────────────────────────────────────────────────────────────────────
  7) Final cohort — row grain marked; stable hash-bucket sampling on hadm_id
───────────────────────────────────────────────────────────────────────────────*/
SELECT
  -- Grain marker & stable row id
  CASE WHEN i.stay_id IS NULL THEN 'ADMISSION_ONLY' ELSE 'ICU_STAY' END AS row_grain,
  IFNULL(CAST(i.stay_id AS STRING), CONCAT('HADM_', CAST(b.hadm_id AS STRING))) AS grain_id,

  -- Admission-level features
  b.*,

  -- Admission-level ICU summary
  COALESCE(ic.icu_stay_count, 0) AS icu_stay_count,

  -- Discharge note (merged)
  nd.note_id,
  nd.note_text,

  -- Diagnosis / procedure counts
  dx.dx_count, dx.dx_code_uniq, dx.dx_root3_uniq,
  pr.proc_count, pr.proc_code_uniq, pr.proc_root3_uniq,

  -- Charlson flags + score
  cf.cci_mi, cf.cci_chf, cf.cci_pvd, cf.cci_cevd, cf.cci_dementia,
  cf.cci_copd, cf.cci_rheum, cf.cci_pud, cf.cci_mild_liver,
  cf.cci_diabetes_wo_comp, cf.cci_diabetes_w_comp, cf.cci_hemiplegia,
  cf.cci_renal, cf.cci_any_cancer, cf.cci_metastatic_solid,
  cf.cci_severe_liver, cf.cci_aids,
  COALESCE(ci.cci_score, 0) AS cci_score,

  -- Elixhauser flags + van Walraven score
  ef.ex_congestive_heart_failure,
  ef.ex_cardiac_arrhythmias,
  ef.ex_valvular_disease,
  ef.ex_pulmonary_circulation,
  ef.ex_peripheral_vascular_disorders,
  ef.ex_hypertension_uncomplicated,
  ef.ex_hypertension_complicated,
  ef.ex_paralysis,
  ef.ex_other_neurological,
  ef.ex_chronic_pulmonary,
  ef.ex_diabetes_uncomplicated,
  ef.ex_diabetes_complicated,
  ef.ex_hypothyroidism,
  ef.ex_renal_failure,
  ef.ex_liver_disease,
  ef.ex_peptic_ulcer,
  ef.ex_aids,
  ef.ex_lymphoma,
  ef.ex_metastatic_cancer,
  ef.ex_solid_tumor_without_metastasis,
  ef.ex_rheumatoid_arthritis_collagen,
  ef.ex_coagulopathy,
  ef.ex_obesity,
  ef.ex_weight_loss,
  ef.ex_fluid_electrolyte,
  ef.ex_blood_loss_anemia,
  ef.ex_deficiency_anemias,
  ef.ex_alcohol_abuse,
  ef.ex_drug_abuse,
  ef.ex_psychoses,
  ef.ex_depression,
  COALESCE(es.elix_vw_score, 0) AS elix_vw_score,

  -- Labs (admission-level)
  lw.wbc_last, lw.wbc_min, lw.wbc_max,
  lw.plt_last, lw.plt_min, lw.plt_max,
  lw.hgb_last, lw.hgb_min, lw.hgb_max,
  lw.sodium_last, lw.sodium_min, lw.sodium_max,
  lw.potassium_last, lw.potassium_min, lw.potassium_max,
  lw.chloride_last, lw.chloride_min, lw.chloride_max,
  lw.bicarb_last, lw.bicarb_min, lw.bicarb_max,
  lw.glucose_last, lw.glucose_min, lw.glucose_max,
  lw.creat_last, lw.creat_min, lw.creat_max,
  lw.bun_last, lw.bun_min, lw.bun_max,
  lw.lactate_last, lw.lactate_min, lw.lactate_max,

  -- ICU stay metadata (NULL if non-ICU)
  i.stay_id       AS icu_stay_id,
  i.icu_intime    AS icu_intime,
  i.icu_outtime   AS icu_outtime,

  -- ICU vitals averaged within each stay (NULL for non-ICU)
  v.hr_mean, v.rr_mean, v.spo2_mean, v.temp_c_mean,
  COALESCE(v.map_art_mean, v.map_ni_mean) AS map_mean_pref,
  v.map_art_mean, v.map_ni_mean,
  v.sbp_art_mean, v.dbp_art_mean, v.sbp_ni_mean, v.dbp_ni_mean

FROM base_adm b
JOIN adm_with_disc_note nd
  ON nd.subject_id = b.subject_id AND nd.hadm_id = b.hadm_id
LEFT JOIN dx_feats        dx ON dx.hadm_id = b.hadm_id
LEFT JOIN proc_feats      pr ON pr.hadm_id = b.hadm_id
LEFT JOIN charlson_flags  cf ON cf.hadm_id = b.hadm_id
LEFT JOIN charlson_index  ci ON ci.hadm_id = b.hadm_id
LEFT JOIN elix_flags      ef ON ef.hadm_id = b.hadm_id
LEFT JOIN elix_score      es ON es.hadm_id = b.hadm_id
LEFT JOIN labs_wide       lw ON lw.hadm_id = b.hadm_id
LEFT JOIN icu_stays        i ON i.hadm_id = b.hadm_id
LEFT JOIN icu_counts      ic ON ic.hadm_id = b.hadm_id
LEFT JOIN vitals_per_stay  v ON v.stay_id = i.stay_id

-- Stable hash-bucket sampling by admission id (keeps all ICU rows for sampled admissions)
WHERE MOD(ABS(FARM_FINGERPRINT(CAST(b.hadm_id AS STRING))), @sample_mod) = @sample_bucket
LIMIT 10000
""", cohort_name)
USING sample_mod AS sample_mod, sample_bucket AS sample_bucket;
