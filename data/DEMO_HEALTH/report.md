# Causal Inference Pipeline Report

**User ID**: `DEMO_HEALTH`
**Generated**: 2026-03-18

---

## Stage 0: Preprocess
> Parses raw data files and prepares them for downstream analysis.

- **Records**: 93
- **Columns**: 4
- **Date range**: Sep 15, 2021 - Dec 20, 2024

### Data Sample (first 10 rows)

| datetime | activity_type | content | location |
| --- | --- | --- | --- |
| 2021-09-15T09:30:00+02:00 | consultation | Consultation Dr. Laurent (Médecine Générale). Découverte diabète type 2 (HbA1c 7.8%). HTA grade 1 (152/95). Initiation metformine 1000mg x2/j, ramipril 5mg/j. | Cabinet Dr. Laurent, 18 rue de Rivoli, Paris 1er |
| 2021-09-15T10:00:00+02:00 | prescription | Metformine 1000mg — 2 cp/jour. Ramipril 5mg — 1 cp/jour. Durée: 3 mois. | Cabinet Dr. Laurent, 18 rue de Rivoli, Paris 1er |
| 2021-10-02T08:15:00+02:00 | lab_result | LDL 4.1 mmol/L, HDL 1.2 mmol/L, TG 2.3 mmol/L. Glycémie à jeun: 8.9 mmol/L. Créatinine: 78 µmol/L, DFG: 82 mL/min. | Laboratoire BioSanté, 42 bd Haussmann, Paris 8e |
| 2022-01-10T10:00:00+01:00 | consultation | Suivi Dr. Laurent. HbA1c 7.2% (amélioration). TA 145/92. Poids 78kg. Augmentation ramipril à 10mg. | Cabinet Dr. Laurent, 18 rue de Rivoli, Paris 1er |
| 2022-06-14T08:30:00+02:00 | lab_result | LDL 4.2 mmol/L, HDL 1.3 mmol/L, TG 2.1 mmol/L. HbA1c 6.9%. DFG: 80 mL/min. | Laboratoire BioSanté, 42 bd Haussmann, Paris 8e |
| 2022-06-20T11:00:00+02:00 | prescription | Atorvastatine 20mg — 1 cp/jour (soir). Ajouté au traitement existant. | Cabinet Dr. Laurent, 18 rue de Rivoli, Paris 1er |
| 2022-10-18T09:45:00+02:00 | consultation | Suivi 4 mois post-statine. LDL 2.8 mmol/L (∙33%). Bonne tolérance. HbA1c 6.8%. TA 138/88. | Cabinet Dr. Laurent, 18 rue de Rivoli, Paris 1er |
| 2023-03-07T14:00:00+01:00 | procedure | Cardiologie Prof. Moreau. ECG: rythme sinusal normal. Écho: FEVG 62%, pas de valvulopathie. Recommandation: activité physique 150min/sem. | Hôpital Pitié-Salpêtrière, Cardiologie, Paris 13e |
| 2023-06-12T10:30:00+02:00 | consultation | TA toujours 140/90 malgré ramipril 10mg. Ajout amlodipine 5mg. LDL 2.5 mmol/L. HbA1c 6.7%. | Cabinet Dr. Laurent, 18 rue de Rivoli, Paris 1er |
| 2023-09-25T09:00:00+02:00 | vital_sign | TA 128/82 mmHg, FC 72 bpm, Poids 76.5kg, IMC 28.1. | Cabinet Dr. Laurent, 18 rue de Rivoli, Paris 1er |

### Column Descriptions

| Column | Type | Description |
| --- | --- | --- |
| datetime | Utf8 | Timestamp of the medical event |
| activity_type | Utf8 | Type of medical activity (consultation, prescription, lab_result, procedure, vital_sign) |
| content | Utf8 | Free-text description of the medical event |
| location | Utf8 | Location where the event took place |

---

## Stage 1a: Latent Model
> Proposes a latent causal model based on domain knowledge alone, specifying theoretical constructs and their causal relationships.

- **Outcome**: cardiovascular_risk
- **Treatments**: lipid_burden, vascular_inflammation, glycemic_control, arterial_pressure, medication_adherence

### Constructs

| Name | Description | Role | Temporal | Outcome |
| --- | --- | --- | --- | --- |
| cardiovascular_risk | Overall 10-year cardiovascular event risk integrating lipid, glycemic, hemodynamic, and inflammatory factors | endogenous | time_varying | Yes |
| lipid_burden | Atherogenic lipid profile reflecting LDL, triglycerides, and HDL balance | endogenous | time_varying | No |
| vascular_inflammation | Arterial wall inflammatory state driven by atherogenic lipid infiltration and immune activation | endogenous | time_varying | No |
| glycemic_control | Quality of blood glucose regulation reflecting insulin sensitivity and hepatic glucose output | endogenous | time_varying | No |
| arterial_pressure | Systemic blood pressure level reflecting vascular resistance and cardiac output | endogenous | time_varying | No |
| medication_adherence | Prescription fill rate and treatment compliance across all prescribed medications | exogenous | time_varying | No |
| genetic_predisposition | Inherited cardiovascular risk variants (LDLR, APOB, PCSK9, 9p21.3, APOE4) affecting lipid metabolism and vascular biology independently | exogenous | time_invariant | No |
| psychosocial_stress | Chronic psychosocial burden (occupational stress, social isolation, depressive symptoms) driving neuroendocrine dysregulation | exogenous | time_invariant | No |

### Causal Edges

| Cause | Effect | Lagged | Description |
| --- | --- | --- | --- |
| lipid_burden | vascular_inflammation | No | Oxidized LDL infiltrates the arterial intima, recruiting monocytes and triggering foam cell formation (atherogenesis initiation) |
| vascular_inflammation | cardiovascular_risk | No | Inflammatory plaque destabilization thins fibrous caps and promotes thrombosis, driving acute cardiovascular events |
| glycemic_control | cardiovascular_risk | No | Chronic hyperglycemia generates advanced glycation end products (AGEs), accelerating micro- and macrovascular complications |
| arterial_pressure | cardiovascular_risk | No | Elevated blood pressure increases cardiac workload and accelerates atherosclerotic plaque progression |
| medication_adherence | lipid_burden | Yes | Atorvastatin inhibits HMG-CoA reductase, reducing hepatic LDL synthesis |
| medication_adherence | glycemic_control | Yes | Metformin improves hepatic insulin sensitivity and reduces gluconeogenesis |
| medication_adherence | arterial_pressure | Yes | Ramipril and amlodipine reduce systemic vascular resistance |
| genetic_predisposition | lipid_burden | No | Familial hypercholesterolemia variants (LDLR, APOB, PCSK9 gain-of-function) elevate baseline LDL cholesterol |
| genetic_predisposition | cardiovascular_risk | No | 9p21.3 risk locus and APOE4 confer lipid-independent cardiovascular risk through endothelial dysfunction and platelet hyperreactivity |
| psychosocial_stress | glycemic_control | No | Chronic HPA axis activation elevates cortisol, driving hepatic gluconeogenesis and peripheral insulin resistance |
| psychosocial_stress | cardiovascular_risk | No | Sustained sympathetic overdrive and systemic inflammation from chronic stress independently increase cardiovascular event risk |

---

## Stage 1b: Measurement & Nonparametric Identification
> Maps latent constructs to observable indicators and verifies nonparametric identifiability via do-calculus.

> **PIPELINE STOPPED**: No identifiable treatment effects remain after Stage 1b.

### Non-Identifiable Treatments

- **glycemic_control**: confounded by psychosocial_stress

### Identifiable Treatments

| Treatment | Method | Estimand |
| --- | --- | --- |
| lipid_burden | do_calculus | ∑_{vasc_infl} P(vasc_infl \| lipid) ∑_{lipid'} P(cv_risk \| lipid', vasc_infl) P(lipid') |
| vascular_inflammation | do_calculus | P(cardiovascular_risk \| do(vascular_inflammation)) |
| arterial_pressure | do_calculus | P(cardiovascular_risk \| do(arterial_pressure)) |
| medication_adherence | do_calculus | P(cardiovascular_risk \| do(medication_adherence)) |

### Indicators

| Indicator | Construct | Type | Aggregation | How to Measure |
| --- | --- | --- | --- | --- |
| ldl_cholesterol | lipid_burden | continuous | mean | LDL cholesterol in mmol/L from lipid panel |
| triglycerides | lipid_burden | continuous | mean | Serum triglycerides in mmol/L |
| hdl_cholesterol | lipid_burden | continuous | mean | HDL cholesterol in mmol/L from lipid panel (EAL); inversely related to atherogenic burden |
| crp | vascular_inflammation | continuous | mean | High-sensitivity C-reactive protein (hs-CRP) in mg/L; marker of systemic and vascular inflammation |
| fibrinogen | vascular_inflammation | continuous | mean | Plasma fibrinogen in g/L; acute-phase reactant reflecting inflammatory and prothrombotic state |
| hba1c | glycemic_control | continuous | mean | Glycated hemoglobin %; lower = better control |
| fasting_glucose | glycemic_control | continuous | mean | Fasting blood glucose in mmol/L |
| systolic_bp | arterial_pressure | continuous | mean | Systolic blood pressure in mmHg |
| diastolic_bp | arterial_pressure | continuous | mean | Diastolic blood pressure in mmHg |
| resting_heart_rate | arterial_pressure | continuous | mean | Resting heart rate in bpm at consultation |
| prescription_renewals | medication_adherence | continuous | mean | Proportion of prescriptions renewed on time via DemoHealth (0–1, quarterly) |
| appointment_attendance | medication_adherence | continuous | mean | Proportion of kept vs missed DemoHealth appointments (0–1, quarterly) |

---

## Stage 2: Data Extraction
> Dispatches worker LLMs to extract indicator observations from raw activity data, processing each chunk independently.

- **Workers**: 4 succeeded, 0 failed, 4 total

### Extractions per Indicator

| Indicator | Count |
| --- | --- |
| ldl_cholesterol | 13 |
| triglycerides | 13 |
| hba1c | 13 |
| fasting_glucose | 15 |
| systolic_bp | 24 |
| diastolic_bp | 24 |
| resting_heart_rate | 24 |
| crp | 13 |
| fibrinogen | 7 |
| prescription_renewals | 13 |
| hdl_cholesterol | 13 |
| appointment_attendance | 13 |

### Extractions Sample

| Indicator | Tick | Value |
| --- | --- | --- |
| ldl_cholesterol | — | 4.1 |
| ldl_cholesterol | — | 4.1 |
| ldl_cholesterol | — | 4.1 |
| ldl_cholesterol | — | 3.5 |
| ldl_cholesterol | — | 3.5 |
| ldl_cholesterol | — | 3.2 |
| ldl_cholesterol | — | 3.3 |
| ldl_cholesterol | — | 3 |
| ldl_cholesterol | — | 2.7 |
| ldl_cholesterol | — | 2.6 |
| ldl_cholesterol | — | 2.4 |
| ldl_cholesterol | — | 2.3 |
| ldl_cholesterol | — | 2.1 |
| triglycerides | — | 2.3 |
| triglycerides | — | 2.1 |
| triglycerides | — | 2 |
| triglycerides | — | 2.1 |
| triglycerides | — | 2.3 |
| triglycerides | — | 2 |
| triglycerides | — | 1.9 |

---

## Stage 3: Validation
> Validates extraction quality, checking for missing data, outliers, and consistency across indicators.

> **PIPELINE STOPPED**: Data validation failed.

---

## Stage 4: Model Specification
> Specifies prior distributions and model parameters using domain knowledge and empirical data.

### State Dynamics

$$
\begin{aligned}
\eta_{\text{lipid}}(0) &\sim \mathcal{N}(\mu_{0,\text{lipid}},\; \sigma_{0,\text{lipid}}^{2}) \\
\eta_{\text{pressure}}(0) &\sim \mathcal{N}(\mu_{0,\text{pressure}},\; \sigma_{0,\text{pressure}}^{2}) \\
\eta_{\text{inflammation}}(0) &\sim \mathcal{N}(\mu_{0,\text{inflammation}},\; \sigma_{0,\text{inflammation}}^{2}) \\
\eta_{\text{lipid}}(t) &= \rho_{\text{lipid}} \, \eta_{\text{lipid}}(t\!-\!1) + \beta_{\text{adherence} \to \text{lipid}} \, \eta_{\text{adherence}}(t\!-\!1) + \varepsilon_{\text{lipid}}(t) \\
\eta_{\text{pressure}}(t) &= \rho_{\text{pressure}} \, \eta_{\text{pressure}}(t\!-\!1) + \beta_{\text{adherence} \to \text{pressure}} \, \eta_{\text{adherence}}(t\!-\!1) + \varepsilon_{\text{pressure}}(t) \\
\eta_{\text{inflammation}}(t) &= \rho_{\text{inflammation}} \, \eta_{\text{inflammation}}(t\!-\!1) + \beta_{\text{lipid} \to \text{inflammation}} \, \eta_{\text{lipid}}(t\!-\!1) + \varepsilon_{\text{inflammation}}(t) \\
\varepsilon_{\text{lipid}}(t) &\sim \mathcal{N}(0,\, \sigma_{\text{lipid}}^2) \\
\varepsilon_{\text{pressure}}(t) &\sim \mathcal{N}(0,\, \sigma_{\text{pressure}}^2) \\
\varepsilon_{\text{inflammation}}(t) &\sim \mathcal{N}(0,\, \sigma_{\text{inflammation}}^2)
\end{aligned}
$$

### Observation Model

$$
\begin{aligned}
y_{\text{ldl cholesterol}}(t) &\sim \mathcal{N}(\lambda_{\text{ldl cholesterol}} \, \eta_{\text{lipid burden}}(t),\; \sigma_{\text{ldl cholesterol}}^{2}) \\
y_{\text{triglycerides}}(t) &\sim \mathcal{N}(\lambda_{\text{triglycerides}} \, \eta_{\text{lipid burden}}(t),\; \sigma_{\text{triglycerides}}^{2}) \\
y_{\text{hba1c}}(t) &\sim \mathcal{N}(\lambda_{\text{hba1c}} \, \eta_{\text{glycemic control}}(t),\; \sigma_{\text{hba1c}}^{2}) \\
y_{\text{fasting glucose}}(t) &\sim \mathcal{N}(\lambda_{\text{fasting glucose}} \, \eta_{\text{glycemic control}}(t),\; \sigma_{\text{fasting glucose}}^{2}) \\
y_{\text{systolic bp}}(t) &\sim \mathcal{N}(\lambda_{\text{systolic bp}} \, \eta_{\text{arterial pressure}}(t),\; \sigma_{\text{systolic bp}}^{2}) \\
y_{\text{diastolic bp}}(t) &\sim \mathcal{N}(\lambda_{\text{diastolic bp}} \, \eta_{\text{arterial pressure}}(t),\; \sigma_{\text{diastolic bp}}^{2}) \\
y_{\text{resting heart rate}}(t) &\sim \mathcal{N}(\lambda_{\text{resting heart rate}} \, \eta_{\text{arterial pressure}}(t),\; \sigma_{\text{resting heart rate}}^{2}) \\
y_{\text{crp}}(t) &\sim \mathcal{N}(\lambda_{\text{crp}} \, \eta_{\text{vascular inflammation}}(t),\; \sigma_{\text{crp}}^{2}) \\
y_{\text{fibrinogen}}(t) &\sim \mathcal{N}(\lambda_{\text{fibrinogen}} \, \eta_{\text{vascular inflammation}}(t),\; \sigma_{\text{fibrinogen}}^{2}) \\
y_{\text{prescription renewals}}(t) &\sim \text{Beta}(\sigma(\lambda_{\text{prescription renewals}} \, \eta_{\text{medication adherence}}(t))\,\phi,\; (1 - \sigma(\lambda_{\text{prescription renewals}} \, \eta_{\text{medication adherence}}(t)))\,\phi) \\
y_{\text{hdl cholesterol}}(t) &\sim \mathcal{N}(\lambda_{\text{hdl cholesterol}} \, \eta_{\text{lipid burden}}(t),\; \sigma_{\text{hdl cholesterol}}^{2}) \\
y_{\text{appointment attendance}}(t) &\sim \text{Beta}(\sigma(\lambda_{\text{appointment attendance}} \, \eta_{\text{medication adherence}}(t))\,\phi,\; (1 - \sigma(\lambda_{\text{appointment attendance}} \, \eta_{\text{medication adherence}}(t)))\,\phi)
\end{aligned}
$$

### Measurement Model

| Variable | Distribution | Link | Reasoning | Sources |
| --- | --- | --- | --- | --- |
| ldl_cholesterol | gaussian | identity | Continuous biomarker (mmol/L), approximately normal | — |
| triglycerides | gaussian | identity | Continuous biomarker (mmol/L) | — |
| hba1c | gaussian | identity | Continuous percentage, normal in clinical range | — |
| fasting_glucose | gaussian | identity | Continuous biomarker (mmol/L) | — |
| systolic_bp | gaussian | identity | Continuous mmHg, normal distribution | — |
| diastolic_bp | gaussian | identity | Continuous mmHg | — |
| resting_heart_rate | gaussian | identity | Continuous bpm | — |
| crp | gaussian | identity | Continuous mg/L; hs-CRP measuring vascular inflammation | — |
| fibrinogen | gaussian | identity | Continuous g/L; acute-phase reactant for inflammatory state | — |
| prescription_renewals | beta | logit | Proportion bounded [0,1] | — |
| hdl_cholesterol | gaussian | identity | Continuous biomarker (mmol/L); inversely loads on lipid burden | — |
| appointment_attendance | beta | logit | Proportion bounded [0,1]; DemoHealth attendance rate | — |

### Prior Distributions

$$
\begin{aligned}
\beta_{\text{lipid cv}} &\sim \mathcal{N}(0.4,\; 0.15) \\
\beta_{\text{pressure cv}} &\sim \mathcal{N}(0.35,\; 0.12) \\
\beta_{\text{glycemic cv}} &\sim \mathcal{N}(-0.25,\; 0.12) \\
\beta_{\text{lipid inflammation}} &\sim \mathcal{N}(0.45,\; 0.15) \\
\beta_{\text{inflammation cv}} &\sim \mathcal{N}(0.38,\; 0.12) \\
\beta_{\text{adherence lipid}} &\sim \mathcal{N}(-0.35,\; 0.1) \\
\beta_{\text{adherence pressure}} &\sim \mathcal{N}(-0.3,\; 0.12) \\
\rho_{\text{lipid}} &\sim \text{Beta}(8,\; 2) \\
\rho_{\text{pressure}} &\sim \text{Beta}(5,\; 3) \\
\sigma_{\text{lipid}} &\sim \text{HalfNormal}(0.3) \\
\sigma_{\text{pressure}} &\sim \text{HalfNormal}(5) \\
\rho_{\text{inflammation}} &\sim \text{Beta}(4,\; 2) \\
\lambda_{\text{triglycerides lipid burden}} &\sim \text{HalfNormal}(1) \\
\lambda_{\text{hdl cholesterol lipid burden}} &\sim \text{HalfNormal}(1) \\
\lambda_{\text{fibrinogen vascular inflammation}} &\sim \text{HalfNormal}(1) \\
\lambda_{\text{fasting glucose glycemic control}} &\sim \text{HalfNormal}(1) \\
\lambda_{\text{diastolic bp arterial pressure}} &\sim \text{HalfNormal}(1) \\
\lambda_{\text{resting heart rate arterial pressure}} &\sim \text{HalfNormal}(1) \\
\lambda_{\text{appointment attendance medication adherence}} &\sim \text{HalfNormal}(1) \\
\sigma_{\text{ldl cholesterol}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{triglycerides}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{hdl cholesterol}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{crp}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{fibrinogen}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{hba1c}} &\sim \text{HalfNormal}(0.3) \\
\sigma_{\text{fasting glucose}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{systolic bp}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{diastolic bp}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{resting heart rate}} &\sim \text{HalfNormal}(0.5) \\
\phi_{\text{prescription renewals}} &\sim \text{LogNormal}(2,\; 1) \\
\phi_{\text{appointment attendance}} &\sim \text{LogNormal}(2,\; 1) \\
\mu_{0,\,\text{lipid}} &\sim \mathcal{N}(0,\; 2) \\
\sigma_{0,\,\text{lipid}} &\sim \text{HalfNormal}(2) \\
\mu_{0,\,\text{pressure}} &\sim \mathcal{N}(0,\; 2) \\
\sigma_{0,\,\text{pressure}} &\sim \text{HalfNormal}(2) \\
\mu_{0,\,\text{inflammation}} &\sim \mathcal{N}(0,\; 2) \\
\sigma_{0,\,\text{inflammation}} &\sim \text{HalfNormal}(2)
\end{aligned}
$$

| Parameter | Prior | Sources | Reasoning |
| --- | --- | --- | --- |
| $\beta_{\text{lipid cv}}$ | Normal(mu=0.4, sigma=0.15) | [CTT Collaborators (2010) — Efficacy of LDL-lowering and vascular events](https://doi.org/10.1016/S0140-6736(10)61350-5); [Ference et al. (2017) — LDL and cardiovascular risk: Mendelian randomization](https://doi.org/10.1016/j.jacc.2017.09.006) | Strong causal evidence from Mendelian randomization and CTT meta-analysis: LDL is a direct driver of atherosclerosis |
| $\beta_{\text{pressure cv}}$ | Normal(mu=0.35, sigma=0.12) | [Ettehad et al. (2016) — Blood pressure lowering and CV risk](https://doi.org/10.1016/S0140-6736(15)01225-8); [SPRINT Research Group (2015) — Intensive vs standard BP targets](https://doi.org/10.1056/NEJMoa1511939) | Well-established dose-response: every 10 mmHg SBP reduction yields ~20% CV risk reduction |
| $\beta_{\text{glycemic cv}}$ | Normal(mu=-0.25, sigma=0.12) | [UKPDS 35 (2000) — HbA1c and complications in T2DM](https://doi.org/10.1136/bmj.321.7258.405); [Holman et al. (2008) — UKPDS 10-year post-trial monitoring](https://doi.org/10.1056/NEJMoa0806470) | Negative effect: better glycemic control (higher construct value) reduces CV risk. UKPDS established the dose-response. |
| $\beta_{\text{lipid inflammation}}$ | Normal(mu=0.45, sigma=0.15) | [Ridker et al. (2008) — CRP, LDL, and cardiovascular events (JUPITER)](https://doi.org/10.1056/NEJMoa0807646); [Libby et al. (2011) — Inflammation in atherosclerosis pathogenesis](https://doi.org/10.1016/j.jacc.2011.10.365) | Strong positive effect: atherogenic lipids drive vascular inflammation through oxidized LDL infiltration and foam cell formation. JUPITER showed statins reduce both LDL and CRP. |
| $\beta_{\text{inflammation cv}}$ | Normal(mu=0.38, sigma=0.12) | [Ridker et al. (2017) — CANTOS: Anti-inflammatory therapy and CV outcomes](https://doi.org/10.1056/NEJMoa1707914); [Emerging Risk Factors Collaboration (2010) — CRP and CV disease](https://doi.org/10.1016/S0140-6736(09)61717-7) | CANTOS proved causality: targeting inflammation directly (without lipid changes) reduces CV events. CRP is independently predictive of CV risk. |
| $\beta_{\text{adherence lipid}}$ | Normal(mu=-0.35, sigma=0.1) | [CARDS Trial (2004) — Atorvastatin in T2DM](https://doi.org/10.1016/S0140-6736(04)16895-5) | Statin adherence directly reduces LDL burden; strong negative effect. |
| $\beta_{\text{adherence pressure}}$ | Normal(mu=-0.3, sigma=0.12) | [HOPE Trial (2000) — Ramipril and CV events](https://doi.org/10.1056/NEJM200001203420301) | ACE-inhibitor adherence reduces arterial pressure; ramipril has proven CV benefit beyond BP lowering. |
| $\rho_{\text{lipid}}$ | Beta(alpha=8, beta=2) | [Mora et al. (2019) — LDL autocorrelation in statin-treated patients](https://doi.org/10.1161/CIRCULATIONAHA.118.038034) | LDL is highly persistent month-to-month (slow metabolic turnover) |
| $\rho_{\text{pressure}}$ | Beta(alpha=5, beta=3) | [Muntner et al. (2015) — Visit-to-visit BP variability](https://doi.org/10.1161/HYPERTENSIONAHA.115.05422) | Blood pressure moderately autocorrelated; more variable than lipids |
| $\sigma_{\text{lipid}}$ | HalfNormal(sigma=0.3) | [Bangalore et al. (2015) — LDL-C variability and cardiovascular outcomes](https://doi.org/10.1016/j.jacc.2015.02.068) | Weakly informative prior on lipid residual SD |
| $\sigma_{\text{pressure}}$ | HalfNormal(sigma=5) | [Rothwell et al. (2010) — BP variability and stroke risk](https://doi.org/10.1016/S0140-6736(10)60309-1) | Weakly informative prior on BP residual SD (mmHg scale) |
| $\rho_{\text{inflammation}}$ | Beta(alpha=4, beta=2) | [Emerging Risk Factors Collaboration (2010) — CRP stability](https://doi.org/10.1016/S0140-6736(09)61717-7) | CRP has moderate month-to-month persistence; Beta(4,2) centers at 0.67 reflecting this autocorrelation. |
| $\lambda_{\text{triglycerides lipid burden}}$ | HalfNormal(sigma=1) | [Mora et al. (2014) — Lipid biomarker factor structure](https://doi.org/10.1161/CIRCULATIONAHA.114.010604) | Triglycerides co-load with LDL on atherogenic lipid burden; weakly informative HalfNormal allows data to determine scale. |
| $\lambda_{\text{hdl cholesterol lipid burden}}$ | HalfNormal(sigma=1) | [Nordestgaard & Varbo (2014) — Triglycerides and cardiovascular disease](https://doi.org/10.1016/S0140-6736(14)61177-6) | HDL reflects lipid burden inversely; constraint is positive here since the construct represents net atherogenic load and HDL measurement direction is handled in the link function. |
| $\lambda_{\text{fibrinogen vascular inflammation}}$ | HalfNormal(sigma=1) | [Fibrinogen Studies Collaboration (2005) — Plasma fibrinogen and CV disease](https://doi.org/10.1001/jama.294.14.1799) | Fibrinogen is an acute-phase reactant loading on the same vascular inflammation factor as CRP; HalfNormal(1) is weakly informative. |
| $\lambda_{\text{fasting glucose glycemic control}}$ | HalfNormal(sigma=1) | [DCCT/EDIC Research Group (2005) — HbA1c and glucose as glycemic markers](https://doi.org/10.2337/diacare.28.5.1231) | Fasting glucose and HbA1c both reflect underlying glycemic control; weakly informative prior lets the data determine the relative scaling. |
| $\lambda_{\text{diastolic bp arterial pressure}}$ | HalfNormal(sigma=1) | [Franklin et al. (2009) — Systolic vs diastolic BP components](https://doi.org/10.1161/HYPERTENSIONAHA.108.125609) | Diastolic BP co-loads with systolic BP on an arterial pressure factor; factor loading typically near 0.7–0.9. |
| $\lambda_{\text{resting heart rate arterial pressure}}$ | HalfNormal(sigma=1) | [Palatini & Julius (2004) — Heart rate and cardiovascular risk](https://doi.org/10.1016/j.ijcard.2003.12.028) | Heart rate is a weaker indicator of arterial pressure than SBP/DBP, reflecting autonomic rather than purely vascular tone; prior allows both weak and strong loadings. |
| $\lambda_{\text{appointment attendance medication adherence}}$ | HalfNormal(sigma=1) | [Osterberg & Blaschke (2005) — Adherence to medication](https://doi.org/10.1056/NEJMra050100) | Appointment attendance and prescription renewals co-load on a latent medication adherence factor; weakly informative prior. |
| $\sigma_{\text{ldl cholesterol}}$ | HalfNormal(sigma=0.5) | [Glasziou & Irwig (1995) — Biological variation of cholesterol](https://doi.org/10.1016/0895-4356(95)00015-1) | Observation noise for LDL cholesterol reflects biological variation and lab assay imprecision (CV ~9%). |
| $\sigma_{\text{triglycerides}}$ | HalfNormal(sigma=0.5) | [Marcovina et al. (1994) — Biological variability of lipids](https://doi.org/10.1093/clinchem/40.6.869) | Triglycerides have higher measurement noise than other lipids due to fasting variation and diurnal effects. |
| $\sigma_{\text{hdl cholesterol}}$ | HalfNormal(sigma=0.5) | [Smith et al. (1993) — Biological variability of HDL](https://doi.org/10.1093/clinchem/39.11.2276) | HDL has moderate measurement noise from biological variation and assay imprecision. |
| $\sigma_{\text{crp}}$ | HalfNormal(sigma=0.5) | [Macy et al. (1997) — Variability of CRP](https://doi.org/10.1093/clinchem/43.1.52) | CRP has high within-person variability due to acute inflammatory triggers; weakly informative HalfNormal allows flexible noise estimation. |
| $\sigma_{\text{fibrinogen}}$ | HalfNormal(sigma=0.5) | [Fibrinogen Studies Collaboration (2005)](https://doi.org/10.1001/jama.294.14.1799) | Fibrinogen measurement noise from biological variation and assay imprecision. |
| $\sigma_{\text{hba1c}}$ | HalfNormal(sigma=0.3) | [Rohlfing et al. (2002) — Biological variation of HbA1c](https://doi.org/10.2337/diacare.25.2.275) | HbA1c is a stable 3-month average with low measurement noise; tighter prior reflects this. |
| $\sigma_{\text{fasting glucose}}$ | HalfNormal(sigma=0.5) | [Mooy et al. (1996) — Intra-individual variation of fasting glucose](https://doi.org/10.1007/BF00400569) | Fasting glucose has moderate day-to-day variation from metabolic fluctuations. |
| $\sigma_{\text{systolic bp}}$ | HalfNormal(sigma=0.5) | [Parati et al. (2013) — Blood pressure variability](https://doi.org/10.1097/HJH.0b013e3283621c71) | Systolic BP has substantial measurement noise from white-coat effects, circadian variation, and measurement technique. |
| $\sigma_{\text{diastolic bp}}$ | HalfNormal(sigma=0.5) | [Parati et al. (2013) — Blood pressure variability](https://doi.org/10.1097/HJH.0b013e3283621c71) | Diastolic BP measurement noise from similar sources as SBP but smaller magnitude. |
| $\sigma_{\text{resting heart rate}}$ | HalfNormal(sigma=0.5) | [Palatini (1999) — Heart rate variability](https://doi.org/10.1097/00004872-199912000-00002) | Resting heart rate varies with anxiety, caffeine, activity level at time of measurement. |
| $\phi_{\text{prescription renewals}}$ | LogNormal(mu=2, sigma=1) | [Ferrari & Cribari-Neto (2004) — Beta regression](https://doi.org/10.1080/0266476042000214501) | Beta precision for prescription renewal proportions; LogNormal(2,1) centers around φ≈7 allowing moderate overdispersion. |
| $\phi_{\text{appointment attendance}}$ | LogNormal(mu=2, sigma=1) | [Ferrari & Cribari-Neto (2004) — Beta regression](https://doi.org/10.1080/0266476042000214501) | Beta precision for appointment attendance proportions; same weakly informative prior as prescription renewals. |
| $\mu_{0,\,\text{lipid}}$ | Normal(mu=0, sigma=2) | — | Default weakly informative prior for the initial state mean of lipid. |
| $\sigma_{0,\,\text{lipid}}$ | HalfNormal(sigma=2) | — | Default weakly informative prior for the initial state standard deviation of lipid. |
| $\mu_{0,\,\text{pressure}}$ | Normal(mu=0, sigma=2) | — | Default weakly informative prior for the initial state mean of pressure. |
| $\sigma_{0,\,\text{pressure}}$ | HalfNormal(sigma=2) | — | Default weakly informative prior for the initial state standard deviation of pressure. |
| $\mu_{0,\,\text{inflammation}}$ | Normal(mu=0, sigma=2) | — | Default weakly informative prior for the initial state mean of inflammation. |
| $\sigma_{0,\,\text{inflammation}}$ | HalfNormal(sigma=2) | — | Default weakly informative prior for the initial state standard deviation of inflammation. |

---

## Stage 4b: Parametric Identifiability
> Checks whether the specified model parameters are identifiable from the available data.

### T-Rule

- **Free parameters**: 11
- **Manifest variables**: 12
- **Timepoints**: 36
- **Moment conditions**: 25
- **Satisfies**: Yes

### Inference Structure

- **Likelihood path**: particle
- **Auto method**: aux_gibbs
- **First-pass RB**: inactive
- **First-pass RB reason**: no_executable_partition

### Parameter Classification

| Parameter | Classification | Contraction Ratio |
| --- | --- | --- |
| beta_lipid_inflammation | identified | 0.860 |
| beta_pressure_cv | identified | 0.850 |
| beta_inflammation_cv | identified | 0.840 |
| sigma_pressure | practically_unidentifiable | 0.420 |

### Sensitivity Analysis

- **Deficient directions**: 2/11
- **Parameters**: 11
- **Draws**: 8

| Parameter | Sensitivity Norm | Effective SV | SV Status | Normalized SV | Norm. Status |
| --- | --- | --- | --- | --- | --- |
| beta_lipid_inflammation | 2.840 | 1.930 | pass | 48.200 | pass |
| beta_pressure_cv | 2.310 | 1.410 | pass | 35.600 | pass |
| beta_inflammation_cv | 2.670 | 1.930 | pass | 42.100 | pass |
| beta_glycemic_lipid | 1.520 | 0.870 | pass | 21.400 | pass |
| beta_cv_pressure | 1.890 | 1.410 | pass | 28.700 | pass |
| beta_glycemic_inflammation | 1.140 | 0.520 | pass | 14.300 | pass |
| rho_cv | 3.210 | 2.640 | pass | 56.800 | pass |
| rho_lipid | 2.950 | 2.640 | pass | 51.300 | pass |
| rho_pressure | 1.730 | 0.310 | pass | 7.200 | warn |
| sigma_cv | 0.870 | 0.140 | pass | 3.800 | warn |
| sigma_pressure | 0.090 | 0.003 | warn | 0.600 | fail |

---

## Stage 5a: SVI Preflight
> Fast variational fit as a diagnostic before expensive inference. Shows ELBO convergence and approximate posterior.

### SVI / ELBO Convergence

```
ELBO loss over optimization steps
 1822.20 │•                                                           
         │ ••                                                         
         │   ••                                                       
         │     ••                                                     
         │       ••                                                   
         │         •••                                                
         │            ••                                              
         │              ••                                            
         │                ••                                          
 1356.90 │                  ••                                        
         └────────────────────────────────────────────────────────────
  x: [0.000, 19.000]
  mean=8.985  sd=5.751  mode=0.000
```
Initial loss: 1822.2, Final loss: 1356.9, Improvement: 25.5%, Converged: No

### Posterior Marginals

```
beta_lipid_cv  (mean=0.440, sd=0.130, HDI=[0.180, 0.680])
    5.00 │      •                                                     
         │      │•                                                    
         │      │ │                                                   
         │     •  │                                                   
         │     │  │                                                   
         │     │  •                                                   
         │    •    │                                                  
         │    │    │                                                  
         │   •     •                                                  
    0.02 │•••       •                                                 
         └────────────────────────────────────────────────────────────
  x: [0.000, 0.800]
  mean=0.479  sd=0.123  mode=0.480
```

```
beta_pressure_cv  (mean=0.370, sd=0.120, HDI=[0.140, 0.580])
    5.10 │      •                                                     
         │      ││                                                    
         │     • •                                                    
         │     │  │                                                   
         │     │  │                                                   
         │    •   │                                                   
         │    │   •                                                   
         │   •     │                                                  
         │  •      •                                                  
    0.03 │••        •                                                 
         └────────────────────────────────────────────────────────────
  x: [0.000, 0.700]
  mean=0.397  sd=0.110  mode=0.420
```

```
beta_glycemic_cv  (mean=-0.250, sd=0.110, HDI=[-0.480, -0.050])
    5.20 │      •                                                     
         │      ││                                                    
         │     • •                                                    
         │     │  │                                                   
         │     │  │                                                   
         │     │  │                                                   
         │    •   •                                                   
         │    │    │                                                  
         │   •     │                                                  
    0.02 │•••      ••                                                 
         └────────────────────────────────────────────────────────────
  x: [-0.650, 0.150]
  mean=-0.182  sd=0.114  mode=-0.170
```

### Posterior Pairs

```
beta_lipid_cv vs beta_pressure_cv
   0.49 │                 •                         •      
        │                              ••                  
        │                                                  
        │              •                                   
        │              •                                   
        │                           •                      
        │•           •      •                              
        │                       •                         •
        │                                                  
        │                       •     •   •                
        │                                                  
        │                   •                     •        
        │                   •                              
        │                                                  
   0.21 │                                   •              
        └──────────────────────────────────────────────────
        0.23                                          0.64
  n=20  x: mean=0.436 sd=0.097  y: mean=0.361 sd=0.081
```
Pearson r: -0.116

*SVI Preflight: svi | 500 samples | 8.3s*

---

## Stage 5b: Inference & Diagnostics
> Fits the Bayesian model via MCMC or SVI and runs convergence and sensitivity diagnostics.

### SMC Diagnostics

- **Particles**: 20
- **Levels**: 12

#### Tempering Schedule & ESS

| Level | β | ESS | Accept Rate |
| --- | --- | --- | --- |
| 0 | 0.012 | 18 | 72.0% |
| 1 | 0.041 | 16 | 68.0% |
| 2 | 0.098 | 15 | 61.0% |
| 3 | 0.187 | 13 | 55.0% |
| 4 | 0.312 | 12 | 48.0% |
| 5 | 0.471 | 11 | 43.0% |
| 6 | 0.622 | 11 | 46.0% |
| 7 | 0.753 | 12 | 51.0% |
| 8 | 0.861 | 14 | 56.0% |
| 9 | 0.934 | 15 | 62.0% |
| 10 | 0.978 | 16 | 67.0% |
| 11 | 1.000 | 17 | 71.0% |

```
ESS over tempering levels
   18.40 │•                                                 
         │ │         •                                      
         │ •        •                                       
         │  •      •                                        
         │   │    •                                         
         │   •    │                                         
         │    • ••                                          
   10.70 │     •                                            
         └──────────────────────────────────────────────────
  x: [0.000, 11.000]
  mean=5.467  sd=3.692  mode=0.000
```
Min ESS: 11, Mean ESS: 14, Final ESS: 17

### Posterior Predictive Checks

**Status**: Misfit detected

| Variable | Check | Result | Value | Message |
| --- | --- | --- | --- | --- |
| ldl_cholesterol | calibration | Pass | 0.941 | 95% CI coverage: 94.1% (expected ~95%) |
| ldl_cholesterol | autocorrelation | Pass | 0.180 | Residual autocorrelation at lag 1: 0.18 |
| ldl_cholesterol | variance | Pass | 0.860 | Predicted variance 0.62 vs observed 0.72 (ratio 0.86) |
| systolic_bp | calibration | Pass | 0.963 | 95% CI coverage: 96.3% (expected ~95%) |
| systolic_bp | autocorrelation | Pass | 0.250 | Residual autocorrelation at lag 1: 0.25 |
| systolic_bp | variance | Pass | 0.900 | Predicted variance 88.2 vs observed 98.5 (ratio 0.90) |
| triglycerides | calibration | Pass | 0.938 | 95% CI coverage: 93.8% (expected ~95%) |
| triglycerides | autocorrelation | Pass | 0.140 | Residual autocorrelation at lag 1: 0.14 |
| triglycerides | variance | Pass | 0.910 | Predicted variance 0.16 vs observed 0.18 (ratio 0.91) |
| hba1c | calibration | Fail | 0.885 | 95% CI coverage: 88.5% (expected ~95%) |
| hba1c | autocorrelation | Pass | 0.210 | Residual autocorrelation at lag 1: 0.21 |
| hba1c | variance | Pass | 0.820 | Predicted variance 0.17 vs observed 0.21 (ratio 0.82) |
| fasting_glucose | calibration | Pass | 0.947 | 95% CI coverage: 94.7% (expected ~95%) |
| fasting_glucose | autocorrelation | Pass | 0.190 | Residual autocorrelation at lag 1: 0.19 |
| fasting_glucose | variance | Pass | 0.880 | Predicted variance 1.28 vs observed 1.45 (ratio 0.88) |
| diastolic_bp | calibration | Pass | 0.952 | 95% CI coverage: 95.2% (expected ~95%) |
| diastolic_bp | autocorrelation | Pass | 0.160 | Residual autocorrelation at lag 1: 0.16 |
| diastolic_bp | variance | Pass | 0.930 | Predicted variance 39.3 vs observed 42.3 (ratio 0.93) |
| resting_heart_rate | calibration | Pass | 0.942 | 95% CI coverage: 94.2% (expected ~95%) |
| resting_heart_rate | autocorrelation | Pass | 0.120 | Residual autocorrelation at lag 1: 0.12 |
| resting_heart_rate | variance | Pass | 0.950 | Predicted variance 27.3 vs observed 28.7 (ratio 0.95) |
| prescription_renewals | calibration | Pass | 0.956 | 95% CI coverage: 95.6% (expected ~95%) |
| prescription_renewals | autocorrelation | Fail | 0.350 | Residual autocorrelation at lag 1: 0.35 |
| prescription_renewals | variance | Pass | 0.870 | Predicted variance 0.0026 vs observed 0.003 (ratio 0.87) |
| hdl_cholesterol | calibration | Pass | 0.935 | 95% CI coverage: 93.5% (expected ~95%) |
| hdl_cholesterol | autocorrelation | Pass | 0.150 | Residual autocorrelation at lag 1: 0.15 |
| hdl_cholesterol | variance | Pass | 0.890 | Predicted variance 0.008 vs observed 0.009 (ratio 0.89) |
| appointment_attendance | calibration | Pass | 0.961 | 95% CI coverage: 96.1% (expected ~95%) |
| appointment_attendance | autocorrelation | Pass | 0.290 | Residual autocorrelation at lag 1: 0.29 |
| appointment_attendance | variance | Pass | 0.830 | Predicted variance 0.0025 vs observed 0.003 (ratio 0.83) |
| crp | calibration | Pass | 0.948 | 95% CI coverage: 94.8% (expected ~95%) |
| crp | autocorrelation | Pass | 0.170 | Residual autocorrelation at lag 1: 0.17 |
| crp | variance | Pass | 0.900 | Predicted variance 0.38 vs observed 0.42 (ratio 0.90) |
| fibrinogen | calibration | Pass | 0.931 | 95% CI coverage: 93.1% (expected ~95%) |
| fibrinogen | autocorrelation | Pass | 0.220 | Residual autocorrelation at lag 1: 0.22 |
| fibrinogen | variance | Pass | 0.850 | Predicted variance 0.11 vs observed 0.13 (ratio 0.85) |

#### Posterior Predictive Overlays

```
ldl_cholesterol (• observed, ◦ median)
    4.10 │◦•                                                          
         │ ◦◦•                                                        
         │   ◦◦                                                       
         │    •◦◦◦•                                                   
         │        ◦◦                                                  
         │          ◦◦                                                
         │          ••◦◦◦                                             
         │               ◦◦  •                                        
         │               • ◦◦│                                        
    2.00 │                  •◦                                        
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=3.055  sd=0.617  range=[2.000, 4.100]
  ◦ series 2: n=20  mean=3.050  sd=0.577  range=[2.100, 4.000]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.150, MAE: 0.125, Pearson r: 0.971

```
systolic_bp (• observed, ◦ median)
  148.00 │◦◦                                                          
         │• ◦◦                                                        
         │  ••◦◦                                                      
         │    • ◦◦                                                    
         │       •◦◦◦  ••                                             
         │           ◦◦│ •                                            
         │            │◦◦ │                                           
         │            ││ ◦◦◦                                          
         │            •     ◦•                                        
  124.00 │                  •◦                                        
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=136.250  sd=6.115  range=[124.000, 147.000]
  ◦ series 2: n=20  mean=136.600  sd=6.960  range=[125.000, 148.000]
```
95% CI coverage: 100.0% (20/20), RMSE: 3.186, MAE: 2.450, Pearson r: 0.891

```
triglycerides (• observed, ◦ median)
    2.10 │◦◦◦•                                                        
         │ │ ││                                                       
         │ ••◦◦◦◦    •                                                
         │     │││   ││                                               
         │     ││◦◦◦◦ •    •                                          
         │     •     ◦◦◦◦  ││                                         
         │              ││ ││                                         
         │              │◦◦◦◦                                         
         │              │ │  │                                        
    1.60 │              ••   ◦                                        
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=1.870  sd=0.155  range=[1.600, 2.100]
  ◦ series 2: n=20  mean=1.875  sd=0.148  range=[1.600, 2.100]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.102, MAE: 0.065, Pearson r: 0.773

```
hba1c (• observed, ◦ median)
    8.20 │◦ •                                                         
         │ ◦◦◦◦••  •   •                                              
         │ ││ │◦◦│• │  ││                                             
         │ •  •  ◦◦ │  ││   •                                         
         │       • ◦◦◦◦ │•  ││                                        
         │          •  ◦◦││ ││                                        
         │              │◦◦ ││                                        
         │              ││•◦◦◦                                        
         │              ││                                            
    7.00 │              •                                             
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=7.740  sd=0.335  range=[7.000, 8.200]
  ◦ series 2: n=20  mean=7.700  sd=0.292  range=[7.200, 8.200]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.251, MAE: 0.180, Pearson r: 0.696

```
fasting_glucose (• observed, ◦ median)
    8.20 │        •                                                   
         │       • │                                                  
         │◦◦◦  • │ │                                                  
         │ │ ◦◦◦││ │                                                  
         │ ••• │◦◦◦◦         •                                        
         │    •     ◦        │                                        
         │          │◦◦◦◦◦   │                                        
         │          ││ ││ ◦◦◦◦                                        
         │          •  ││  │ │                                        
    6.40 │             •   ••                                         
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=7.180  sd=0.505  range=[6.400, 8.200]
  ◦ series 2: n=20  mean=7.230  sd=0.344  range=[6.700, 7.800]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.400, MAE: 0.310, Pearson r: 0.621

```
diastolic_bp (• observed, ◦ median)
   95.80 │•                                                           
         │ │   •                                                      
         │◦◦•  │•         •                                           
         │ │◦◦◦◦ │    •   │•                                          
         │ ││ • ◦◦◦   │•• │ │                                         
         │ ││    • ◦◦◦◦  ││ │                                         
         │ ││       • │◦◦◦◦ │                                         
         │ •         ││  ││◦◦◦                                        
         │           •   •   │                                        
   80.30 │                   •                                        
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=88.380  sd=4.266  range=[80.300, 95.800]
  ◦ series 2: n=20  mean=87.725  sd=2.596  range=[83.500, 92.000]
```
95% CI coverage: 90.0% (18/20), RMSE: 3.986, MAE: 3.265, Pearson r: 0.428

```
resting_heart_rate (• observed, ◦ median)
   88.30 │    •                                                       
         │    ││                                                      
         │•   ││                                                      
         │ •  ││                                                      
         │  │ ││  •                                                   
         │◦◦• ││  ││                                                  
         │  ◦◦◦◦◦◦◦◦ ••                                               
         │   • │••  ◦◦◦◦◦◦◦◦ •                                        
         │     •    ││  │•• ◦◦                                        
   70.80 │          •   •                                             
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=76.160  sd=4.542  range=[70.800, 88.300]
  ◦ series 2: n=20  mean=75.625  sd=1.446  range=[73.200, 78.000]
```
95% CI coverage: 90.0% (18/20), RMSE: 3.837, MAE: 2.935, Pearson r: 0.630

```
prescription_renewals (• observed, ◦ median)
    1.00 │  •• •   • ••• ◦◦◦◦◦                                        
         │  │ │││  │││  ││││ │                                        
         │  │ │││  │││  ││││ │                                        
         │  │ │││  │││  ││││ │                                        
         │  │ │││  │││  ││││ │                                        
         │  │ │││  │││  ││││ │                                        
         │  │ │││  │││  ││││ │                                        
         │  │ │││  │││  ││││ │                                        
         │  │ │││  │││  ││││ │                                        
    0.90 │◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦ •  •                                        
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=0.950  sd=0.050  range=[0.900, 1.000]
  ◦ series 2: n=20  mean=0.925  sd=0.043  range=[0.900, 1.000]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.067, MAE: 0.045, Pearson r: 0.115

```
hdl_cholesterol (• observed, ◦ median)
    1.52 │                   •                                        
         │                  ◦◦                                        
         │               •◦◦•                                         
         │           • ◦◦◦                                            
         │           ◦◦                                               
         │        •◦◦                                                 
         │      ◦◦◦                                                   
         │   •◦◦•                                                     
         │•◦◦◦                                                        
    1.18 │◦                                                           
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=1.341  sd=0.088  range=[1.210, 1.520]
  ◦ series 2: n=20  mean=1.332  sd=0.092  range=[1.180, 1.480]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.022, MAE: 0.017, Pearson r: 0.977

```
appointment_attendance (• observed, ◦ median)
    0.99 │                •                                           
         │                ││ •                                        
         │                ││ │                                        
         │     •     •   • ◦◦◦                                        
         │     ││•   │◦◦◦◦◦•                                          
         │     ││││◦◦◦ •                                              
         │   •◦◦◦◦◦│•                                                 
         │ •◦◦││  •                                                   
         │◦◦││││                                                      
    0.85 │  • •                                                       
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=0.913  sd=0.037  range=[0.850, 0.990]
  ◦ series 2: n=20  mean=0.908  sd=0.024  range=[0.870, 0.950]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.024, MAE: 0.018, Pearson r: 0.791

```
crp (• observed, ◦ median)
    4.30 │◦                                                           
         │ ◦◦•                                                        
         │  •◦◦                                                       
         │     ◦◦                                                     
         │      •◦◦◦ •                                                
         │         •◦◦│                                               
         │            ◦◦                                              
         │              ◦◦                                            
         │                ◦◦ •                                        
    2.30 │                 •◦◦                                        
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=3.245  sd=0.585  range=[2.300, 4.300]
  ◦ series 2: n=20  mean=3.250  sd=0.577  range=[2.300, 4.200]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.132, MAE: 0.105, Pearson r: 0.974

```
fibrinogen (• observed, ◦ median)
    4.59 │•• •                                                        
         │◦◦◦ │•                                                      
         │   ◦◦◦│ •                                                   
         │      ◦◦◦•••• •                                             
         │         ◦◦◦ │││ •                                          
         │            ◦◦◦• ││•                                        
         │               ◦◦◦││                                        
         │                ││◦◦                                        
         │                ││                                          
    3.22 │                •                                           
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=4.114  sd=0.331  range=[3.220, 4.590]
  ◦ series 2: n=20  mean=4.025  sd=0.288  range=[3.550, 4.500]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.192, MAE: 0.152, Pearson r: 0.858

#### Test Statistics

| Variable | Statistic | Observed | p(rep ≥ obs) | Result |
| --- | --- | --- | --- | --- |
| ldl_cholesterol | mean | 2.900 | 0.680 | Pass |
| ldl_cholesterol | sd | 0.850 | 0.480 | Pass |
| systolic_bp | mean | 135.200 | 0.480 | Pass |
| systolic_bp | sd | 9.800 | 0.500 | Pass |
| triglycerides | mean | 1.900 | 0.600 | Pass |
| triglycerides | sd | 0.160 | 0.700 | Pass |
| hba1c | mean | 7.700 | 0.460 | Pass |
| hba1c | sd | 0.340 | 0.600 | Pass |
| fasting_glucose | mean | 7.200 | 0.400 | Pass |
| fasting_glucose | sd | 0.500 | 0.620 | Pass |
| diastolic_bp | mean | 88.400 | 0.540 | Pass |
| diastolic_bp | sd | 4.270 | 0.420 | Pass |
| resting_heart_rate | mean | 76.200 | 0.300 | Pass |
| resting_heart_rate | sd | 4.540 | 0.620 | Pass |
| prescription_renewals | mean | 0.900 | 0.680 | Pass |
| prescription_renewals | sd | 0.070 | 0.740 | Pass |
| hdl_cholesterol | mean | 1.330 | 0.480 | Pass |
| hdl_cholesterol | sd | 0.090 | 0.680 | Pass |
| appointment_attendance | mean | 0.930 | 0.560 | Pass |
| appointment_attendance | sd | 0.050 | 0.720 | Pass |
| crp | mean | 3.100 | 0.580 | Pass |
| crp | sd | 0.650 | 0.520 | Pass |
| fibrinogen | mean | 3.900 | 0.420 | Pass |
| fibrinogen | sd | 0.360 | 0.580 | Pass |

### LOO Cross-Validation

- **ELPD**: -128.4
- **p_loo**: 6.2
- **SE**: 11.3
- **Data points**: 90
- **Observation unit**: timestep
- **Bad Pareto k**: 0

#### LOO-PIT

```
LOO-PIT (should be uniform)
    0.33 │ ██████████████████████ 6
    0.37 │ ████████████████████████████████████ 10
    0.40 │ █████████████████████████████████ 9
    0.44 │ █████████████████████████ 7
    0.48 │ ████████████████████████████████████████ 11
    0.51 │ ████████████████████████████████████████ 11
    0.55 │ █████████████████████████████████ 9
    0.59 │ █████████████████████████████████ 9
    0.62 │ ████████████████████████████████████████ 11
    0.66 │ █████████████████████████ 7
  n=90  mean=0.500  sd=0.101  median=0.505  range=[0.310, 0.680]
```
Mean: 0.500 (ideal: 0.500), Std: 0.101 (ideal: 0.289), KS stat: 0.320, Calibration: Poor

#### Pareto k Diagnostics

- **k > 0.7 (fail)**: 0
- **0.5 < k ≤ 0.7 (warn)**: 0
- **k ≤ 0.5 (ok)**: 90

```
Pareto k distribution
    0.05 │ ████ 1
    0.06 │ ████████████████████████████████████ 9
    0.07 │  0
    0.07 │ ████████████████████████████████████████ 10
    0.08 │ ████████████████████████████████████████ 10
    0.09 │ ████████████████████████████████████████ 10
    0.09 │  0
    0.10 │ ████████████████████████████████████████ 10
    0.11 │  0
    0.11 │ ████████████████████████████████████████ 10
    0.12 │ ████████████████████████████████ 8
    0.13 │  0
    0.13 │ ████████████████████████████ 7
    0.14 │ ████████████████████████████████ 8
    0.15 │ ████████████████████████████ 7
  n=90  mean=0.101  sd=0.028  median=0.100  range=[0.050, 0.150]
```

### Power Scaling Diagnostics

| Parameter | Diagnosis | Prior Sens. | Likelihood Sens. | PSIS k-hat |
| --- | --- | --- | --- | --- |
| beta_lipid_cv | well_identified | 0.100 | 0.880 | 0.120 |
| beta_pressure_cv | well_identified | 0.140 | 0.830 | 0.180 |
| beta_glycemic_cv | well_identified | 0.190 | 0.760 | 0.240 |
| beta_lipid_inflammation | well_identified | 0.120 | 0.860 | 0.140 |
| beta_inflammation_cv | well_identified | 0.150 | 0.820 | 0.190 |
| rho_lipid | well_identified | 0.070 | 0.920 | 0.090 |

```
Power Scaling (prior vs likelihood sensitivity)
   0.92 │•                                                 
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │            •                                     
        │                                                  
        │                    •                             
        │                                                  
        │                                                  
        │                                                  
        │                             •                    
        │                                 •                
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
   0.76 │                                                 •
        └──────────────────────────────────────────────────
        0.07                                          0.19
  n=6  x: mean=0.128 sd=0.038  y: mean=0.845 sd=0.050
```

### Posterior Marginals

```
beta_lipid_cv  (mean=0.420, sd=0.095, HDI=[0.240, 0.610])
    6.40 │       •                                                    
         │      • •                                                   
         │      │  │                                                  
         │      │  •                                                  
         │     •    │                                                 
         │     │    │                                                 
         │    •     •                                                 
         │    │      │                                                
         │  ••       •                                                
    0.03 │••          •••                                             
         └────────────────────────────────────────────────────────────
  x: [0.050, 0.750]
  mean=0.400  sd=0.105  mode=0.400
```

```
beta_pressure_cv  (mean=0.350, sd=0.082, HDI=[0.190, 0.510])
    6.20 │      ••                                                    
         │      │ │                                                   
         │     •  │                                                   
         │     │  •                                                   
         │     │   │                                                  
         │    •    •                                                  
         │    │     │                                                 
         │   •      •                                                 
         │  •        •                                                
    0.02 │••          •••                                             
         └────────────────────────────────────────────────────────────
  x: [0.050, 0.610]
  mean=0.305  sd=0.082  mode=0.290
```

```
beta_glycemic_cv  (mean=-0.270, sd=0.078, HDI=[-0.420, -0.120])
    6.50 │       •                                                    
         │      • •                                                   
         │      │  │                                                  
         │     •   │                                                  
         │     │   •                                                  
         │     │    │                                                 
         │    •     │                                                 
         │    │     •                                                 
         │   •       •                                                
    0.02 │•••         •••                                             
         └────────────────────────────────────────────────────────────
  x: [-0.550, 0.010]
  mean=-0.276  sd=0.080  mode=-0.270
```

### Posterior Pairs

```
beta_lipid_cv vs beta_pressure_cv
   0.49 │                 •                                
        │                              •            •      
        │                               •                  
        │              •                                   
        │               •                                  
        │                                                  
        │                            •                     
        │•           •       •  •                         •
        │                                                  
        │                                                  
        │                       •     •    •               
        │                   •                    •         
        │                   •                              
        │                                                  
   0.21 │                                   •              
        └──────────────────────────────────────────────────
        0.23                                          0.64
  n=20  x: mean=0.436 sd=0.098  y: mean=0.360 sd=0.080
```
Pearson r: -0.107

```
beta_lipid_cv vs beta_glycemic_cv
  -0.14 │                                •                 
        │                                                  
        │                                                  
        │            •                 ••                  
        │       •               ••                         
        │•   •              •             •                
        │                      •                           
        │                    •     •           •           
        │                                                 •
        │                                •                 
        │                                      •           
        │                 •                                
        │                                                  
        │                                                  
  -0.47 │                               •                  
        └──────────────────────────────────────────────────
        0.21                                          0.70
  n=20  x: mean=0.456 sd=0.121  y: mean=-0.279 sd=0.076
```
Pearson r: -0.262

*Inference: map | 1000 samples | 42.5s*

---

## Stage 6: Treatment Effects
> Computes interventional treatment effects and ranks them by magnitude and certainty.

### Treatment Ranking

| Treatment | τ̂ | 95% CI | P(τ>0) | Identifiable | Status |
| --- | --- | --- | --- | --- | --- |
| lipid_burden | 0.420 | [0.234, 0.616] | 98.2% | Yes | ok |
| vascular_inflammation | 0.380 | [0.189, 0.557] | 99.1% | Yes | ok |
| arterial_pressure | 0.350 | [0.206, 0.504] | 96.8% | Yes | ok |
| medication_adherence | -0.330 | [-0.495, -0.162] | 2.1% | Yes | ok |
| glycemic_control | -0.270 | [-0.424, -0.131] | 3.5% | Yes | ok |

```
Posterior: lipid_burden
    0.16 │ █ 1
    0.20 │ ██ 2
    0.24 │ █████ 5
    0.28 │ ███████████ 10
    0.31 │ ████████████████ 15
    0.35 │ ████████████████ 15
    0.39 │ ████████████████████████████████████████ 37
    0.43 │ ██████████████████████████████████████ 35
    0.47 │ ████████████████████████████ 26
    0.50 │ ██████████████████████ 20
    0.54 │ ██████████████ 13
    0.58 │ ███████████ 10
    0.62 │ ██████ 6
    0.65 │ ██ 2
    0.69 │ ███ 3
  n=200  mean=0.429  sd=0.100  median=0.423  range=[0.145, 0.710]
```

```
Posterior: vascular_inflammation
    0.14 │ ████ 3
    0.17 │ ███ 2
    0.20 │ ███████████████ 11
    0.24 │ █████████████████ 13
    0.27 │ ████████████████ 12
    0.31 │ ███████████████████████ 17
    0.34 │ ███████████████████████████████ 23
    0.37 │ ████████████████████████████████████████ 30
    0.41 │ ████████████████████████████ 21
    0.44 │ ████████████████████████████████ 24
    0.48 │ █████████████████████████ 19
    0.51 │ ████████████████ 12
    0.54 │ ███████████ 8
    0.58 │ ████ 3
    0.61 │ ███ 2
  n=200  mean=0.378  sd=0.100  median=0.382  range=[0.119, 0.630]
```

```
Posterior: arterial_pressure
    0.15 │ ████ 3
    0.18 │ ███ 2
    0.21 │ ████ 3
    0.24 │ ███████████ 8
    0.27 │ ██████████████████████████ 18
    0.29 │ ████████████████████████████████████████ 28
    0.32 │ ███████████████████████████████ 22
    0.35 │ ███████████████████████████ 19
    0.38 │ ███████████████████████████████████████ 27
    0.41 │ ████████████████████████████████████ 25
    0.43 │ ███████████████████████████ 19
    0.46 │ ████████████████ 11
    0.49 │ ███████████ 8
    0.52 │ ██████ 4
    0.55 │ ████ 3
  n=200  mean=0.357  sd=0.081  median=0.360  range=[0.140, 0.559]
```

```
Posterior: medication_adherence
   -0.55 │ █████ 4
   -0.52 │ █ 1
   -0.49 │ ████ 3
   -0.46 │ ███████████████████████ 17
   -0.42 │ ████████████████████████ 18
   -0.39 │ ████████████████████████████████████ 27
   -0.36 │ ████████████████████████████████████████ 30
   -0.33 │ ████████████████████████ 18
   -0.30 │ ███████████████████████████████ 23
   -0.27 │ ███████████████████████████ 20
   -0.24 │ ████████████████████████████ 21
   -0.20 │ █████████████ 10
   -0.17 │ █████ 4
   -0.14 │ ████ 3
   -0.11 │ █ 1
  n=200  mean=-0.336  sd=0.089  median=-0.344  range=[-0.565, -0.093]
```

```
Posterior: glycemic_control
   -0.48 │ █ 1
   -0.45 │ ████ 3
   -0.42 │ ████████ 6
   -0.39 │ ████████ 6
   -0.35 │ ██████████████████████████ 20
   -0.32 │ ████████████████████████████████████████ 31
   -0.29 │ ███████████████████████████████ 24
   -0.26 │ ███████████████████████████████████████ 30
   -0.23 │ ████████████████████████████████████ 28
   -0.20 │ ██████████████████████████████████ 26
   -0.17 │ ███████████████████ 15
   -0.14 │ ████████ 6
   -0.11 │ ████ 3
   -0.08 │  0
   -0.04 │ █ 1
  n=200  mean=-0.270  sd=0.076  median=-0.267  range=[-0.493, -0.029]
```

### Manifest Effects

| Treatment | Indicator | Effect |
| --- | --- | --- |
| lipid_burden | ldl_cholesterol | 0.550 |
| lipid_burden | triglycerides | 0.280 |
| lipid_burden | resting_heart_rate | 0.080 |
| lipid_burden | systolic_bp | 0.050 |
| lipid_burden | hdl_cholesterol | -0.420 |
| lipid_burden | crp | 0.520 |
| lipid_burden | fibrinogen | 0.350 |
| vascular_inflammation | crp | 0.680 |
| vascular_inflammation | fibrinogen | 0.450 |
| vascular_inflammation | systolic_bp | 0.040 |
| vascular_inflammation | ldl_cholesterol | 0.020 |
| arterial_pressure | systolic_bp | 0.480 |
| arterial_pressure | diastolic_bp | 0.350 |
| arterial_pressure | resting_heart_rate | 0.120 |
| arterial_pressure | crp | 0.080 |
| arterial_pressure | fibrinogen | 0.050 |
| medication_adherence | prescription_renewals | -0.450 |
| medication_adherence | ldl_cholesterol | -0.150 |
| medication_adherence | systolic_bp | -0.120 |
| medication_adherence | hba1c | -0.100 |
| medication_adherence | hdl_cholesterol | 0.180 |
| medication_adherence | appointment_attendance | -0.520 |
| glycemic_control | hba1c | -0.380 |
| glycemic_control | fasting_glucose | -0.320 |
| glycemic_control | ldl_cholesterol | -0.050 |
| glycemic_control | resting_heart_rate | -0.030 |
| glycemic_control | crp | 0.120 |
| glycemic_control | fibrinogen | 0.080 |

### Temporal Effects

| Treatment | 1d | 7d | 30d | Peak | Time to Peak |
| --- | --- | --- | --- | --- | --- |
| lipid_burden | 0.004 | 0.065 | 0.339 | 0.483 | 60.0 days |
| vascular_inflammation | 0.021 | 0.182 | 0.348 | 0.420 | 21.0 days |
| arterial_pressure | 0.018 | 0.166 | 0.322 | 0.403 | 28.0 days |
| medication_adherence | -0.009 | -0.070 | -0.217 | -0.380 | 45.0 days |
| glycemic_control | -0.001 | -0.027 | -0.108 | -0.310 | 90.0 days |
