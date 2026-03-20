# Pipeline Stages Reference

The causal inference pipeline has 10 stages (0 through 6, with sub-stages 4b, 5a, 5b). Execution order is derived from a dependency DAG via topological sort — there is no manual index. Each stage declares its upstream dependencies; the pipeline runner folds over this graph.

## Stage 0 — Agentic Data Ingestion

Parses arbitrary user-uploaded files into a typed Polars DataFrame with column-level metadata.

| | |
|---|---|
| **Type** | Semantic (agentic, tool-using) |
| **Interactive** | No |
| **Gate** | No |

### Inputs

- `user_id` — identifies the user workspace; the stage discovers the latest uploaded file under `data/{user_id}/input/`

### Process

1. Locates and extracts the uploaded file (handles ZIP archives or raw files).
2. Launches an agentic LLM conversation with four tools:
   - `list_files(path)` — explore the input directory
   - `read_file_sample(path, n_lines)` — peek at file contents
   - `execute_python(code)` — run arbitrary Python in a sandboxed environment to parse files into a Polars DataFrame
   - `submit_table(source_label, column_descriptions_json)` — finalize the DataFrame with human-readable column metadata
3. If the DataFrame is produced but metadata is missing, re-runs a finalization prompt.
4. Validates that both the DataFrame and column descriptions are present.

### Output

| Field | Type | Description |
|---|---|---|
| `source_label` | `str` | Human-readable data source name |
| `n_records` | `int` | Row count |
| `n_columns` | `int` | Column count |
| `date_range` | `{start, end}` | Temporal extent (ISO 8601) |
| `sample` | `list[dict]` | First N rows |
| `column_descriptions` | `list[{name, dtype, description}]` | Per-column metadata |
| `llm_trace` | `LLMTrace?` | Tool call history |

---

## Stage 1a — Latent Model Proposal

Translates a natural-language research question into a theoretical causal DAG.

| | |
|---|---|
| **Type** | Semantic (single conversation, tool-validated) |
| **Interactive** | Yes |
| **Gate** | No |

### Inputs

- `question` — the user's research question in natural language (e.g. "Why do I feel tired on Mondays?")

### Process

1. Single LLM conversation with one tool:
   - `validate_latent_model(structure_json)` — validates the proposed DAG against schema constraints
2. On valid submission the LLM output is parsed into a latent model with:
   - **Constructs**: theoretical variables with name, description, role (endogenous/exogenous), temporal status, and outcome flag
   - **Edges**: causal links between constructs with justification and lag semantics
3. The LLM also identifies the primary outcome construct and candidate treatment variables.
4. A self-review follow-up prompt gives the LLM a chance to refine its proposal.

This stage is purely theoretical — it does not see any data.

### Output

| Field | Type | Description |
|---|---|---|
| `latent_model` | `LatentModel` | DAG with constructs and edges |
| `outcome_name` | `str` | Primary outcome variable |
| `treatments` | `list[str]` | Candidate treatment/intervention variables |
| `llm_trace` | `LLMTrace?` | Conversation trace |

Key data structures:

- **Construct**: `{name, description, role, is_outcome, temporal_status}`
- **CausalEdge**: `{cause, effect, description, lagged}` — `lagged=True` means the effect at time *t* depends on the cause at *t−1*

Unobserved confounding is modeled as explicit latent nodes in the DAG (never as bidirected ADMG edges — ADMGs are only used internally for the y0 identification algorithm).

---

## Stage 1b — Measurement Model + Identifiability

Maps latent constructs to observable indicators from the ingested dataset, then checks whether the target causal effects are identifiable.

| | |
|---|---|
| **Type** | Semantic (single conversation, tool-validated) |
| **Interactive** | Yes |
| **Gate** | Yes — filters non-identifiable treatments; halts if all are blocked by unobserved confounders |

### Inputs

- `question` — research question
- `stage0.result` — ingested DataFrame + column descriptions
- `stage1a.result` — latent model with constructs and edges

### Process

1. Formats the dataset schema (column names, dtypes, descriptions) as LLM context.
2. Single LLM conversation with one tool:
   - `validate_measurement_model(measurement_json)` — performs three checks in one call:
     1. **Schema validation** — indicator names, constructs, and extraction modes match the latent model
     2. **Compiler constraints** — observation models are compilable (valid dtypes, support kinds, aggregation functions)
     3. **Causal identifiability** — uses y0's ID algorithm to check whether each treatment→outcome effect is identified given the DAG structure and unobserved confounders
3. If identifiability fails, the tool returns rich feedback explaining which paths are blocked. The LLM then adds proxy indicators or adjusts the measurement model to unblock identification.
4. Assembles a `CausalSpec` combining the latent model, measurement model, and identifiability status.

### Output

| Field | Type | Description |
|---|---|---|
| `causal_spec` | `CausalSpec` | Combined latent + measurement + identifiability |
| `gate_overridden` | `GateOverrideContract?` | Present if gate was overridden |
| `llm_trace` | `LLMTrace?` | Conversation trace |

Key data structures:

- **Indicator**: `{name, construct_name, how_to_measure, measurement_dtype, aggregation, observation_window, ordinal_levels, source_columns, extraction_mode}`
  - `extraction_mode`: `"computed"` (direct Polars aggregation) or `"semantic"` (LLM worker)
  - `measurement_dtype`: `"continuous"`, `"binary"`, `"count"`, `"ordinal"`, or `"categorical"`
- **MeasurementModel**: `{indicators, model_clock}` — `model_clock` is the observation window width used for extraction and SSM discretization
- **IdentifiabilityStatus**: `{status, non_identifiable_treatments, marginalization_analysis}`

---

## Stage 2 — Indicator Extraction

Extracts numeric indicator values from the raw data using parallel LLM workers (semantic path) or direct Polars aggregation (computed path).

| | |
|---|---|
| **Type** | Semantic + computed (parallel LLM workers for semantic indicators, Polars aggregation for computed) |
| **Interactive** | No |
| **Gate** | No |

### Inputs

- `question` — provides temporal and semantic context for extraction
- `stage0.result` — raw DataFrame + column descriptions
- `stage1b.result` — CausalSpec with indicators and extraction modes
- `root_run_id` — Prefect run ID for worker progress events
- `max_windows` — cap on extraction support windows (limited for free tier)

### Process

1. **Groups indicators** by extraction mode and observation window.
2. **Computed path** (fast, no LLM): direct Polars aggregation on raw DataFrame columns for indicators with `extraction_mode="computed"`.
3. **Semantic path** (LLM workers):
   - Groups raw data into support windows (time intervals derived from `model_clock`)
   - Spawns a thread pool of concurrent LLM workers
   - Each worker processes a chunk of support windows, extracting scalar values from natural-language data within each window
   - Emits worker progress events to Prefect
4. **Merges results** into canonical observation rows: `{indicator, value, anchor_time, support_start, support_end}`.
5. **Encodes non-continuous types**: ordinal → numeric codes, binary → 0/1.
6. **Sorts** by indicator then anchor time.

### Output

| Field | Type | Description |
|---|---|---|
| `workers` | `list[WorkerStatus]` | Per-worker status (id, status, n_extractions, n_windows, error) |
| `combined_extractions_sample` | `list[{indicator, value, anchor_time}]` | First 20 rows |
| `per_indicator_counts` | `dict[str, int]` | Extraction count per indicator |
| `llm_trace` | `LLMTrace?` | Sampled from one worker |

---

## Stage 3 — Extraction Validation

Audits the extracted data with composable validation rules and computes empirical profiles per indicator.

| | |
|---|---|
| **Type** | Computed |
| **Interactive** | No |
| **Gate** | No |

### Inputs

- `stage1b.result` — CausalSpec with indicator metadata and causal structure
- `stage2.result` — raw and model-ready DataFrames

### Process

1. **Validation rules** (composable checks, each producing findings):
   - **Structural**: minimum observations, time coverage, gaps, IQR-based outliers
   - **Data quality**: dtype violations, duplicates, arithmetic sequences
   - **Temporal**: unparseable timestamps, time-variance
   - **Distribution**: zero fraction, non-negativity, unit-interval checks
   - **Alignment**: CFA alignment for ordinal categories
2. **Per-indicator empirical profiles**: mean, std, min, max, quantiles, variance, time coverage ratio, max gap ratio, and additional quality metrics.
3. **Status derivation**:
   - Cell-level (per metric per indicator) → `ok | warning | error`
   - Indicator-level → aggregated from all cells
   - Dataset-level → merged from all indicators + dataset-wide issues

### Output

| Field | Type | Description |
|---|---|---|
| `is_valid` | `bool` | `True` if no errors (warnings are acceptable) |
| `indicators` | `dict[str, IndicatorAudit]` | Per-indicator profile + validation |
| `dataset_issues` | `list[ValidationIssue]` | Dataset-level issues |

Key data structures:

- **IndicatorAudit**: `{profile, validation}`
  - `profile`: empirical statistics (n_obs, mean, std, min, max, q25/q50/q75, variance, time_coverage_ratio, max_gap_ratio, dtype_violations, duplicate_pct, zero_fraction, etc.)
  - `validation`: `{issues: [...], checks: {check_name: "ok"|"warning"|"error"}}`

---

## Stage 4 — Model Specification + Prior Elicitation

An agentic LLM conversation that specifies the statistical model (likelihoods, parameters, constraints) and elicits informative priors grounded in data profiles and optionally literature.

| | |
|---|---|
| **Type** | Semantic (multi-turn agentic conversation) |
| **Interactive** | Yes |
| **Gate** | No |

### Inputs

- `question` — research question
- `stage1b.result` — CausalSpec with latent + measurement models
- `stage2.result` — model-ready observation data
- `stage3.result` — indicator audits (empirical profiles + validation findings)
- `enable_literature` — whether to allow literature search

### Process

1. **Builds decision cards** from the CausalSpec and empirical profiles:
   - Model topology decisions (DAG → Kalman state + emission structure)
   - Distribution choices for ambiguous indicators
   - Loading parameters (fixed vs free)
   - Construct scale anchoring (intercept, coefficients)
   - Prior cards for parameters needing prior specification
2. **Multi-turn LLM conversation** with two tools:
   - `search_literature(query, parameter_name)` — search for empirical effect sizes in the literature (when enabled)
   - `validate_model(model_json)` — validates and compiles the model spec + priors, runs prior predictive simulation, and returns feedback
3. The validation tool performs:
   - Schema validation of `ModelSpec` and `PriorProposal` structures
   - Trial compilation (translates spec → SSM, compiles priors, binds parameters)
   - Prior predictive checks: forward-simulates from priors and compares against empirical data ranges

### Output

| Field | Type | Description |
|---|---|---|
| `model_spec` | `ModelSpec` | Full specification (parameters + likelihoods) |
| `priors` | `dict[str, PriorProposal]` | Prior distribution per parameter |
| `search_queries` | `dict[str, str]?` | Literature search queries used |
| `prior_predictive_samples` | `dict[str, list[float]]?` | Forward-simulated samples |
| `llm_trace` | `LLMTrace?` | Conversation trace |

Key data structures:

- **ModelSpec**: `{parameters: list[ParameterSpec], likelihoods: list[LikelihoodSpec]}`
  - **ParameterSpec**: `{name, role, constraint, description}`
  - **LikelihoodSpec**: `{variable, distribution, link, reasoning, sources}`
- **PriorProposal**: `{parameter, distribution, params, sources, reasoning, reference_interval_days, density_points}`
  - `reference_interval_days` enables discrete-time → continuous-time conversion of effect sizes from literature

---

## Stage 4b — Parametric Identifiability Diagnostics

Pre-fit checks for whether the model parameters can be uniquely recovered from the data.

| | |
|---|---|
| **Type** | Computed |
| **Interactive** | No |
| **Gate** | Yes — halts if T-rule violated (free parameters exceed moment conditions) |

### Inputs

- `stage4.result` — model spec + priors (compiled SSM)
- `stage2.result` — model-ready data

### Process

1. **T-rule check** (necessary condition, fast):
   - Counts free parameters vs available moment conditions (`2 × T × n_manifest`)
   - If free parameters exceed moment conditions → provably non-identified → hard gate failure
2. **Sensitivity analysis** (structural identifiability):
   - Computes Jacobian of state → observations (output sensitivity)
   - Detects flat profile likelihood via singular values near zero
   - Reports condition number
3. **Profile likelihood** (practical identifiability):
   - Grid-scans each parameter while optimizing others
   - Chi-squared threshold (1.92 for 95% CI)
   - Per-parameter diagnosis: `structural_issue`, `boundary_issue`, or `well_identified`

### Output

| Field | Type | Description |
|---|---|---|
| `parametric_id` | `ParametricIdResult` | T-rule + sensitivity + profile likelihood results |
| `inference_structure` | `InferenceStructureResult?` | Active likelihood path, auto routing, and first-pass Rao-Blackwellization plan |
| `gate_overridden` | `GateOverrideContract?` | Present if gate was overridden |

Key data structures:

- **ParametricIdResult**: `{checked, t_rule, summary, error}`
  - `t_rule`: `{satisfies, n_free_params, n_moments}`
  - `summary`: `{structural_issues, boundary_issues, weak_params}`

---

## Stage 5a — SVI Preflight

A fast approximate fit using Stochastic Variational Inference for early sanity-checking before committing to expensive inference.

| | |
|---|---|
| **Type** | Computed |
| **Interactive** | No |
| **Gate** | No |

### Inputs

- `stage4.result` — model spec + priors
- `stage2.result` — model-ready data

### Process

1. Runs SVI with fixed configuration: 5000 optimization steps, 500 posterior samples.
2. Produces ELBO convergence curve and approximate posterior marginals.
3. Best-effort only — does not block the pipeline on failure.

Always recomputed on resume (never restored from checkpoint).

### Output

| Field | Type | Description |
|---|---|---|
| `inference_metadata` | `{method, n_samples, duration_seconds}` | Always `method="svi"` |
| `svi_diagnostics` | `SVIDiagnostics?` | ELBO curve, convergence metrics |
| `posterior_marginals` | `list[PosteriorMarginal]?` | Approximate marginal distributions |
| `posterior_pairs` | `list[PosteriorPair]?` | Pairwise posterior scatter plots |

---

## Stage 5b — Inference + Diagnostics

Full Bayesian inference with post-fit diagnostics: power-scaling sensitivity analysis and posterior predictive checks.

| | |
|---|---|
| **Type** | Computed |
| **Interactive** | No |
| **Gate** | No |

### Inputs

- `stage4.result` — model spec + priors
- `stage2.result` — model-ready data
- `inference_method` — optional override for the sampler (defaults to config routing)

### Process

1. **Model fitting** via the configured inference method. Supported backends:
   - SVI, NUTS, NUTS-DA, Hessian-MC², PGAS, Tempered SMC, Laplace-EM, Structured VI, DPF
   - Method selection can be automatic (config-based routing) or manually overridden
2. **Power-scaling sensitivity analysis** (post-fit):
   - Perturbs prior and likelihood contributions, measures posterior shift
   - Per-parameter diagnosis: `prior_dominated`, `well_identified`, or `prior_data_conflict`
3. **Posterior predictive checks** (PPC):
   - Forward-simulates from posterior samples
   - Per-variable checks: calibration, autocorrelation, variance
   - Returns warnings for miscalibrated variables

### Output

| Field | Type | Description |
|---|---|---|
| `power_scaling` | `list[PowerScalingResult]` | Per-parameter sensitivity diagnosis |
| `ppc` | `PPCResult` | PPC warnings, overlays, test statistics |
| `inference_metadata` | `{method, n_samples, duration_seconds}` | Sampler used and runtime |
| `mcmc_diagnostics` | `MCMCDiagnostics?` | Rhat, ESS, divergences (NUTS/NUTS-DA) |
| `svi_diagnostics` | `SVIDiagnostics?` | ELBO curve (SVI) |
| `smc_diagnostics` | `SMCDiagnostics?` | Log evidence, effective sample size (SMC) |
| `loo_diagnostics` | `LOODiagnostics?` | Pareto k diagnostics |
| `posterior_marginals` | `list[PosteriorMarginal]?` | Marginal distributions |
| `posterior_pairs` | `list[PosteriorPair]?` | Pairwise scatter plots |

---

## Stage 6 — Intervention Analysis

Applies do-operator interventions to the fitted model and ranks treatments by estimated causal effect size.

| | |
|---|---|
| **Type** | Computed |
| **Interactive** | No |
| **Gate** | No |

### Inputs

- `stage5b.result` — fitted model artifact
- `stage1a.result` — outcome name + treatments list
- `stage1b.result` — CausalSpec (identifiability status)
- `stage1b_gate.result` — filtered treatment list (non-identifiable removed)

### Process

1. For each identifiable treatment:
   - Applies `do(treatment = baseline + 1σ)` intervention
   - Computes counterfactual outcome at steady state
   - Effect size = E[outcome | do(treatment)] − E[outcome | no intervention]
2. Per posterior sample: repeats intervention to get posterior draws of the effect size, computes `prob_positive = P(effect > 0)`.
3. Annotates each treatment with:
   - PPC warnings from stage 5b (if the outcome variable has calibration issues)
   - Prior sensitivity warnings (if the parameter driving the effect is prior-dominated)
   - Temporal effects (time-varying effect trajectory, if applicable)
   - Manifest effects (per-indicator effect decomposition)
4. Ranks treatments by `|effect_size|` descending.

### Output

| Field | Type | Description |
|---|---|---|
| `intervention_results` | `list[TreatmentEffect]` | Ranked treatment effects |

Key data structures:

- **TreatmentEffect**: `{treatment, effect_size, posterior_draws, prob_positive, identifiable, ppc_warnings, prior_sensitivity_warning, temporal, manifest_effects}`
  - `prob_positive`: P(effect > 0) across posterior draws
  - `manifest_effects`: effect decomposed by indicator
  - `temporal`: time-varying effect trajectory (if applicable)

---

## Cross-Cutting Concerns

### Resume and Replay

- **Resume from checkpoint**: specify `start_stage` + `end_stage` to re-run only that window. Earlier stages are restored from persisted snapshots and parquet/pickle artifacts.
- **Stage overrides**: stages 1a, 1b, and 4 are interactive. Submit a custom payload and the pipeline skips computation, resuming downstream with the override.
- Stage 5a is always recomputed on resume (never restored from checkpoint).

### Persisted Artifacts

| Stage | Artifact | Format |
|---|---|---|
| 0 | Raw ingested data | Parquet |
| 2 | Raw observation rows | Parquet |
| 2 | Model-ready encoded data | Parquet |
| 5b | Fitted model + diagnostics | Pickle |

All stage outputs are also serialized to JSON for the web layer. Internal fields (prefixed with `_`) are stripped from the web payload.
