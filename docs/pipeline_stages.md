# Pipeline Stages Reference

The causal inference pipeline has 10 stages (0 through 6, with sub-stages 4b, 5a, 5b). Execution order is derived from a dependency DAG via topological sort; there is no manual index. Each stage declares its upstream dependencies, and the pipeline runner folds over this graph.

Every stage below uses the same frame: at a glance, inputs, process, outputs, and key structures when the payload needs extra shape detail.

This document is intentionally stage-ordered. For the cross-cutting view of the pipeline, including artifact lineage, temporal semantics, assurance surfaces, and persistence/runtime semantics, see [architecture/pipeline_dimensions.md](architecture/pipeline_dimensions.md).

## Stage Matrix

This matrix is the quick cross-stage summary. The sections below remain the canonical per-stage reference.

| Stage | Primary artifact | Modality | Interactive | Gate semantics | Runtime note |
|---|---|---|---|---|---|
| 0 | Typed ingested dataframe | Semantic | No | None | Persists raw dataframe parquet |
| 1a | `LatentModel` | Semantic | Yes | None | Replay override eligible |
| 1b | `CausalSpec` | Semantic | Yes | Hard gate | Replay override eligible |
| 2 | Observation rows | Hybrid | No | None | Persists raw + model-ready parquet |
| 3 | Indicator audits | Computed | No | None | Standard restore |
| 4 | `ModelSpec` + priors | Semantic | Yes | None | Replay override eligible; restores compiled runtime state |
| 4b | Parametric identifiability payload | Computed | No | Warning-only | Restores compiled analysis state |
| 5a | SVI preflight | Computed | No | None | Always recomputed on resume |
| 5b | Fitted artifact + diagnostics | Computed | No | None | Persists fitted pickle artifact |
| 6 | Intervention ranking + follow-up trace | Hybrid | Yes | None | Terminal in-place persistence; no downstream replay |

## Stage 0 - Agentic Data Ingestion

Parses arbitrary user-uploaded files into a typed Polars DataFrame with column-level metadata.

### At a Glance

| Property | Value |
|---|---|
| Type | Semantic (agentic, tool-using) |
| Interactive | No |
| Gate | No |

### Inputs

| Input | Source | Description |
|---|---|---|
| `workspace_id` | Pipeline request | Identifies the workspace; the stage discovers the latest uploaded file under `data/{workspace_id}/input/`. |

### Process

1. Locates and extracts the uploaded file (handles ZIP archives or raw files).
2. Launches an agentic LLM conversation with four tools:
   - `list_files(path)` - explore the input directory
   - `read_file_sample(path, n_lines)` - peek at file contents
   - `execute_python(code)` - run arbitrary Python in a sandboxed environment to parse files into a Polars DataFrame
   - `submit_table(source_label, column_descriptions_json)` - finalize the DataFrame with human-readable column metadata
3. If the DataFrame is produced but metadata is missing, re-runs a finalization prompt.
4. Validates that both the DataFrame and column descriptions are present.

### Outputs

| Output | Type | Description |
|---|---|---|
| `source_label` | `str` | Human-readable data source name |
| `n_records` | `int` | Row count |
| `n_columns` | `int` | Column count |
| `date_range` | `{start, end}` | Temporal extent (ISO 8601) |
| `sample` | `list[dict]` | First N rows |
| `column_descriptions` | `list[{name, dtype, description}]` | Per-column metadata |
| `llm_trace` | `LLMTrace?` | Tool call history |

---

## Stage 1a - Latent Model Proposal

Translates a natural-language research question into a theoretical causal DAG.

### At a Glance

| Property | Value |
|---|---|
| Type | Semantic (single conversation, tool-validated) |
| Interactive | Yes |
| Gate | No |

### Inputs

| Input | Source | Description |
|---|---|---|
| `question` | Pipeline request | User's research question in natural language (for example, "Why do I feel tired on Mondays?") |

### Process

1. Runs a single LLM conversation with one tool:
   - `validate_latent_model(structure_json)` - validates the proposed DAG against schema constraints
2. On valid submission the LLM output is parsed into a latent model with:
   - Constructs: theoretical variables with name, description, role (endogenous/exogenous), temporal status, and outcome flag
   - Edges: causal links between constructs with justification and lag semantics
3. The LLM also identifies the primary outcome construct and candidate treatment variables.
4. A self-review follow-up prompt gives the LLM a chance to refine its proposal.

This stage is purely theoretical; it does not see any data.

### Outputs

| Output | Type | Description |
|---|---|---|
| `latent_model` | `LatentModel` | DAG with constructs and edges |
| `outcome_name` | `str` | Primary outcome variable |
| `treatments` | `list[str]` | Candidate treatment/intervention variables |
| `llm_trace` | `LLMTrace?` | Conversation trace |

### Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `Construct` | `{name, description, role, is_outcome, temporal_status}` | Theoretical variable in the latent model |
| `CausalEdge` | `{cause, effect, description, lagged}` | `lagged=True` means the effect at time `t` depends on the cause at `t-1` |

Unobserved confounding is modeled as explicit latent nodes in the DAG (never as bidirected ADMG edges). ADMGs are only used internally for the y0 identification algorithm.

---

## Stage 1b - Measurement Model + Identifiability

Maps latent constructs to observable indicators from the ingested dataset, then checks whether the target causal effects are identifiable.

### At a Glance

| Property | Value |
|---|---|
| Type | Semantic (single conversation, tool-validated) |
| Interactive | Yes |
| Gate | Yes; filters non-identifiable treatments and halts if all are blocked by unobserved confounders |

### Inputs

| Input | Source | Description |
|---|---|---|
| `question` | Pipeline request | Research question used to ground measurement choices |
| `stage0.result` | Stage 0 | Ingested DataFrame plus column descriptions |
| `stage1a.result` | Stage 1a | Latent model with constructs and edges |

### Process

1. Formats the dataset schema (column names, dtypes, descriptions) as LLM context.
2. Runs a single LLM conversation with one tool:
   - `validate_measurement_model(measurement_json)` - performs three checks in one call:
     1. Schema validation: indicator names, constructs, and extraction modes match the latent model
     2. Compiler constraints: observation models are compilable (valid dtypes, support kinds, aggregation functions)
     3. Causal identifiability: uses y0's ID algorithm to check whether each treatment -> outcome effect is identified given the DAG structure and unobserved confounders
3. If identifiability fails, the tool returns rich feedback explaining which paths are blocked. The LLM then adds proxy indicators or adjusts the measurement model to unblock identification.
4. Assembles a `CausalSpec` combining the latent model, measurement model, and identifiability status.

### Outputs

| Output | Type | Description |
|---|---|---|
| `causal_spec` | `CausalSpec` | Combined latent, measurement, and identifiability payload |
| `gate_overridden` | `GateOverrideContract?` | Present if the gate was overridden |
| `llm_trace` | `LLMTrace?` | Conversation trace |

### Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `Indicator` | `{name, construct_name, how_to_measure, measurement_dtype, aggregation, observation_window, ordinal_levels, source_columns, extraction_mode}` | Observed signal mapped to a construct; `extraction_mode` is `"computed"` (direct Polars aggregation) or `"semantic"` (LLM worker) |
| `MeasurementModel` | `{indicators, model_clock}` | `model_clock` is the observation-window width used for extraction and SSM discretization |
| `IdentifiabilityStatus` | `{status, non_identifiable_treatments, marginalization_analysis}` | Summarizes which treatment -> outcome effects are identified |

---

## Stage 2 - Indicator Extraction

Extracts numeric indicator values from the raw data using parallel LLM workers (semantic path) or direct Polars aggregation (computed path).

### At a Glance

| Property | Value |
|---|---|
| Type | Semantic + computed (parallel LLM workers for semantic indicators, Polars aggregation for computed indicators) |
| Interactive | No |
| Gate | No |

### Inputs

| Input | Source | Description |
|---|---|---|
| `question` | Pipeline request | Provides temporal and semantic context for extraction |
| `stage0.result` | Stage 0 | Raw DataFrame plus column descriptions |
| `stage1b.result` | Stage 1b | `CausalSpec` with indicators and extraction modes |
| `root_run_id` | Orchestrator runtime | Prefect run ID for worker progress events |
| `max_windows` | Pipeline config | Cap on extraction support windows (limited for free tier) |

### Process

1. Groups indicators by extraction mode and observation window.
2. Computed path (fast, no LLM): direct Polars aggregation on raw DataFrame columns for indicators with `extraction_mode="computed"`.
3. Semantic path (LLM workers):
   - Groups raw data into support windows (time intervals derived from `model_clock`)
   - Spawns a thread pool of concurrent LLM workers
   - Each worker processes a chunk of support windows, extracting scalar values from natural-language data within each window
   - Emits worker progress events to Prefect
4. Merges results into canonical observation rows: `{indicator, value, anchor_time, support_start, support_end}`.
5. Encodes non-continuous types: ordinal -> numeric codes, binary -> 0/1.
6. Sorts by indicator then anchor time.

### Outputs

| Output | Type | Description |
|---|---|---|
| `workers` | `list[WorkerStatus]` | Per-worker status (`worker_id`, `status`, `n_extractions`, `n_windows`, `error`) |
| `combined_extractions_sample` | `list[{indicator, value, anchor_time}]` | First 20 rows |
| `per_indicator_counts` | `dict[str, int]` | Extraction count per indicator |
| `llm_trace` | `LLMTrace?` | Sampled from one worker |

### Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `WorkerStatus` | `{worker_id, status, n_extractions, n_windows, error}` | Runtime status for each semantic worker |
| Observation row | `{indicator, value, anchor_time, support_start, support_end}` | Canonical row shape after computed and semantic paths are merged |

---

## Stage 3 - Extraction Validation

Audits the extracted data with composable validation rules and computes empirical profiles per indicator.

### At a Glance

| Property | Value |
|---|---|
| Type | Computed |
| Interactive | No |
| Gate | No |

### Inputs

| Input | Source | Description |
|---|---|---|
| `stage1b.result` | Stage 1b | `CausalSpec` with indicator metadata and causal structure |
| `stage2.result` | Stage 2 | Raw and model-ready DataFrames |

### Process

1. Runs validation rules, each producing findings:
   - Structural: minimum observations, time coverage, gaps, IQR-based outliers
   - Data quality: dtype violations, duplicates, arithmetic sequences
   - Temporal: unparseable timestamps, time-variance
   - Distribution: zero fraction, non-negativity, unit-interval checks
   - Alignment: CFA alignment for ordinal categories
2. Computes per-indicator empirical profiles: mean, std, min, max, quantiles, variance, time coverage ratio, max gap ratio, and additional quality metrics.
3. Derives status at three levels:
   - Cell-level (per metric per indicator) -> `ok | warning | error`
   - Indicator-level -> aggregated from all cells
   - Dataset-level -> merged from all indicators plus dataset-wide issues

### Outputs

| Output | Type | Description |
|---|---|---|
| `is_valid` | `bool` | `True` if no errors are present; warnings are acceptable |
| `indicators` | `dict[str, IndicatorAudit]` | Per-indicator profile and validation results |
| `dataset_issues` | `list[ValidationIssue]` | Dataset-level issues |

### Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `IndicatorAudit` | `{profile, validation}` | Bundles empirical statistics with validation findings |
| `profile` | `n_obs, mean, std, min, max, q25/q50/q75, variance, time_coverage_ratio, max_gap_ratio, dtype_violations, duplicate_pct, zero_fraction, ...` | Empirical profile fields per indicator |
| `validation` | `{issues: [...], checks: {check_name: status}}` | Validation output attached to each indicator; each `status` is `ok`, `warning`, or `error` |

---

## Stage 4 - Model Specification + Prior Elicitation

An agentic LLM conversation that specifies the statistical model (likelihoods, parameters, constraints) and elicits informative priors grounded in data profiles and optionally literature.

### At a Glance

| Property | Value |
|---|---|
| Type | Semantic (multi-turn agentic conversation) |
| Interactive | Yes |
| Gate | No |

### Inputs

| Input | Source | Description |
|---|---|---|
| `question` | Pipeline request | Research question |
| `stage1b.result` | Stage 1b | `CausalSpec` with latent and measurement models |
| `stage2.result` | Stage 2 | Model-ready observation data |
| `stage3.result` | Stage 3 | Indicator audits (empirical profiles plus validation findings) |
| `enable_literature` | Pipeline config | Whether to allow literature search |

### Process

1. Builds decision cards from the `CausalSpec` and empirical profiles:
   - Model topology decisions (DAG -> Kalman state and emission structure)
   - Distribution choices for ambiguous indicators
   - Loading parameters (fixed vs free)
   - Construct scale anchoring (intercept, coefficients)
   - Prior cards for parameters needing prior specification
2. Runs a multi-turn LLM conversation with tools:
   - `validate_model(model_json)` - validates and compiles the model spec plus priors, runs prior predictive simulation, and returns feedback
   - `search_literature(query, parameter_name)` - searches for empirical effect sizes in the literature when enabled
   - `elicit_prior_gmm(...)` - optional paraphrased prior elicitation with GMM aggregation when Stage 4 paraphrasing is enabled
3. The validation tool performs:
   - Schema validation of `ModelSpec` and `PriorProposal` structures
   - Trial compilation (translates spec -> SSM, compiles priors, binds parameters)
   - Prior predictive checks: forward-simulates from priors and compares against empirical data ranges

### Outputs

| Output | Type | Description |
|---|---|---|
| `model_spec` | `ModelSpec` | Full specification (parameters plus likelihoods) |
| `priors` | `dict[str, PriorProposal]` | Prior distribution per parameter |
| `search_queries` | `dict[str, str]?` | Literature search queries used |
| `prior_predictive_samples` | `dict[str, list[float]]?` | Forward-simulated samples |
| `llm_trace` | `LLMTrace?` | Conversation trace |

### Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `ModelSpec` | `{parameters: list[ParameterSpec], likelihoods: list[LikelihoodSpec]}` | Top-level Stage 4 output |
| `ParameterSpec` | `{name, role, constraint, description}` | Parameter definition in the compiled model |
| `LikelihoodSpec` | `{variable, distribution, link, reasoning, sources}` | Observation model choice per variable |
| `PriorProposal` | `{parameter, distribution, params, sources, reasoning, reference_interval_days, density_points}` | `reference_interval_days` supports discrete-time -> continuous-time conversion of effect sizes from literature |

---

## Stage 4b - Parametric Identifiability Diagnostics

Pre-fit checks for whether the model parameters can be uniquely recovered from the data.

### At a Glance

| Property | Value |
|---|---|
| Type | Computed |
| Interactive | No |
| Gate | No hard gate; a failed T-rule screen marks the stage as warning-only |

### Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | Stage 4 | Model spec and priors, including the compiled SSM |
| `stage2.result` | Stage 2 | Model-ready data |

### Process

1. Runs the T-rule check (necessary condition, fast):
   - Counts free parameters vs a conservative lower bound on available moment conditions
   - If free parameters exceed that lower bound, Stage 4b emits a warning about likely overparameterization and still allows downstream inference
2. Runs sensitivity analysis (structural identifiability):
   - Computes the Jacobian of state -> observations (output sensitivity)
   - Detects flat profile likelihood via singular values near zero
   - Reports condition number
3. Runs profile likelihood analysis (practical identifiability):
   - Grid-scans each parameter while optimizing others
   - Uses a chi-squared threshold (`1.92` for a 95% CI)
   - Produces per-parameter diagnoses: `structural_issue`, `boundary_issue`, or `well_identified`

### Outputs

| Output | Type | Description |
|---|---|---|
| `parametric_id` | `ParametricIdResult` | T-rule, sensitivity, and profile-likelihood results |
| `inference_structure` | `InferenceStructureResult?` | Active likelihood path, auto routing, and first-pass Rao-Blackwellization plan |
| `gate_overridden` | `GateOverrideContract?` | Present if the gate was overridden |

### Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `ParametricIdResult` | `{checked, t_rule, sensitivity_analysis?, summary, per_param_classification?, threshold?, error}` | Combined parametric-identifiability payload |
| `t_rule` | `{satisfies, n_free_params, n_moments}` | Necessary-condition check |
| `summary` | `{structural_issues, boundary_issues, weak_params}` | High-level diagnosis from sensitivity and profile-likelihood checks |

---

## Stage 5a - SVI Preflight

A fast approximate fit using Stochastic Variational Inference for early sanity-checking before committing to expensive inference.

### At a Glance

| Property | Value |
|---|---|
| Type | Computed |
| Interactive | No |
| Gate | No |

### Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | Stage 4 | Model spec and priors |
| `stage2.result` | Stage 2 | Model-ready data |

### Process

1. Runs SVI with fixed configuration: 5000 optimization steps and 500 posterior samples.
2. Produces an ELBO convergence curve and approximate posterior marginals.
3. Treats the pass as best-effort only; pipeline execution does not block on failure.

Always recomputed on resume (never restored from checkpoint).

### Outputs

| Output | Type | Description |
|---|---|---|
| `inference_metadata` | `{method, n_samples, duration_seconds}` | Web-facing summary metadata; current implementation always reports `method="svi"`, `n_samples=500`, and placeholder `duration_seconds=0.0` |
| `svi_diagnostics` | `SVIDiagnostics?` | ELBO curve and convergence metrics |
| `posterior_marginals` | `list[PosteriorMarginal]?` | Approximate marginal distributions |
| `posterior_pairs` | `list[PosteriorPair]?` | Pairwise posterior scatter plots |

---

## Stage 5b - Inference + Diagnostics

Full Bayesian inference with post-fit diagnostics: power-scaling sensitivity analysis and posterior predictive checks.

### At a Glance

| Property | Value |
|---|---|
| Type | Computed |
| Interactive | No |
| Gate | No |

### Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | Stage 4 | Model spec and priors |
| `stage2.result` | Stage 2 | Model-ready data |
| `inference_method` | Pipeline config | Optional override for the sampler; defaults to config routing |

### Process

1. Fits the model via the configured inference method. Supported backends:
   - SVI, NUTS, NUTS-DA, Hessian-MC2, PGAS, Tempered SMC, Laplace-EM, Structured VI, DPF
   - Method selection can be automatic (config-based routing) or manually overridden
2. Runs power-scaling sensitivity analysis after fitting:
   - Perturbs prior and likelihood contributions, measures posterior shift
   - Produces per-parameter diagnoses: `prior_dominated`, `well_identified`, or `prior_data_conflict`
3. Runs posterior predictive checks (PPC):
   - Forward-simulates from posterior samples
   - Performs per-variable checks for calibration, autocorrelation, and variance
   - Returns warnings for miscalibrated variables

### Outputs

| Output | Type | Description |
|---|---|---|
| `power_scaling` | `list[PowerScalingResult]` | Per-parameter sensitivity diagnosis |
| `ppc` | `PPCResult` | PPC warnings, overlays, and test statistics |
| `inference_metadata` | `{method, n_samples, duration_seconds}` | Web-facing summary metadata; current implementation reports the inferred method plus placeholder `n_samples=10000` and `duration_seconds=0.0` |
| `mcmc_diagnostics` | `MCMCDiagnostics?` | Rhat, ESS, and divergences for NUTS or NUTS-DA |
| `svi_diagnostics` | `SVIDiagnostics?` | ELBO curve for SVI |
| `smc_diagnostics` | `SMCDiagnostics?` | Log evidence and effective sample size for SMC |
| `loo_diagnostics` | `LOODiagnostics?` | Pareto-k diagnostics |
| `posterior_marginals` | `list[PosteriorMarginal]?` | Marginal distributions |
| `posterior_pairs` | `list[PosteriorPair]?` | Pairwise posterior scatter plots |

---

## Stage 6 - Intervention Analysis

Applies do-operator interventions to the fitted model and ranks treatments by estimated causal effect size.

### At a Glance

| Property | Value |
|---|---|
| Type | Computed baseline ranking + interactive terminal follow-up |
| Interactive | Yes |
| Gate | No |

### Inputs

| Input | Source | Description |
|---|---|---|
| `stage5b.result` | Stage 5b | Fitted model artifact |
| `stage1a.result` | Stage 1a | Outcome name and treatment list |
| `stage1b.result` | Stage 1b | `CausalSpec`, including identifiability status |
| `stage1b_gate.result` | Stage 1b gate | Filtered treatment list with non-identifiable treatments removed |
| `question` | Pipeline request | Optional question used when generating the opening Stage 6 commentary |

### Process

1. Computes the canonical Stage 6 baseline ranking:
   - For each identifiable treatment, applies a default steady-state intervention of `do(treatment = baseline + 1)` in latent units
   - Computes the counterfactual outcome at steady state
   - Defines effect size as `E[outcome | do(treatment)] - E[outcome | no intervention]`
2. Repeats the intervention per posterior sample to get posterior draws of the effect size, then computes `prob_positive = P(effect > 0)`.
3. Annotates each treatment with:
   - PPC warnings from Stage 5b if the outcome variable has calibration issues
   - Prior sensitivity warnings if the parameter driving the effect is prior-dominated
   - Temporal effects (time-varying effect trajectory, if applicable)
   - Manifest effects (per-indicator effect decomposition)
4. Ranks treatments by `|effect_size|` in descending order.
5. Generates the initial Stage 6 interpretation and persists it as both:
   - `final_summary` for direct rendering in the web payload
   - `llm_trace` for full conversational audit/history
6. Exposes a narrow read-only interactive surface for follow-up analysis:
   - `get_model_info` for model, measurement, identifiability, and diagnostics summaries
   - `simulate_intervention` for Pearl rung-2 intervention queries
   - `simulate_counterfactual` for Pearl rung-3 counterfactual queries conditioned on an observed history window
7. Stage 6 is terminal: applying an interactive draft persists it in place rather than triggering downstream replay.

### Outputs

| Output | Type | Description |
|---|---|---|
| `intervention_results` | `list[TreatmentEffect]` | Ranked treatment effects |
| `saved_scenarios` | `list[SavedScenario]` | Optional saved follow-up simulations or scenario notes from interactive use |
| `final_summary` | `str?` | Persisted Stage 6 interpretation, initialized from the baseline ranking and optionally updated by the terminal interactive workflow |
| `llm_trace` | `LLMTrace?` | Opening commentary plus any follow-up interactive turns |

### Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `TreatmentEffect` | `{treatment, effect_size, posterior_draws, prob_positive, identifiable, ppc_warnings, prior_sensitivity_warning, temporal, manifest_effects}` | Final intervention-analysis payload |
| `prob_positive` | `P(effect > 0)` | Posterior probability that the effect is positive |
| `manifest_effects` | Effect decomposed by indicator | Optional per-indicator effect breakdown |
| `temporal` | Time-varying effect trajectory | Present when the intervention effect is time-dependent |

### Follow-Up Tools

| Tool | Purpose |
|---|---|
| `get_model_info` | Read-only summary of variables, measurement, identifiability, diagnostics, and baseline effects |
| `simulate_intervention` | Pearl rung-2 intervention simulation (`set` or `shift`, steady-state or trajectory) |
| `simulate_counterfactual` | Pearl rung-3 forecast conditioned on an observed evidence window |

---

## Cross-Cutting Concerns

### Resume and Replay

- Resume from checkpoint: specify `start_stage` plus `end_stage` to re-run only that window. Earlier stages are restored from persisted snapshots and parquet or pickle artifacts.
- Interactive edits: stages 1a, 1b, 4, and 6 expose interactive surfaces. For stages 1a/1b/4, submitting a custom payload skips computation and resumes downstream. Stage 6 is terminal, so applying an interactive draft persists the updated result in place.
- Stage 5a is always recomputed on resume (never restored from checkpoint).

### Persisted Artifacts

| Stage | Artifact | Format |
|---|---|---|
| 0 | Raw ingested data | Parquet |
| 2 | Raw observation rows | Parquet |
| 2 | Model-ready encoded data | Parquet |
| 5b | Fitted model plus diagnostics | Pickle |

All stage outputs are also serialized to JSON for the web layer. Internal fields prefixed with `_` are stripped from the web payload.
