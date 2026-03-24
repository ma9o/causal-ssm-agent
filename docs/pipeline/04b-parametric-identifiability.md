# Stage 4b: Parametric Identifiability Diagnostics

| Type | Interactive | Gate | Produces |
|---|---|---|---|
| grounding | No | Warning-only | [`ParametricIdResult`](#parametricidresult), [`InferenceStructureResult`](#inferencestructureresult) |

Checks whether the [Stage 4 functional specification](04-model-specification-priors.md) is recoverable from the observed data before committing to expensive inference. The stage runs three progressively finer diagnostics—a counting screen, a Jacobian-based structural check, and a profile-likelihood practical check—then plans the [inference routing](../model-runtime/inference-routing.md) that downstream stages will use.

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | [Stage 4](04-model-specification-priors.md) | Model spec, authored priors, and the compiled SSM artifact |
| `stage2.result` | [Stage 2](02-indicator-extraction.md) | Model-ready observation data (long-format canonical rows) |

Stage 4 provided the parametric model and priors without seeing how tightly the data constrain them. Stage 4b is the first point where the model's parameter recoverability is tested against the actual dataset.

## Process

Stage 4b is a fully deterministic grounding stage (no LLM). It builds the SSM runtime from the compiled artifact, pivots the observation data to wide format, and runs three diagnostic phases in sequence. If a diagnostic fails or is uninformative, the stage degrades gracefully and continues to the next check.

**Model preparation.** The compiled SSM from Stage 4 is hydrated into an `SSMModel` via `prepare_model_runtime`. This step pivots the long-format canonical rows to a `(T, n_manifest)` wide matrix, compiles observation-support metadata (handling interval-summary columns and discrete-manifest families), and resolves the [inference structure](#inferencestructureresult)—the likelihood path, auto-selected inference method, and first-pass Rao-Blackwellization plan—which is emitted as a co-output alongside the diagnostics.

**T-rule counting screen.** A fast necessary-condition check that compares the number of free parameters against a conservative lower bound on the number of independent moment conditions available from the data. For a model with `p` manifest variables observed at `T` time points, the available moments are `p` means + `p(p+1)/2` contemporaneous covariance entries + `(T−1)·p` lagged autocovariance entries (a conservative count using `p` per lag rather than the full `p²` cross-autocovariance). If the free-parameter count exceeds this lower bound, the model is at high risk of non-identifiability. This screen is warning-only: passing does not guarantee identification, and failing does not halt inference. When the T-rule fails, the stage short-circuits and skips the more expensive sensitivity and profile-likelihood analyses.

**Output sensitivity analysis.** A structural identifiability check via the Jacobian of the forward model. For each of several prior draws (default 8), the stage computes the sensitivity matrix `S[i,j] = ∂yᵢ/∂θⱼ` where `y` is the vector of predicted observation means and variances from the Kalman prediction equations (no data update) and `θ` is the unconstrained parameter vector. SVD of `S` reveals structurally non-identifiable parameter directions as near-zero singular values. The analysis runs both raw and normalized variants—the normalized Jacobian scales columns by prior standard deviation and rows by observation-noise scale, giving thresholds in interpretable units. Per-parameter identifiability is classified via the effective singular value (the smallest singular value in which the parameter participates with weight > 0.1): `pass` if > 10⁻³ of the maximum, `warn` if > 10⁻⁶, `fail` otherwise. Results are aggregated across prior draws via the median for robustness.

**Profile likelihood analysis.** A per-parameter practical identifiability diagnostic. The stage first finds the MAP (maximum a posteriori) estimate by BFGS optimization of the log-posterior. For each scalar parameter element, it fixes the parameter at `n_grid` (default 20) evenly spaced points around the MAP, re-optimizes all other parameters at each grid point via BFGS, and records the resulting profile log-likelihood curve. Each parameter is then classified by comparing the profile shape against a χ²(1) threshold (default 95% confidence, threshold = 1.92):

- *identified*: profile drops below threshold on both sides of the peak
- *practically unidentifiable*: profile does not cross the threshold on one or both sides
- *structurally unidentifiable*: profile is flat (total range < 0.5 log-likelihood units)

When [first-pass Rao-Blackwellization](../model-runtime/inference-routing.md) is active with a composed likelihood path, only Kalman-block parameters are profiled—particle-block parameters have stochastic likelihoods that make profile curves unreliable.

**Warning gate.** The stage never hard-gates the pipeline. It emits outcome `"warn"` if the T-rule fails, or if any structural issues, boundary issues, or weakly-identified parameters are detected; otherwise outcome is `"success"`. Warnings are logged and surfaced to the UI but do not block downstream stages.

## Outputs

| Output | Type | Description |
|---|---|---|
| `parametric_id` | [`ParametricIdResult`](#parametricidresult) | Combined T-rule, sensitivity, and profile-likelihood diagnostics |
| `inference_structure` | [`InferenceStructureResult`](#inferencestructureresult) | Resolved likelihood path and inference-routing plan |

The public stage payload exposes these two artifacts directly. It may also include `gate_overridden` if the warning gate was overridden.

## Definitions

### ParametricIdResult

`ParametricIdResult` is the combined pre-fit recoverability payload. It owns:

- `checked`—whether the diagnostics ran successfully
- `t_rule`—the [T-rule result](#truleresult), or null if the check was skipped
- `sensitivity_analysis`—the [output sensitivity result](#sensitivityanalysisresult) with SVD spectrum and per-parameter flags, or null if it failed or was skipped (e.g. because the T-rule short-circuited)
- `summary`—a [`ParametricIdSummary`](#parametricidsummary) with lists of structural issues, boundary issues, and weakly-identified parameters
- `per_param_classification`—per-parameter [profile-likelihood classifications](#parameteridentification) with profile curve data for visualization
- `threshold`—the χ²(1) threshold used for profile-likelihood classification
- `error`—a human-readable error string if the diagnostics failed entirely

### InferenceStructureResult

`InferenceStructureResult` is the resolved inference-routing plan emitted alongside the diagnostics. It records:

- `likelihood_path`—the active likelihood evaluation strategy: `"kalman"` (all-linear-Gaussian, exact), `"composed"` (Kalman sub-block + particle filter), or `"particle"` (full particle filter)
- `auto_method`—the auto-selected inference method: `"nuts"` for fully Gaussian models, `"laplace_em"` when a particle block is present, or `"svi"` when interval-summary support requires it
- `first_pass_rb`—the [first-pass Rao-Blackwellization plan](#firstpassrbresult) with per-variable Kalman/particle assignments

Later stages should treat `InferenceStructureResult` as the authoritative answer to "how will this model be fitted?"

### TRuleResult

`TRuleResult` records the counting-condition check. It carries `n_free_params`, `n_manifest`, `n_timepoints`, `n_moments` (the conservative lower bound), `satisfies` (whether params ≤ moments), and `param_counts`—a breakdown of the free-parameter count by site name (e.g. `drift: 4`, `diffusion: 3`, `lambda: 2`).

### SensitivityAnalysisResult

`SensitivityAnalysisResult` records the Jacobian-based structural check. It carries `singular_values` (the median SVD spectrum across prior draws, descending), `condition_number` (max/min singular value), `per_parameter` (a list of per-parameter entries with `sensitivity_norm`, `effective_sv`, `sv_status`, `normalized_effective_sv`, `normalized_sv_status`, and `identifiable` flag), plus `n_draws`, `n_observations`, and `n_parameters`.

### ParametricIdSummary

`ParametricIdSummary` aggregates the diagnostic findings into three lists: `structural_issues` (parameters classified as structurally unidentifiable), `boundary_issues` (parameters with boundary identifiability problems at some prior draws), and `weak_params` (parameters with low contraction—practically unidentifiable or poorly constrained).

### ParameterIdentification

`ParameterIdentification` is the per-parameter profile-likelihood result. It carries `name`, `classification` (one of `"identified"`, `"practically_unidentifiable"`, `"structurally_unidentifiable"`), `contraction_ratio` (optional), and `profile_x` / `profile_ll` (the profile curve data points for visualization).

### FirstPassRBResult

`FirstPassRBResult` records the first-pass Rao-Blackwellization plan. It carries `status` (`"active"` or `"inactive"`), `inactive_reason` (one of `"disabled_in_spec"`, `"interval_summary_support"`, `"no_executable_partition"`, `"likelihood_override"`), and two variable lists—`latent_variables` and `obs_variables`—each entry carrying a `name` and its `method` assignment (`"kalman"` or `"particle"`).

Example: for a model with three latent constructs (Stress, Sleep Quality, Work Performance) and four indicators, where Stress has a Poisson-distributed indicator, the inference structure might route Sleep Quality and Work Performance through the Kalman filter and Stress through the particle filter (`likelihood_path: "composed"`, `auto_method: "laplace_em"`). The sensitivity analysis might flag the diffusion parameter for Stress as structurally unidentifiable (effective SV < 10⁻⁶ of max), while the profile likelihood confirms Sleep Quality's drift parameter as well-identified (profile crosses threshold on both sides) and Stress's diffusion as practically unidentifiable (profile flat on the right side).
