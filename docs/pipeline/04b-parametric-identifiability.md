# Stage 4b: Parametric Identifiability Diagnostics

| Modality | Interactive | Produces |
|---|---|---|
| Computed | No | [`ParametricIdResult`](#parametricidresult), [`InferenceStructureResult`](#inferencestructureresult) |

Checks whether the [Stage 4 functional specification](04-model-specification-priors.md) passes conservative parametric identification diagnostics before committing to expensive inference, and plans the [inference routing](../reference/inference-routing.md) that downstream stages will use.

## Inputs

| Input | Source | Description |
|---|---|---|
| `compiled_ssm` | [Stage 4](04-model-specification-priors.md) | [`CompiledSSMArtifact`](../reference/compilation.md) with model spec, priors, and compiled SSM |
| `data_for_model` | [Stage 2](02-indicator-extraction.md) | Encoded long-format [`ObservationRecord`](02-indicator-extraction.md#observationrecord) table |

Stage 4 provided the parametric model and priors without seeing how tightly the data constrain them. Stage 4b is the first point where the pipeline evaluates three complementary questions: local structural identifiability via a Jacobian rank diagnostic, dataset-conditioned local curvature via a multi-start MAP Hessian check, and practical identifiability via profile likelihood, following the standard SEM and dynamical-systems terminology in Hunter et al. (2025)[^hunter2025] and Raue et al. (2009)[^raue2009]. This is distinct from [Stage 1b](01b-measurement-identifiability.md) causal identifiability, which asks whether the treatment effect is identified from the causal graph; Stage 4b addresses the complementary question of whether the parameterization is estimable under the available data.

## Process

Stage 4b is a fully deterministic diagnostics stage (no LLM). It prepares the model for evaluation and runs two diagnostic phases in sequence. If a diagnostic fails or is uninformative, the stage degrades gracefully and continues with whatever downstream checks remain valid.

```mermaid
flowchart LR
    M[Model preparation] --> S[Sensitivity analysis] --> G[MAP geometry] --> P[Profile likelihood] --> R([ParametricIdResult])
```

**Model preparation:** The compiled SSM from Stage 4 is built into a runnable model and the observation data are aligned to it. This step also resolves the [inference structure](#inferencestructureresult)—the likelihood path, default inference method, and first-pass Rao-Blackwellization plan—which is emitted as a co-output alongside the diagnostics.

**Sensitivity analysis:** A local structural-identifiability check via the Jacobian rank criterion. For each of several prior draws (default 8), the stage computes the sensitivity matrix `S[i,j] = ∂yᵢ/∂θⱼ` where `y` is an emitted-observation moment summary built from the [Kalman prediction equations](../reference/estimation.md#kalman-backend) without data updates: emitted means, same-row covariance entries, and adjacent-row lagged cross-covariance entries on the observed grid. This follows the same general logic as the Jacobian mapping from free parameters to model-implied moments in Hunter et al. (2025)[^hunter2025]: near-zero singular values reveal locally non-identifiable parameter directions.

*Raw and normalized variants.* The analysis runs both raw and normalized Jacobians. The normalized variant scales columns by prior standard deviation and rows by an observation-noise factor. For mean features the row factor is exactly `σ_obs`, so the squared row weighting equals `Var(y)`; for covariance features it is the product `σ_obs[m] · σ_obs[n]`, which captures only the noise-only `σ_m² σ_n²` term of the sampling variance and ignores the signal-mean and inter-moment cross terms. The normalized Gram matrix is therefore a **diagonal approximation to the Fisher information** scaled by prior variance, not the exact FIM — covariance-loaded directions can appear more identifiable than they actually are when manifest means are large relative to `σ_obs`. The normalized thresholds below are conservative project heuristics, not cutoffs from a standard reference.

*Per-parameter classification.* Each parameter's identifiability is classified via its effective singular value — the smallest singular value in which the parameter participates with weight > 0.1:

- *Pass*: effective SV > 10⁻³ of the maximum
- *Warn*: effective SV > 10⁻⁶ of the maximum
- *Fail*: effective SV ≤ 10⁻⁶ of the maximum

Results are aggregated across prior draws via the median for robustness.

**MAP geometry:** A dataset-conditioned local-geometry diagnostic. The stage runs a multi-start MAP search on the log-posterior, then evaluates both `H_lik(θ̂) = -∇² log p(y | θ)` and `H_post(θ̂) = -∇² log p(θ | y)` at the selected mode `θ̂`. The likelihood Hessian answers which directions the realized data identify locally; the posterior Hessian answers which curvature the optimizer and downstream Gaussian approximations actually see. Both Hessians are reported in raw form and after prior-standard-deviation normalization. Weak directions are summarized by the small normalized eigenvalues and their dominant parameter loadings. Parameters that are weak under `H_lik` but strong under `H_post` are reported as prior-rescued.

**Profile likelihood:** A per-parameter practical-identifiability diagnostic following Raue et al. (2009)[^raue2009]. The stage first finds the MAP (maximum a posteriori) estimate by BFGS optimization of the log-posterior. For each scalar parameter element, it fixes the parameter at `n_grid` (default 20) evenly spaced points around the MAP, re-optimizes all other parameters at each grid point via BFGS, and records the resulting profile log-likelihood curve. Each parameter is then classified by comparing the profile shape against a χ²(1) threshold (default 95% confidence, threshold = 1.92):

- *identified*: profile drops below threshold on both sides of the peak
- *practically unidentifiable*: profile does not cross the threshold on one or both sides
- *structurally unidentifiable*: profile is flat (total range < 0.5 log-likelihood units)

When [first-pass Rao-Blackwellization](../reference/inference-routing.md#first-pass-rao-blackwellization) is active with a composed likelihood path, only Kalman-block parameters are profiled—particle-block parameters have stochastic likelihoods that make profile curves unreliable.

### Example

For a model with three latent constructs (Stress, Sleep Quality, Work Performance) and four indicators, where Stress has a Poisson-distributed indicator, the inference structure might route Sleep Quality and Work Performance through the Kalman filter and Stress through the particle filter (`likelihood_path: "composed"`, `auto_method: "aux_gibbs"`). The sensitivity analysis might flag the diffusion parameter for Stress as structurally unidentifiable (effective SV < 10⁻⁶ of max), while the profile likelihood confirms Sleep Quality's drift parameter as well-identified (profile crosses threshold on both sides) and Stress's diffusion as practically unidentifiable (profile flat on the right side).

## Outputs

| Output | Type | Description |
|---|---|---|
| `parametric_id` | `ParametricIdResult` | Combined sensitivity, MAP-geometry, and profile-likelihood diagnostics |
| `inference_structure` | [`InferenceStructureResult`](#inferencestructureresult) | Resolved likelihood path and inference-routing plan; consumed by the web frontend — [Stage 5](05a-svi-preflight.md) re-derives the routing at fit time |

### `ParametricIdResult`

| Field | Type | Description |
|---|---|---|
| `checked` | `bool` | Whether Stage 4b completed its diagnostics pass |
| `sensitivity_analysis` | `SensitivityAnalysisResult \| null` | Output-space Jacobian sensitivity diagnostic |
| `map_geometry` | `MAPGeometryResult \| null` | Multi-start MAP search with likelihood and posterior Hessian summaries |
| `summary` | `ParametricIdSummary \| null` | Aggregated structural, boundary, and weak-parameter findings |
| `per_param_classification` | `list[ParameterIdentification] \| null` | Per-parameter profile-likelihood classifications and optional profile traces |
| `threshold` | `float \| null` | χ²-derived profile-likelihood threshold used for classification |
| `error` | `str \| null` | Diagnostic failure message when Stage 4b could not complete |

### `ParametricIdResult.ParametricIdSummary`

| Field | Type | Description |
|---|---|---|
| `structural_issues` | `list[str]` | Parameters classified as structurally unidentifiable |
| `boundary_issues` | `list[str]` | Parameters with boundary-pathology warnings |
| `weak_params` | `list[str]` | Parameters flagged as weakly identified by sensitivity or profiling |

### `ParametricIdResult.SensitivityAnalysisResult`

| Field | Type | Description |
|---|---|---|
| `singular_values` | `list[float]` | Median SVD spectrum across prior draws, descending |
| `deficiency_count` | `int` | Number of normalized singular values below the fail threshold, counting non-identifiable directions in parameter space |
| `per_parameter` | `list[dict]` | Per-parameter pass/warn/fail classification via effective singular value |

### `ParametricIdResult.MAPGeometryResult`

| Field | Type | Description |
|---|---|---|
| `n_starts` | `int` | Number of MAP starts actually optimized |
| `n_successful_starts` | `int` | Number of starts whose optimizer reported convergence |
| `best_start_index` | `int` | Index of the selected best run inside `starts` |
| `map_log_posterior` | `float` | Log-posterior at the selected MAP |
| `map_log_likelihood` | `float` | Log-likelihood at the selected MAP |
| `map_log_prior` | `float` | Log-prior in unconstrained space at the selected MAP |
| `final_grad_norm` | `float` | Euclidean norm of the optimizer gradient at the selected MAP |
| `runner_up_objective_gap` | `float \| null` | Objective gap between the best and second-best finite runs |
| `starts` | `list[MAPOptimizationRun]` | Per-start MAP optimization outcomes |
| `likelihood_curvature` | `MAPCurvatureResult` | Local Hessian summary for `H_lik` |
| `posterior_curvature` | `MAPCurvatureResult` | Local Hessian summary for `H_post` |
| `prior_rescued_parameters` | `list[str]` | Parameters weak under `H_lik` but strong under `H_post` |
| `boundary_parameters` | `list[str]` | Parameters whose MAP lies near a support or prior bound |

### `ParametricIdResult.MAPCurvatureResult`

| Field | Type | Description |
|---|---|---|
| `eigenvalues` | `list[float]` | Raw Hessian eigenvalues, descending |
| `normalized_eigenvalues` | `list[float]` | Prior-SD-normalized eigenvalues, descending |
| `negative_direction_count` | `int` | Number of negative-curvature directions in the local Hessian |
| `deficiency_count` | `int` | Number of normalized eigenvalues below the weak-direction threshold |
| `positive_definite` | `bool` | Whether every eigenvalue is positive up to numerical tolerance |
| `condition_number` | `float \| null` | Raw Hessian condition number when the matrix is positive definite |
| `normalized_condition_number` | `float \| null` | Condition number after prior-SD normalization |
| `weak_directions` | `list[CurvatureDirection]` | Weak normalized eigen-directions with dominant parameter loadings |
| `per_parameter` | `list[CurvatureParameterEntry]` | Per-parameter effective-curvature summary |

### `ParametricIdResult.MAPOptimizationRun`

| Field | Type | Description |
|---|---|---|
| `index` | `int` | Start index within the multi-start run list |
| `start_kind` | `str` | Origin of the starting point such as `zero`, `prior_median`, or `prior_draw_k` |
| `start_log_posterior` | `float` | Log-posterior at the initial point before optimization |
| `log_posterior` | `float` | Log-posterior at the optimized point |
| `log_likelihood` | `float` | Log-likelihood at the optimized point |
| `log_prior` | `float` | Log-prior at the optimized point |
| `objective` | `float` | Final negative log-posterior minimized by the optimizer |
| `success` | `bool` | Optimizer convergence flag |
| `status` | `int` | Optimizer status code |
| `message` | `str` | Optimizer termination message |
| `n_iters` | `int` | Optimizer iteration count |
| `n_function_evals` | `int` | Number of objective evaluations |
| `grad_norm` | `float` | Euclidean norm of the final gradient |
| `distance_to_best` | `float` | Euclidean distance from this solution to the selected best MAP |

### `ParametricIdResult.ParameterIdentification`

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Parameter name |
| `classification` | `str` | One of `"identified"`, `"practically_unidentifiable"`, `"structurally_unidentifiable"` |
| `contraction_ratio` | `float` \| `null` | Ratio measuring how much the data contract the prior |
| `profile_x` | `list[float]` \| `null` | Grid of constrained parameter values used for the profile |
| `profile_ll` | `list[float]` \| `null` | Profile log-likelihood values recentered at the local peak |

### `InferenceStructureResult`

`InferenceStructureResult` records the resolved routing plan for display in the web frontend. Stage 5 re-derives the same plan at fit time from the compiled SSM.

| Field | Type | Description |
|---|---|---|
| `likelihood_path` | `str` | Likelihood evaluation strategy per [structural routing](../reference/inference-routing.md#structural-routing): `"kalman"`, `"composed"`, or `"particle"` |
| `auto_method` | `str` | Current default inference method per [structural routing](../reference/inference-routing.md#structural-routing): `"aux_gibbs"` |
| `first_pass_rb` | `FirstPassRBResult` | [First-pass Rao-Blackwellization](../reference/inference-routing.md#first-pass-rao-blackwellization) plan with per-variable Kalman/particle assignments |

[^hunter2025]: Hunter, M. D., Kirkpatrick, R. M., & Neale, M. C. (2025). Show Me Some ID: A Universal Identification Program for Structural Equation Models. *Psychometrika*, 90(2), 418-441. [Bibliography entry](../reference/bibliography.md)
[^raue2009]: Raue, A., et al. (2009). Structural and Practical Identifiability Analysis of Partially Observed Dynamical Models. *Bioinformatics*, 25(15), 1923–1929. [Bibliography entry](../reference/bibliography.md)
