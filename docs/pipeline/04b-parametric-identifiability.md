# Stage 4b: Parametric Identifiability Diagnostics

| Modality | Interactive | Produces |
|---|---|---|
| Computed | No | [`ParametricIdResult`](#parametricidresulttruleresult), [`InferenceStructureResult`](#inferencestructureresult) |

Checks whether the [Stage 4 functional specification](04-model-specification-priors.md) is numerically recoverable from the observed data before committing to expensive inference, and plans the [inference routing](../reference/inference-routing.md) that downstream stages will use.

## Inputs

| Input | Source | Description |
|---|---|---|
| `compiled_ssm` | [Stage 4](04-model-specification-priors.md) | [`CompiledSSMArtifact`](../reference/compilation.md) with model spec, priors, and compiled SSM |
| `data_for_model` | [Stage 2](02-indicator-extraction.md) | Encoded long-format [`ObservationRecord`](02-indicator-extraction.md#observationrecord) table |

Stage 4 provided the parametric model and priors without seeing how tightly the data constrain them. Stage 4b is the first point where parameter recoverability is tested against the actual dataset. This is distinct from [Stage 1b](01b-measurement-identifiability.md) causal identifiability, which asks whether the treatment effect is identified from the causal graph; Stage 4b addresses the complementary question of whether the parameterization is constrained by the available data.

## Process

Stage 4b is a fully deterministic grounding stage (no LLM). It prepares the model for evaluation and runs three diagnostic phases in sequence. If a diagnostic fails or is uninformative, the stage degrades gracefully and continues to the next check.

```mermaid
flowchart LR
    M[Model prep] --> T{T-rule} -- fail --> R([ParametricIdResult])
    T -- pass --> S[Sensitivity analysis] --> P[Profile likelihood] --> R
```

**Model preparation:** The compiled SSM from Stage 4 is built into a runnable model and the observation data are aligned to it. This step also resolves the [inference structure](#inferencestructureresult)—the likelihood path, auto-selected inference method, and first-pass Rao-Blackwellization plan—which is emitted as a co-output alongside the diagnostics.

**T-rule counting screen:** A fast necessary-condition check that compares the number of free parameters against a conservative lower bound on the number of independent moment conditions available from the data. For a model with `p` manifest variables observed at `T` time points, the available moments are:

- `p` means
- `p(p+1)/2` contemporaneous covariance entries
- `(T−1)·p` lagged autocovariance entries (a conservative count using `p` per lag rather than the full `p²` cross-autocovariance)

If the free-parameter count exceeds this lower bound, the model is at high risk of non-identifiability. This screen is warning-only: passing does not guarantee identification, and failing does not halt inference. When the T-rule fails, the stage short-circuits and skips the more expensive sensitivity and profile-likelihood analyses.

**Output sensitivity analysis:** A structural identifiability check via the Jacobian of the forward model. For each of several prior draws (default 8), the stage computes the sensitivity matrix `S[i,j] = ∂yᵢ/∂θⱼ` where `y` is the vector of predicted observation means and variances from the [Kalman prediction equations](../reference/estimation.md#kalman-backend) (no data update) and `θ` is the unconstrained parameter vector. SVD of `S` reveals structurally non-identifiable parameter directions as near-zero singular values.

*Raw and normalized variants.* The analysis runs both raw and normalized Jacobians. The normalized variant scales columns by prior standard deviation and rows by observation-noise scale, giving thresholds in interpretable units.

*Per-parameter classification.* Each parameter's identifiability is classified via its effective singular value — the smallest singular value in which the parameter participates with weight > 0.1:

- *Pass*: effective SV > 10⁻³ of the maximum
- *Warn*: effective SV > 10⁻⁶ of the maximum
- *Fail*: effective SV ≤ 10⁻⁶ of the maximum

Results are aggregated across prior draws via the median for robustness.

**Profile likelihood analysis:** A per-parameter practical identifiability diagnostic. The stage first finds the MAP (maximum a posteriori) estimate by BFGS optimization of the log-posterior. For each scalar parameter element, it fixes the parameter at `n_grid` (default 20) evenly spaced points around the MAP, re-optimizes all other parameters at each grid point via BFGS, and records the resulting profile log-likelihood curve. Each parameter is then classified by comparing the profile shape against a χ²(1) threshold (default 95% confidence, threshold = 1.92):

- *identified*: profile drops below threshold on both sides of the peak
- *practically unidentifiable*: profile does not cross the threshold on one or both sides
- *structurally unidentifiable*: profile is flat (total range < 0.5 log-likelihood units)

When [first-pass Rao-Blackwellization](../reference/inference-routing.md#first-pass-rao-blackwellization) is active with a composed likelihood path, only Kalman-block parameters are profiled—particle-block parameters have stochastic likelihoods that make profile curves unreliable.

### Example

For a model with three latent constructs (Stress, Sleep Quality, Work Performance) and four indicators, where Stress has a Poisson-distributed indicator, the inference structure might route Sleep Quality and Work Performance through the Kalman filter and Stress through the particle filter (`likelihood_path: "composed"`, `auto_method: "laplace_em"`). The sensitivity analysis might flag the diffusion parameter for Stress as structurally unidentifiable (effective SV < 10⁻⁶ of max), while the profile likelihood confirms Sleep Quality's drift parameter as well-identified (profile crosses threshold on both sides) and Stress's diffusion as practically unidentifiable (profile flat on the right side).

## Outputs

| Output | Type | Description |
|---|---|---|
| `parametric_id` | `ParametricIdResult` | Combined T-rule, sensitivity, and profile-likelihood diagnostics |
| `inference_structure` | [`InferenceStructureResult`](#inferencestructureresult) | Resolved likelihood path and inference-routing plan; consumed by the web frontend — [Stage 5](05a-svi-preflight.md) re-derives the routing at fit time |

### `ParametricIdResult.TRuleResult`

| Field | Type | Description |
|---|---|---|
| `n_free_params` | `int` | Total count of free parameters in the model |
| `n_manifest` | `int` | Number of manifest (observed) variables |
| `n_timepoints` | `int` | Number of observed time points |
| `n_moments` | `int` | Conservative lower bound on independent moment conditions |
| `satisfies` | `bool` | Whether `n_free_params` ≤ `n_moments` |
| `param_counts` | `dict[str, int]` | Free-parameter count by site name (e.g. `drift: 4`, `diffusion: 3`, `lambda: 2`) |

### `ParametricIdResult.SensitivityAnalysisResult`

| Field | Type | Description |
|---|---|---|
| `singular_values` | `list[float]` | Median SVD spectrum across prior draws, descending |
| `condition_number` | `float` | Ratio of max to min singular value |
| `per_parameter` | `list[dict]` | Per-parameter pass/warn/fail classification via effective singular value |

### `ParametricIdResult.ParameterIdentification`

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Parameter name |
| `classification` | `str` | One of `"identified"`, `"practically_unidentifiable"`, `"structurally_unidentifiable"` |
| `contraction_ratio` | `float` \| `null` | Ratio measuring how much the data contract the prior |

### `InferenceStructureResult`

`InferenceStructureResult` records the resolved routing plan for display in the web frontend. Stage 5 re-derives the same plan at fit time from the compiled SSM.

| Field | Type | Description |
|---|---|---|
| `likelihood_path` | `str` | Likelihood evaluation strategy per [Axis B](../reference/inference-routing.md#axis-b-marginal-likelihood-computation): `"kalman"`, `"composed"`, or `"particle"` |
| `auto_method` | `str` | Auto-selected inference method per the [structural routing](../reference/inference-routing.md#structural-routing) decision tree: `"nuts"`, `"laplace_em"`, or `"svi"` |
| `first_pass_rb` | `FirstPassRBResult` | [First-pass Rao-Blackwellization](../reference/inference-routing.md#first-pass-rao-blackwellization) plan with per-variable Kalman/particle assignments |
