# Stage 5b: Inference and Diagnostics

| Modality | Interactive | Gate | Produces |
|---|---|---|---|
| Computed | No | No | [`FittedArtifact`](#fittedartifact) plus [PPC](#ppcresult), [power-scaling](#powerscalingresult), and [backend-specific diagnostics](#backend-specific-diagnostics) |

Fits the compiled state-space model from [Stage 4](04-model-specification-priors.md) to the extracted observation data from [Stage 2](02-indicator-extraction.md), then runs post-fit diagnostics that assess prior–data agreement, posterior predictive calibration, and leave-one-out cross-validation. Backend selection follows the [structural routing](../reference/inference-routing.md) decision tree: NUTS for Kalman-eligible models (all Gaussian emissions with identity links), Laplace-EM for non-Gaussian emissions. The user can override to any of the nine [available methods](../reference/inference-routing.md#method-taxonomy).

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | [Stage 4](04-model-specification-priors.md) | Compiled SSM (`_compiled_ssm`) and [model spec](04-model-specification-priors.md#modelspec) with priors |
| `stage2.result` | [Stage 2](02-indicator-extraction.md) | Model-ready observation data (Parquet path via `_data_for_model_path`) |
| `inference_method` | Pipeline config | Optional sampler override (`"nuts"`, `"laplace_em"`, `"svi"`, etc.); `null` triggers [auto-routing](../reference/inference-routing.md#structural-routing) |

Stage 4 provided the functional specification and priors; Stage 2 provided the extracted indicator time series. Stage 5b is where the compiled model is fitted to data and the posterior is characterized.

## Process

Stage 5b runs three sequential tasks with no LLM involvement: model fitting, power-scaling sensitivity analysis, and posterior predictive checks. The output is a deterministic function of the compiled model, the data, and the inference configuration.

**Runtime preparation.** The stage runs the same [`prepare_model_runtime`](05a-svi-preflight.md#process) path as Stage 5a—pivoting observation rows to wide format and compiling the executable SSM—but with the full inference method and budget rather than the fixed SVI configuration.

**Model fitting.** The `fit_model` task resolves the inference method—either the user-supplied override or the [auto-routed default](../reference/inference-routing.md#decision-tree)—and delegates to the corresponding [backend](../reference/inference-routing.md#method-reference). The two structural defaults are NUTS (Kalman-eligible models) and Laplace-EM (non-Gaussian emissions); all nine methods are available as user overrides.

The fit produces an `InferenceResult` containing the posterior samples dict `{name: (n_draws, *shape)}`, the method identifier, and backend-specific diagnostics. From this the stage extracts backend-specific diagnostic payloads (MCMC convergence, SVI ELBO curve, SMC tempering schedule), LOO-CV diagnostics, posterior marginals, and posterior pairs.

**LOO cross-validation.** For both MCMC and SMC backends, the stage computes PSIS-LOO via ArviZ using the innovation decomposition: each "observation" is one complete timestep (all manifest variables at time *t*), and the per-timestep log-likelihoods log p(y\_t | y\_{1:t−1}, θ) are conditionally independent given θ. The result includes expected log pointwise predictive density (ELPD), the effective number of parameters (p\_loo), per-timestep Pareto-k values (flagging observations with k > 0.7 as poorly approximated), and LOO-PIT values for calibration assessment.

**Power-scaling sensitivity analysis.** The `run_power_scaling` task (Kallioinen et al. 2023) detects whether each parameter's posterior is dominated by the prior, well-identified by the data, or in prior–data conflict. It perturbs the prior and likelihood contributions by a small power-scaling factor (α ± 0.01), reweights the posterior draws via PSIS, and measures the resulting shift in posterior means. Each parameter receives one of three diagnoses: `well_identified` (both sensitivities low), `prior_dominated` (high prior sensitivity, low likelihood sensitivity), or `prior_data_conflict` (both sensitivities high). The PSIS k-hat diagnostic for each perturbation direction indicates whether the importance-weighted estimate is reliable (k < 0.7).

**Posterior predictive checks.** The `run_ppc` task forward-simulates observations from posterior parameter draws through the full generative model (latent dynamics → discretization → emission sampling) and compares the simulated data to the real observations. For each manifest variable it produces:

- *Calibration check*: fraction of observed values falling within the 95% posterior predictive interval. Flags if coverage is below 0.80 or above 0.99.
- *Autocorrelation check*: compares lag-1 autocorrelation of the observed series to the distribution of lag-1 autocorrelations across replicated datasets.
- *Variance check*: compares the observed variance to the distribution of variances across replicated datasets.
- *Overlay data*: quantile bands (2.5%, 25%, 50%, 75%, 97.5%) of the posterior predictive distribution at each timestep, plus spaghetti draws for density overlay plots.
- *Test statistics*: distribution of T(y\_rep) for mean, standard deviation, min, and max, with a vertical marker at T(y\_observed).

**Artifact assembly and persistence.** The stage assembles a [`FittedArtifact`](#fittedartifact) packaging the `InferenceResult`, the runtime builder (needed by Stage 6 for intervention simulations), timing metadata, observation-support metadata, and the PPC and power-scaling results. This artifact is pickled to `stage5b-fitted-result.pkl` and is the sole runtime object consumed by [Stage 6](06-intervention-analysis.md). The web-facing diagnostic payloads (power-scaling list, PPC result, backend diagnostics, marginals, pairs) are persisted separately as the public JSON contract.

**Outcome classification.** The stage sets `outcome` to `"warn"` if any power-scaling diagnosis is `prior_dominated` or `prior_data_conflict`, or if any PPC variable has warnings. Otherwise `outcome` is `"success"`. There is no hard gate—the pipeline always continues to Stage 6 regardless of diagnostic results.

## Outputs

| Output | Type | Description |
|---|---|---|
| `power_scaling` | list\[[`PowerScalingResult`](#powerscalingresult)\] | Per-parameter sensitivity diagnosis with PSIS reliability |
| `ppc` | [`PPCResult`](#ppcresult) | Per-variable calibration, autocorrelation, and variance checks plus overlay data |
| `inference_metadata` | [`InferenceMetadata`](#inferencemetadata) | Method, sample count, and timing |
| `mcmc_diagnostics` | [`MCMCDiagnostics`](#mcmcdiagnostics) \| null | NUTS convergence diagnostics (null for non-MCMC backends) |
| `svi_diagnostics` | [`SVIDiagnostics`](#svidiagnostics) \| null | ELBO loss curve (null for non-SVI backends) |
| `smc_diagnostics` | [`SMCDiagnostics`](#smcdiagnostics) \| null | Tempering schedule and ESS history (null for non-SMC backends) |
| `loo_diagnostics` | [`LOODiagnostics`](#loodiagnostics) \| null | PSIS-LOO cross-validation with per-timestep Pareto-k values |
| `posterior_marginals` | list\[[`PosteriorMarginal`](05a-svi-preflight.md#posteriormarginal)\] \| null | Per-parameter density summaries |
| `posterior_pairs` | list\[[`PosteriorPair`](05a-svi-preflight.md#posteriorpair)\] \| null | Pairwise scatter data (includes `divergent` flags for MCMC backends) |

The contract also exposes `outcome` (`"success"` or `"warn"`) inherited from the base stage contract.

## Definitions

### FittedArtifact

`FittedArtifact` is the persisted runtime object produced by Stage 5b and consumed by [Stage 6](06-intervention-analysis.md). It is the sole handoff object in the [model-runtime chain](../reference/compilation.md).

| Field | Type | Description |
|---|---|---|
| `result` | `InferenceResult` \| null | Posterior samples and backend-specific diagnostics |
| `builder` | `SSMModelBuilder` \| null | Runtime builder needed for intervention simulations |
| `times` | `jnp.ndarray` \| null | Observation time points `(T,)` |
| `observation_support` | `ObservationSupportRuntime` \| null | Interval-summary and support-boundary metadata |
| `ppc_result` | `dict` \| null | PPC diagnostics attached to the fitted run |
| `power_scaling_result` | `dict` \| null | Power-scaling diagnostics attached to the fitted run |

### PowerScalingResult

Per-parameter power-scaling sensitivity diagnosis (Kallioinen et al. 2023).

| Field | Type | Description |
|---|---|---|
| `parameter` | `str` | Parameter name |
| `diagnosis` | `"prior_dominated"` \| `"well_identified"` \| `"prior_data_conflict"` | Classification of prior–data relationship |
| `prior_sensitivity` | `float` | Posterior mean shift under prior power-scaling perturbation |
| `likelihood_sensitivity` | `float` | Posterior mean shift under likelihood power-scaling perturbation |
| `psis_k_hat` | `float` \| null | Pareto-k reliability diagnostic for the importance-weighted estimate |

### PPCResult

Aggregate posterior predictive check result.

| Field | Type | Description |
|---|---|---|
| `per_variable_warnings` | `list[PPCWarning]` | Per-manifest-variable diagnostic warnings |
| `checked` | `bool` \| null | Whether PPC ran successfully |
| `n_subsample` | `int` \| null | Number of posterior draws used for PPC |
| `overlays` | `list[PPCOverlay]` | Per-variable quantile bands for ribbon/density overlay plots |
| `test_stats` | `list[PPCTestStat]` | Per-variable test statistic distributions vs observed |

`PPCWarning` carries `variable`, `check_type` (`"calibration"`, `"autocorrelation"`, or `"variance"`), `message`, `value`, and `passed`.

`PPCOverlay` provides `observed` (with nulls for missing timesteps), quantile bands (`q025`, `q25`, `median`, `q75`, `q975`), and optional `spaghetti_draws` for density overlay plots.

`PPCTestStat` provides `stat_name` (`"mean"`, `"sd"`, `"min"`, `"max"`), the `observed_value`, and the distribution of `rep_values` across posterior predictive draws.

### InferenceMetadata

Same schema as [Stage 5a `InferenceMetadata`](05a-svi-preflight.md#inferencemetadata), with `method` reflecting the actual backend used (e.g. `"nuts"`, `"laplace_em"`).

### Backend-Specific Diagnostics

Exactly one of the three backend-specific diagnostic payloads is non-null, determined by which inference method ran.

#### MCMCDiagnostics

Produced by NUTS, NUTS-DA, and PGAS backends.

| Field | Type | Description |
|---|---|---|
| `per_parameter` | `list[MCMCParamDiagnostic]` | Per-parameter R-hat, ESS (bulk + tail), and MCSE |
| `num_divergences` | `int` | Total divergent transitions |
| `divergence_rate` | `float` | Fraction of transitions that diverged |
| `tree_depth_mean` | `float` | Mean tree depth (NUTS only) |
| `tree_depth_max` | `int` | Maximum tree depth reached |
| `accept_prob_mean` | `float` | Mean MH acceptance probability |
| `num_chains` | `int` \| null | Number of chains |
| `num_samples` | `int` \| null | Draws per chain |
| `trace_data` | `list[TraceData]` \| null | Per-parameter thinned trace values across chains (at most 200 points per chain) |
| `rank_histograms` | `list[RankHistogram]` \| null | Per-parameter rank histograms for chain mixing assessment (20 bins) |
| `energy` | `EnergyDiagnostics` \| null | NUTS energy diagnostics with BFMI ([Betancourt 2017](https://arxiv.org/abs/1701.02434)) |

#### SVIDiagnostics

Produced by the SVI backend. Same schema as [Stage 5a `SVIDiagnostics`](05a-svi-preflight.md#svidiagnostics).

#### SMCDiagnostics

Produced by Laplace-EM, tempered SMC, Hess-MC², structured VI, and DPF backends.

| Field | Type | Description |
|---|---|---|
| `beta_schedule` | `list[float]` | Tempering ladder β₀=0 → β\_K=1 |
| `ess_history` | `list[float]` | Effective sample size at each tempering level |
| `accept_rates` | `list[float]` | Mutation acceptance rate at each level |
| `n_levels` | `int` | Number of tempering levels |
| `n_particles` | `int` | Number of SMC particles |

### LOODiagnostics

Leave-one-out cross-validation via PSIS-LOO (Vehtari, Gelman & Gabry 2017). Uses the innovation decomposition so each LOO "observation" is one complete timestep, not individual cells.

| Field | Type | Description |
|---|---|---|
| `elpd_loo` | `float` | Expected log pointwise predictive density |
| `p_loo` | `float` | Effective number of parameters |
| `se` | `float` | Standard error of the ELPD estimate |
| `n_data_points` | `int` | Number of timesteps used |
| `observation_unit` | `str` | Always `"timestep"` for the SSM path |
| `pareto_k` | `list[float]` \| null | Per-timestep Pareto-k shape parameters |
| `n_bad_k` | `int` \| null | Count of timesteps with k > 0.7 |
| `loo_pit` | `list[float]` \| null | LOO-PIT values for calibration assessment |

Example: for a longitudinal study of teacher workload and student outcomes with latent constructs `Teacher Burnout`, `Instructional Quality`, and `Student Achievement`, Stage 5b would auto-route to NUTS (all Gaussian emissions), run 4 chains of 2 500 draws each, then produce: MCMC diagnostics showing R-hat < 1.01 and ESS > 400 for all drift and loading parameters; power-scaling results classifying the cross-lag from `Teacher Burnout` to `Instructional Quality` as `well_identified` and a weakly informed diffusion parameter as `prior_dominated`; PPC overlays showing 93% calibration coverage for each manifest indicator; and LOO diagnostics with no Pareto-k values exceeding 0.7—all before Stage 6 uses the fitted artifact to simulate interventions.
