# Stage 5a: SVI Preflight

| Modality | Interactive | Produces |
|---|---|---|
| Computed | No | approximate posterior and [ELBO diagnostics](#svidiagnostics) |

Runs a cheap variational fit as a sanity check before [Stage 5b](05b-inference-diagnostics.md) commits to expensive inference. The stage forces SVI with a lightweight fixed configuration, produces an ELBO convergence curve and approximate posterior summaries, and treats failure as best-effort—the pipeline never halts here.

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | [Stage 4](04-model-specification-priors.md) | Compiled SSM (`_compiled_ssm`) and model spec |
| `stage2.result` | [Stage 2](02-indicator-extraction.md) | Model-ready observation data (Parquet) |

Stage 4 provided the [ModelSpec](04-model-specification-priors.md#modelspec) and priors; Stage 2 provided the extracted indicator time series. Stage 5a is the first point where the compiled model meets the data for fitting.

## Process

Stage 5a reuses the same `fit_model` task as Stage 5b but with method and budget forced. The runtime preparation, variational optimization, and summary extraction run as a single deterministic pipeline with no LLM involvement.

**Model compilation.** The stage loads the Stage 2 Parquet data, then calls `prepare_model_runtime` which pivots the long-form observation rows into wide format, compiles the Stage 4 `_compiled_ssm` into an executable `SSMModel` (via `SSMSpec` + `SSMPriors`), and plans the [inference structure](../reference/estimation.md) (likelihood backend, Rao-Blackwellization split). The result is a `PreparedModelRuntime` carrying the JAX observation array `(T, n_manifest)`, the time array `(T,)`, and the built model.

**SVI optimization.** The stage calls `fit` with a fixed configuration: `method="svi"`, `num_steps=5000`, `num_samples=500`. Internally, `_fit_svi` constructs an `AutoMultivariateNormal` guide over all latent sample sites, pairs it with a `ClippedAdam` optimizer (learning rate 0.01) and a `Trace_ELBO` loss, and runs 5 000 gradient steps. Non-finite losses or guide parameters raise a `FloatingPointError`, which the outer try/except catches and maps to `outcome="warn"`.

**Posterior sampling.** After optimization converges, 500 draws are sampled from the fitted guide via NumPyro's `Predictive`. The samples are filtered to public sites only (excluding internal factor sites).

**Diagnostic extraction.** From the fitted `InferenceResult` the stage extracts:

- *ELBO curve*: the full loss trace, thinned to at most 500 points for the frontend.
- *Posterior marginals*: for each scalar parameter (and the first 20 elements of array parameters), a histogram-based density curve with 50 bins, the posterior mean and standard deviation, and the 94% [highest density interval](https://en.wikipedia.org/wiki/Credible_interval#Highest_density_interval) (HDI) bounds.
- *Posterior pairs*: pairwise scatter plots for up to 6 scalar parameters, thinned to at most 200 points. Because SVI produces no chain-level divergence information, the `divergent` field is always `null` for this stage.

**Failure semantics.** If model fitting raises any exception (missing implementation, numerical failure, etc.), the stage returns `outcome="warn"` with `n_samples=0` and all diagnostic fields set to `null`. The pipeline continues to Stage 5b regardless because Stage 5a is best-effort preflight only.

**Recompute-only resume.** Stage 5a is marked `skip_restore=True` in the stage registry—it is never restored from a prior run and always recomputed when the pipeline executes. This follows the recompute rules in [execution-semantics.md](../reference/execution-semantics.md#resume-semantics).

## Outputs

| Output | Type | Description |
|---|---|---|
| `inference_metadata` | [`InferenceMetadata`](#inferencemetadata) | Method, sample count, and timing |
| `svi_diagnostics` | [`SVIDiagnostics`](#svidiagnostics) &#124; null | ELBO loss curve (null on failure) |
| `posterior_marginals` | list\[[`PosteriorMarginal`](#posteriormarginal)\] &#124; null | Per-parameter density summaries (null on failure) |
| `posterior_pairs` | list\[[`PosteriorPair`](#posteriorpair)\] &#124; null | Pairwise scatter data (null on failure) |

The contract also exposes `outcome` (`"success"` or `"warn"`) inherited from the base stage contract.

## Definitions

The definitions in this section are shared by Stage 5a and Stage 5b. Stage 5b links back here for the common inference-output payloads rather than restating them.

### InferenceMetadata

Summary metadata for the web frontend. Fields:

| Field | Type | Description |
|---|---|---|
| `method` | `str` | Always `"svi"` for this stage |
| `n_samples` | `int` | Number of posterior draws (500 on success, 0 on failure) |
| `duration_seconds` | `float` | Wall-clock time for the fit |

### SVIDiagnostics

Convergence diagnostics for the variational optimization.

| Field | Type | Description |
|---|---|---|
| `elbo_losses` | `list[float]` | ELBO loss at each optimization step, thinned to at most 500 points |

A monotonically decreasing curve indicates convergence; large oscillations or a plateau at a high value suggest the guide family cannot capture the posterior geometry. Stage 5b may choose a more expressive backend (NUTS, Laplace-EM) in that case.

### PosteriorMarginal

Marginal posterior density for a single scalar parameter, computed via histogram binning with 5% tail padding.

| Field | Type | Description |
|---|---|---|
| `parameter` | `str` | Parameter name (array elements indexed as `name[i]`) |
| `x_values` | `list[float]` | Bin centers for the density curve |
| `density` | `list[float]` | Normalized density at each bin center |
| `mean` | `float` | Posterior mean |
| `sd` | `float` | Posterior standard deviation |
| `hdi_3` | `float` | Lower bound of the 94% HDI |
| `hdi_97` | `float` | Upper bound of the 94% HDI |

### PosteriorPair

Pairwise posterior scatter data for joint visualization. Up to 6 parameters are selected; array parameters contribute at most 4 elements each.

| Field | Type | Description |
|---|---|---|
| `param_x` | `str` | Name of the x-axis parameter |
| `param_y` | `str` | Name of the y-axis parameter |
| `x_values` | `list[float]` | Thinned posterior draws for x (at most 200 points) |
| `y_values` | `list[float]` | Thinned posterior draws for y |
| `divergent` | `list[bool]` &#124; null | Always `null` for SVI (no chain-level divergence data) |

Example: for a model of symptom burden and medication adherence with latent constructs `Symptom Burden`, `Medication Adherence`, and `Functional Capacity`, Stage 5a would run 5 000 SVI steps over the compiled SSM, then produce an ELBO curve showing whether the multivariate normal guide captured the joint posterior, marginal densities for each drift and loading parameter, and pairwise scatter plots revealing any strong posterior correlations before Stage 5b invests in a full NUTS or Laplace-EM run.
