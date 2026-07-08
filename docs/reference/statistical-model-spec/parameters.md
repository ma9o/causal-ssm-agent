# Parameters and Priors

Defines the parameter roles, prior vocabulary, and default guidance for [`ParameterSpec`](../../pipeline/statistical-model-spec.md#parameterspec) and [`PriorProposal`](../../pipeline/statistical-model-spec.md) entries in a [`StatisticalModelSpec`](../../pipeline/statistical-model-spec.md#statisticalmodelspec).

> All sections below are generated from `nof1_causal_lab.distributions`.
> Edit the Python catalog and re-run `uv run python scripts/export_distribution_docs.py` instead of editing them manually.

## Parameter Roles

The [model-spec skeleton](../../pipeline/statistical-model-spec.md) creates exactly the following parameters from a [`CausalDesign`](../../pipeline/measurement-structure.md#causaldesign):

| Role | Symbol | Count | Constraint | SSM location |
|---|---|---|---|---|
| `ar_coefficient` | `rho` | One per endogenous time-varying construct | `unit_interval` `[0, 1]` | State-decay dynamics site |
| `fixed_effect` | `beta` | One per causal edge | `none` `(-inf, +inf)` | Dynamics edge or input-effect site |
| `dynamics_parameter` | `theta` | One per real-valued component-owned dynamics parameter | `none` `(-inf, +inf)` | Component dynamics site |
| `dynamics_parameter_positive` | `theta+` | One per positive component-owned dynamics parameter | `positive` `(0, +inf)` | Component dynamics site |
| `residual_sd` | `sigma` | One per construct | `positive` `(0, +inf)` | Diffusion diagonal |
| `state_intercept` | `cint` | One per eligible dynamic construct when equilibrium forcing is enabled | `none` `(-inf, +inf)` | Continuous-time state intercept |
| `observation_intercept` | `manifest_mean` | One per manifest channel whose observation family requires a baseline intercept | `none` `(-inf, +inf)` | Manifest intercept vector |
| `initial_state_mean` | `t0_mean` | One per latent construct | `none` `(-inf, +inf)` | Initial-state mean vector |
| `initial_state_sd` | `t0_sd` | One per latent construct | `positive` `(0, +inf)` | Initial-state covariance diagonal |
| `static_state_sd` | `tau` | One per compiled baseline factor induced by marginalized time-invariant confounders | `positive` `(0, +inf)` | Static baseline-factor covariance |
| `loading` | `lambda` | One per non-reference indicator in multi-indicator constructs | `positive` or `negative` | Observation model |
| `measurement_error_sd` | `obs_sd` | One per free manifest measurement-error SD | `positive` `(0, +inf)` | Manifest variance diagonal |
| `observation_hyperparameter` | `obs_*` | One per active real-valued observation-family hyperparameter site | `none` `(-inf, +inf)` | Observation-family auxiliary site |
| `observation_hyperparameter_positive` | `obs_*` | One per active positive observation-family hyperparameter site | `positive` `(0, +inf)` | Observation-family auxiliary site |
| `correlation` | `cor` | One per construct-pair with marginalized confounder | `correlation` `[-1, 1]` | Diffusion covariance |

Constraint notes:

- `ar_coefficient`: model-spec elicits baseline discrete-time persistence absent feedback; [compilation](../compilation.md) binds it to the owning decay component and converts to continuous-time decay scale
- `fixed_effect`: Causal effects can be positive or negative; compiler binds each coefficient to the owning edge component or known-input effect site
- `dynamics_parameter`: Used for component-owned dynamics parameters that are not authored as interval-scale effect coefficients.
- `dynamics_parameter_positive`: Used for positive component-owned dynamics parameters such as Hill Emax and EC50.
- `static_state_sd`: Used to build low-rank initial-state covariance contributions of the form `B diag(tau^2) B^T`.
- `loading`: measurement-structure indicator polarity fixes each loading sign as either `positive` or `negative`; model-spec no longer chooses loading orientation
- `measurement_error_sd`: Surfaced only when measurement error is separately estimated (multi-indicator constructs).
- `observation_hyperparameter`: Examples include ordered-threshold bases and categorical logit offsets.
- `observation_hyperparameter_positive`: Examples include Student-t degrees of freedom, Gamma shape, and NB dispersion.

## Supported Prior Families

| Family | Signature | Support | Use When |
|---|---|---|---|
| `Normal` | `Normal(mu, sigma)` | `real` | Unconstrained effects that can be positive or negative. |
| `HalfNormal` | `HalfNormal(sigma)` | `positive` | Positive-only parameters such as standard deviations and scales. |
| `Beta` | `Beta(alpha, beta)` | `unit_interval` | Parameters constrained to the unit interval [0, 1]. |
| `Uniform` | `Uniform(lower, upper)` | `bounded` | Hard-bounded parameters when only plausible limits are known. |
| `TruncatedNormal` | `TruncatedNormal(mu, sigma, lower, upper)` | `bounded` | Bounded parameters when both a center and hard limits are meaningful. |
| `Gamma` | `Gamma(concentration, rate)` | `positive` | Positive-only parameters when right-skewed uncertainty is plausible. |
| `LogNormal` | `LogNormal(mu, sigma)` | `positive` | Positive-only parameters when uncertainty is multiplicative on the log scale. |
| `Exponential` | `Exponential(rate)` | `positive` | Positive-only parameters with mass near zero and a single decay rate. |
| `Delta` | `Delta(value)` | `positive` | Fixed positive value inserted by compiler-owned deterministic repairs. |

The `Family` values are the exact canonical strings accepted by model-spec prior schemas; aliases are not supported.
The `Use When` column is the authoritative short guidance reused by the model-spec prompts.

## Common Defaults

| Type | Typical Distribution | Typical Range | Scale |
|---|---|---|---|
| beta (causal effect) | Normal(0, 0.5) | [-2, 2] | Authored interval effect (defaults to model interval; use `reference_interval_days` when evidence is on another interval) |
| rho (AR coefficient) | Beta(2, 2) or Uniform(0, 1) | [0, 1] | Baseline discrete-time persistence absent feedback |
| sigma (residual SD) | HalfNormal(1) | [0, 5] | Data scale |
| t0_mean (initial-state mean) | Normal(0, 1) | [-3, 3] | Latent state scale; do not copy raw indicator means or log-means unless the construct is explicitly identified on that observed scale |
| t0_sd (initial-state SD) | HalfNormal(1) | [0, 3] | Latent state scale |
| lambda (loading) | HalfNormal(1) if positive, TruncatedNormal(-1, 0.5, -5, 0) if negative | [-3, 3] | Data scale with sign fixed by indicator polarity |
| obs_sd (measurement error SD) | HalfNormal(0.5) or HalfNormal(1) | [0, 3] | Manifest observation-noise scale; larger values attribute more variation to indicator noise instead of the latent state |
| obs_df (Student-t tails) | Gamma(5, 1) or LogNormal(log(5), 0.3) | [2, 30] | Observation-tail heaviness; smaller values mean heavier tails |
| obs_shape (Gamma shape) | Gamma(2, 1) | [0.5, 10] | Observation overdispersion/shape for Gamma-family emissions |
| obs_r (negative-binomial dispersion) | Gamma(2, 0.5) | [0.5, 20] | Observation overdispersion; smaller values imply heavier count overdispersion |
| obs_concentration (Beta concentration) | Gamma(5, 0.5) | [1, 50] | Observation concentration around the latent mean on (0, 1) |
| obs_ordered_base (ordered thresholds) | Normal(0, 1) | [-3, 3] | Ordered-logistic threshold location on the latent predictor scale |
| obs_ordered_gaps (ordered threshold gaps) | HalfNormal(1) | [0, 3] | Positive spacing between adjacent ordered-logistic thresholds |
| obs_cat_intercepts (categorical logits) | Normal(0, 1) | [-4, 4] | Baseline category-logit offsets on the latent predictor scale |
| obs_cat_slopes (categorical logits) | Normal(0, 1) | [-4, 4] | Category-specific slope adjustments on the latent predictor scale |
| cor (correlation) | Uniform(-1, 1) or TruncatedNormal(0, 0.3, -1, 1) | [-1, 1] | Innovation correlation |
| tau (random SD) | HalfNormal(0.5) | [0, 2] | Data scale |
