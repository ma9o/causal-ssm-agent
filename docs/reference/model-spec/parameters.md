# Parameters and Priors

Defines the parameter roles, prior vocabulary, and default guidance for [`ParameterSpec`](../../pipeline/04-model-specification-priors.md#parameterspec) and [`PriorProposal`](../../pipeline/04-model-specification-priors.md) entries in a [`ModelSpec`](../../pipeline/04-model-specification-priors.md#modelspec).

> All sections below are generated from `causal_ssm_agent.distributions`.
> Edit the Python catalog and re-run `uv run python scripts/export_distribution_docs.py` instead of editing them manually.

## Parameter Roles

The [Stage 4 skeleton](../../pipeline/04-model-specification-priors.md) creates exactly the following parameters from a [`CausalSpec`](../../pipeline/01b-measurement-identifiability.md#causalspec):

| Role | Symbol | Count | Constraint | SSM location |
|---|---|---|---|---|
| `ar_coefficient` | `rho` | One per endogenous time-varying construct | `unit_interval` `[0, 1]` | Drift diagonal |
| `fixed_effect` | `beta` | One per causal edge | `none` `(-inf, +inf)` | Drift off-diagonal |
| `residual_sd` | `sigma` | One per construct | `positive` `(0, +inf)` | Diffusion diagonal |
| `static_state_sd` | `tau` | One per time-invariant endogenous construct (when needed) | `positive` `(0, +inf)` | Static-state block |
| `loading` | `lambda` | One per non-reference indicator in multi-indicator constructs | `positive` or `none` | Measurement model |
| `correlation` | `cor` | One per construct-pair with marginalized confounder | `correlation` `[-1, 1]` | Diffusion covariance |

Constraint notes:

- `ar_coefficient`: Stage 4 elicits discrete-time persistence magnitude; [compilation](../compilation.md) converts to continuous-time drift
- `fixed_effect`: Causal effects can be positive or negative; unconstrained
- `loading`: Stage 4 may enforce `positive` for sign identification or `none` when negative loadings are theoretically justified

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

The `Family` values are the exact canonical strings accepted by Stage 4 prior schemas; aliases are not supported.
The `Use When` column is the authoritative short guidance reused by the Stage 4 prompts.

## Common Defaults

| Type | Typical Distribution | Typical Range | Scale |
|---|---|---|---|
| beta (causal effect) | Normal(0, 0.5) | [-2, 2] | Discrete-time |
| rho (AR coefficient) | Beta(2, 2) or Uniform(0, 1) | [0, 1] | Discrete-time persistence |
| sigma (residual SD) | HalfNormal(1) | [0, 5] | Data scale |
| lambda (loading) | HalfNormal(1) | [0, 3] | Data scale |
| cor (correlation) | Uniform(-1, 1) or TruncatedNormal(0, 0.3, -1, 1) | [-1, 1] | Innovation correlation |
| tau (random SD) | HalfNormal(0.5) | [0, 2] | Data scale |
