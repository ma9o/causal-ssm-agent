# Supported Prior Distribution Families

This page is generated from `causal_ssm_agent.distributions.PRIOR_FAMILY_SPECS`.
Edit the Python catalog and re-run `uv run python scripts/export_distribution_docs.py` instead of editing this file manually.

## Supported Families

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

## Constraint Guidance

| Constraint | Domain | Typical prior families |
|---|---|---|
| `none` | `(-inf, +inf)` | Normal |
| `positive` | `(0, +inf)` | HalfNormal, Gamma, LogNormal, Exponential |
| `unit_interval` | `[0, 1]` | Beta, Uniform(0, 1) |
| `correlation` | `[-1, 1]` | Uniform(-1, 1), TruncatedNormal(0, sigma, -1, 1) |

## Common Parameter Defaults

| Type | Typical Distribution | Typical Range | Scale |
|---|---|---|---|
| beta (causal effect) | Normal(0, 0.5) | [-2, 2] | Discrete-time |
| rho (AR coefficient) | Beta(2, 2) or Uniform(0, 1) | [0, 1] | Discrete-time persistence |
| sigma (residual SD) | HalfNormal(1) | [0, 5] | Data scale |
| lambda (loading) | HalfNormal(1) | [0, 3] | Data scale |
| cor (correlation) | Uniform(-1, 1) or TruncatedNormal(0, 0.3, -1, 1) | [-1, 1] | Innovation correlation |
| tau (random SD) | HalfNormal(0.5) | [0, 2] | Data scale |

The `Use When` column is the authoritative short guidance reused by the Stage 4 prompts.
