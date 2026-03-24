# Supported Prior Distribution Families

This page is generated from `causal_ssm_agent.distributions.PRIOR_FAMILY_SPECS`.
Edit the Python catalog and re-run `uv run python scripts/export_distribution_docs.py` instead of editing this file manually.

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

The `Use When` column is the authoritative short guidance reused by the Stage 4 prompts.
