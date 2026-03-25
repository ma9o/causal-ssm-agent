# Parameters

Defines the parameter roles, enumeration rules, and constraints that govern [`ParameterSpec`](../../pipeline/04-model-specification-priors.md#parameterspec) entries in a [`ModelSpec`](../../pipeline/04-model-specification-priors.md#modelspec).

## Parameter Roles

| Role | Symbol | Meaning | Appears in |
|---|---|---|---|
| `ar_coefficient` | `rho` | Autoregressive persistence of a latent state | Drift diagonal |
| `fixed_effect` | `beta` | Cross-lag causal effect between constructs | Drift off-diagonal |
| `residual_sd` | `sigma` | Innovation process scale | Diffusion diagonal |
| `static_state_sd` | `tau` | Quasi-constant latent-state variation | Static-state block |
| `loading` | `lambda` | Factor loading mapping latent to observed | Measurement model |
| `correlation` | `cor` | Off-diagonal residual correlation between latent innovations | Diffusion covariance |

## Enumeration Rules

The [Stage 4 skeleton](../../pipeline/04-model-specification-priors.md) creates exactly the following parameters from a [`CausalSpec`](../../pipeline/01b-measurement-identifiability.md#causalspec):

| Role | Count | Scope |
|---|---|---|
| `ar_coefficient` | One per endogenous time-varying construct | Persistence of each latent state across time steps |
| `fixed_effect` | One per causal edge | Directed effect between two constructs |
| `residual_sd` | One per construct | Innovation noise scale of each latent state |
| `loading` | One per non-reference indicator in multi-indicator constructs | Mapping strength from latent to observed; the first or reference indicator has its loading fixed to 1 for identification |
| `static_state_sd` | One per time-invariant endogenous construct (when needed) | Quasi-constant variation of a subject-level static state |
| `correlation` | One per pair of constructs whose shared confounder was marginalized at identifiability time | Off-diagonal residual correlation between latent innovations |

## Role-to-Constraint Mapping

| Role | Default constraint | Domain | Rationale |
|---|---|---|---|
| `ar_coefficient` | `unit_interval` | `[0, 1]` | Stage 4 elicits discrete-time persistence magnitude; compilation converts to continuous-time drift |
| `fixed_effect` | `none` | `(-inf, +inf)` | Causal effects can be positive or negative |
| `residual_sd` | `positive` | `(0, +inf)` | Standard deviations are non-negative |
| `static_state_sd` | `positive` | `(0, +inf)` | Static-state scales are non-negative |
| `loading` | `positive` or `none` | `(0, +inf)` or `(-inf, +inf)` | Stage 4 may enforce sign identification while allowing substantively negative loadings when justified |
| `correlation` | `correlation` | `[-1, 1]` | Correlations are bounded by definition |

Typical prior-family guidance by constraint lives in [Supported Prior Distribution Families](prior-distribution-families.md).
