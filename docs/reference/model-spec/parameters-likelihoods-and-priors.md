# ModelSpec: Parameters, Likelihoods, and Priors

This page explains the rule-based part of Stage 4.

## Part 1: Rule-Based Specification (Guardrails)

Deterministic rules enforce modeling assumptions and constrain the space of valid models.

### 1.1 Link Functions from Indicator dtype

| `measurement_dtype` | Default distribution | Link | Alternatives |
|---|---|---|---|
| `continuous` | `gaussian` | `identity` | `student_t`, `gamma` (`log` or `inverse`), `beta` (`logit` or `probit`) |
| `binary` | `bernoulli` | `logit` | `bernoulli` with `probit` |
| `count` | `poisson` | `log` | `negative_binomial` (`log`) |
| `ordinal` | `ordered_logistic` | `cumulative_logit` | None |
| `categorical` | `categorical` | `softmax` | `ordered_logistic` (`cumulative_logit`) when categories are substantively ordered |

The default distribution is selected automatically from `measurement_dtype`. Alternative distributions for the same dtype can be specified explicitly via per-indicator entries in the `likelihoods` field of `ModelSpec`.
Likelihood-family and link names use the exact canonical enum strings shown above; Stage 4 validation does not accept aliases.

### 1.2 Temporal Structure

AR(1) is used for all endogenous time-varying constructs. See [A3](../latent-model/assumptions.md#a3-markov-property-for-temporal-dynamics).

### 1.3 Measurement Model Structure

Single-indicator constructs fix `lambda = 1`; multi-indicator constructs use factor-analysis structure with the first or reference loading fixed for identification. See [A6](../measurement-model/assumptions.md#a6-measurement-error-handling-depends-on-indicator-count) and [A9](../measurement-model/assumptions.md#a9-single-indicator-constructs-absorb-measurement-error).

### 1.4 Cross-Timescale Aggregation

When cause and effect operate at different granularities:

- Finer -> Coarser (for example hourly -> daily): aggregate the finer-grained cause using the indicator's `aggregation` field
- Coarser -> Finer (for example weekly -> daily): broadcast the coarser value to the finer time points it governs

### 1.5 Parameter Roles and Constraints

Each parameter in the SSM has a **role** meaning its function in the model and a **constraint** meaning its domain restriction. These constraints are enforced by construction; the prior family guarantees the allowed domain.

**Roles**

| Role | Symbol | Meaning | Appears in |
|---|---|---|---|
| `ar_coefficient` | `rho` | Autoregressive persistence of a latent state | Diagonal of `A` |
| `fixed_effect` | `beta` | Cross-lag causal effect between constructs | Off-diagonal of `A` |
| `residual_sd` | `sigma` | Scale of the innovation process noise | Diagonal of `G` |
| `loading` | `lambda` | Factor loading mapping latent to observed | Measurement model |
| `correlation` | `Omega` | Off-diagonal correlation between residuals | Noise covariance |

**Constraints**

| Constraint | Domain |
|---|---|
| `none` | `(-inf, +inf)` |
| `positive` | `(0, +inf)` |
| `unit_interval` | `[0, 1]` |
| `correlation` | `[-1, 1]` |

Typical prior-family guidance by constraint lives in [Supported Prior Distribution Families](./prior-distribution-families.md).

**Role -> Constraint mapping**

| Role | Default constraint | Rationale |
|---|---|---|
| `ar_coefficient` | `unit_interval` | Orchestrator elicits `rho ∈ [0, 1]` in discrete-time terms for persistence magnitude, then transforms that to the continuous-time drift diagonal via `-log(rho) / dt`. |
| `fixed_effect` | `none` | Effect sizes can be positive or negative |
| `residual_sd` | `positive` | Standard deviations are non-negative by definition |
| `loading` | `positive` or `none` | Stage 4 may enforce a positive reference loading for sign identification, while allowing unconstrained signs where negative loadings are substantively plausible |
| `correlation` | `correlation` | Bounded by definition |

## PriorProposal

`PriorProposal` is the user-facing prior object attached to one parameter in the `ModelSpec`.

It owns:

- the distribution family
- the parameterization
- provenance and reasoning
- interval metadata needed for downstream time-scale translation

Its `distribution` field uses the prior-specific vocabulary documented in [Supported Prior Distribution Families](./prior-distribution-families.md), not the observation-side `DistributionFamily` enum used by likelihoods. Those prior-family names are also exact canonical strings; aliases are not accepted.

The runtime compiler later transforms these user-facing priors into executable prior arrays, but that compilation step does not change the semantic meaning established here.
