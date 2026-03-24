# LatentModel: Constructs and Edges

`LatentModel` is the domain primitive that captures the theoretical causal structure over constructs before measurement choices are made. The authoritative schema lives in [Stage 1a](../../pipeline/01a-latent-model.md).

## Ontology

**Constructs** are theoretical entities in the causal model such as stress, mood, cognitive load, staffing pressure, or student engagement. They live in the `LatentModel`.

**Indicators** are observed data such as HRV readings, self-report scores, cortisol levels, pull-request counts, or assignment completion rates. They live in the [MeasurementModel](../measurement-model/indicators.md) and reflect their parent construct via factor loadings.

The `LatentModel` therefore owns the construct-level causal graph, not the observed-variable layer.

## Construct Dimensions

Constructs are classified along two dimensions:

| Dimension | Values | Meaning |
|---|---|---|
| **Role** | Exogenous / Endogenous | Whether the construct receives causal edges from other constructs |
| **Temporal** | Time-varying / Time-invariant | Whether the construct changes within person over time |

This yields four construct types:

| Role | Temporal | AR Structure | Example |
|---|---|---|---|
| Exogenous | Time-varying | None (conditioned on) | Weather, day of week |
| Exogenous | Time-invariant | None (conditioned on) | Age, gender, person intercept |
| Endogenous | Time-varying | AR(1) | Mood, stress, sleep quality |
| Endogenous | Time-invariant | None | Baseline severity, stable trait outcome |

Edge restriction: time-invariant constructs may only have time-invariant parents. A time-varying construct cannot cause a time-invariant construct, because the child is fixed within person over the modeled window. See [A5](../latent-model/assumptions.md#a5-time-invariant-latents-as-subject-level-static-states) for the full rationale and runtime implementation (drift ≈ 0, diffusion ≈ 0).

## Temporal Semantics

### Shared Construct Timescale

All time-varying constructs currently share a single timescale set by the [`MeasurementModel` `model_clock`](../measurement-model/indicators.md#observation-windows-and-model-clock). Time-invariant constructs have no temporal granularity of their own.

### Autoregressive Structure

- Endogenous time-varying constructs receive AR(1) under [A3](assumptions.md#a3-markov-property-for-temporal-dynamics).
- Exogenous constructs do not receive AR structure; downstream models condition on their values.
- Indicators do not receive AR structure; temporal dependence in indicator series is attributed to construct dynamics, and indicator residuals are assumed iid under [A8](../measurement-model/assumptions.md#a8-indicator-residuals-are-temporally-independent).

### Edge Lag Rules

Two lag values are valid under the [Markov property](assumptions.md#a3-markov-property-for-temporal-dynamics):

- **Lag = 0:** Contemporaneous effect within the same time index. Under [A4b](assumptions.md#a4b-endogenous-time-varying-directed-effects-are-drift-mediated), this is not a valid encoding for edges between constructs that are both endogenous and time-varying.
- **Lag = 1 model-clock tick:** Lagged effect from `t - 1` to `t`.

Higher-order lags (`t - 2`, `t - 3`, and so on) are not permitted. Under Markovian dynamics, `t - 1` is a sufficient statistic for all prior history, so information from `t - 2` is already propagated through the AR(1) path.

## Edges

A `LatentModel` edge is a directed causal relation between constructs. It says which construct can affect which other construct. It does not yet say how either construct is measured.

The graph stays a DAG in the user-facing contract:

- Use explicit latent confounder nodes when theory posits an unobserved common cause.
- Do not use bidirected edges in user-facing diagrams.
- Do not introduce indicator nodes here; indicators belong to the `MeasurementModel`.

## Outcome Designation

The `LatentModel` carries exactly one designated outcome. See the [Stage 1a definition](../../pipeline/01a-latent-model.md#latent-model) for encoding and treatment derivation.

## Example

For a question about whether staffing pressure affects patient deterioration through care delays:

| Construct | Role | Temporal |
|---|---|---|
| Staffing Pressure | Exogenous | Time-varying |
| Care Delay | Endogenous | Time-varying |
| Patient Severity | Endogenous | Time-varying |
| Patient Deterioration | Endogenous | Time-varying (outcome) |
| Hospital Type | Exogenous | Time-invariant |

All edges between endogenous time-varying constructs are lagged (`t−1 → t`) so cross-construct effects enter through [drift](assumptions.md#a4b-endogenous-time-varying-directed-effects-are-drift-mediated). `Hospital Type` may have a contemporaneous edge to any endogenous construct because it is exogenous. If an unobserved common cause (such as regional healthcare funding) is believed to exist, it appears as an explicit latent confounder node with directed edges—never as a bidirected edge.
