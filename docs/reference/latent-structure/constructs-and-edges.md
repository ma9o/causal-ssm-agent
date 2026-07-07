# LatentStructure: Constructs and Edges

This reference deepens the construct and edge semantics used by [Stage 1a](../../pipeline/01a-latent-structure.md). Stage 1a owns the emitted `LatentStructure` contract; this page focuses on ontology, edge legality, and lag semantics.

## Ontology

**Constructs** are theoretical entities in the causal model such as stress, mood, cognitive load, staffing pressure, or student engagement. They live in the `LatentStructure`.

**Indicators** are observed data such as HRV readings, self-report scores, cortisol levels, pull-request counts, or assignment completion rates. They live in the [Stage 1b measurement structure](../../pipeline/01b-measurement-structure-identifiability.md#measurementstructure) and reflect their parent construct via factor loadings.

The `LatentStructure` therefore lives at the construct layer, not the observed-variable layer.

## Construct Dimensions

Constructs are classified along two dimensions:

| Dimension | Values | Meaning |
|---|---|---|
| **Role** | Exogenous / Endogenous | Whether the construct receives causal edges from other constructs |
| **Temporal** | Time-varying / Time-invariant | Whether the construct changes within person over time |

This yields four construct types:

| Role | Temporal | Example |
|---|---|---|
| Exogenous | Time-varying | Weather, day of week |
| Exogenous | Time-invariant | Age, gender, person intercept |
| Endogenous | Time-varying | Mood, stress, sleep quality |
| Endogenous | Time-invariant | Baseline severity, stable trait outcome |

AR structure and edge restrictions follow from the [latent-structure assumptions](assumptions.md): endogenous time-varying constructs receive [AR(1)](assumptions.md#a3-markov-property-for-temporal-dynamics), and time-invariant constructs may only have [time-invariant parents](assumptions.md#a5-time-invariant-latents-as-subject-level-static-states).

## Temporal Semantics

### Shared Construct Timescale

All time-varying constructs currently share a single timescale set by the [Stage 1b `model_clock`](../../pipeline/01b-measurement-structure-identifiability.md#observation_window-and-model_clock). Time-invariant constructs have no temporal granularity of their own.

### Edge Lag Rules

Two lag values are valid under the [Markov property (A3)](assumptions.md#a3-markov-property-for-temporal-dynamics):

- **Lag = 0:** Contemporaneous effect within the same time index. Under [A4b](assumptions.md#a4b-endogenous-time-varying-directed-effects-are-drift-mediated), not valid for edges between constructs that are both endogenous and time-varying.
- **Lag = 1 model-clock tick:** Lagged effect from `t - 1` to `t`. Higher-order lags are not permitted because [A3](assumptions.md#a3-markov-property-for-temporal-dynamics) makes `t - 1` a sufficient statistic for all prior history.

## Edges

A `LatentStructure` edge is a directed causal relation between constructs. It says which construct can affect which other construct. It does not yet say how either construct is measured.

The graph stays a DAG in the user-facing contract:

- Use explicit latent confounder nodes when theory posits an unobserved common cause.
- Do not use bidirected edges in user-facing diagrams.
- Do not introduce indicator nodes here; indicators belong to the `MeasurementStructure`.

## Outcome Designation

The `LatentStructure` carries exactly one designated outcome. See the [Stage 1a definition](../../pipeline/01a-latent-structure.md#latent-structure) for encoding and treatment derivation.

## Example

For a question about whether staffing pressure affects patient deterioration through care delays:

| Construct | Role | Temporal |
|---|---|---|
| Staffing Pressure | Exogenous | Time-varying |
| Care Delay | Endogenous | Time-varying |
| Patient Severity | Endogenous | Time-varying |
| Patient Deterioration | Endogenous | Time-varying (outcome) |
| Hospital Type | Exogenous | Time-invariant |

All edges between endogenous time-varying constructs are lagged (`t−1 → t`) so cross-construct effects enter through [drift](assumptions.md#a4b-endogenous-time-varying-directed-effects-are-drift-mediated). `Hospital Type` may have a contemporaneous edge to any endogenous construct because it is exogenous. If an unobserved common cause (such as regional healthcare funding) is believed to exist, it appears as an explicit latent confounder node with directed edges to the affected constructs.
