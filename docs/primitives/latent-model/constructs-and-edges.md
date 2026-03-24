# LatentModel: Constructs and Edges

This page explains what the `LatentModel` means as a causal object. The authoritative schema remains [Stage 1a](../../pipeline/01a-latent-model.md).

## Ontology

**Constructs** are theoretical entities in the causal model such as stress, mood, cognitive load, staffing pressure, or student engagement. They live in the `LatentModel`.

**Indicators** are observed data such as HRV readings, self-report scores, cortisol levels, pull-request counts, or assignment completion rates. They live in the [MeasurementModel](../measurement-model/index.md) and reflect their parent construct via factor loadings.

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

Edge restriction: time-invariant constructs may only have time-invariant parents. A time-varying construct cannot cause a time-invariant construct, because the child is fixed within person over the modeled window.

## Edges

A `LatentModel` edge is a directed causal relation between constructs. It says which construct can affect which other construct. It does not yet say how either construct is measured.

The graph stays a DAG in the user-facing contract:

- Use explicit latent confounder nodes when theory posits an unobserved common cause.
- Do not use bidirected edges in user-facing diagrams.
- Do not introduce indicator nodes here; indicators belong to the `MeasurementModel`.

## Outcome Designation

The `LatentModel` carries exactly one designated outcome. Candidate treatments are derived later from the validated graph rather than stored as a separate artifact.

## Example

For a question about whether staffing pressure affects patient deterioration through care delays, Stage 1a may posit constructs such as `Staffing Pressure`, `Care Delay`, `Patient Severity`, and `Patient Deterioration`, plus directed edges between them and an explicit latent confounder node if an unobserved common cause is believed to exist.
