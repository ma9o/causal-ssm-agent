# Documentation Index

## Top-Level Areas

| Family | First stop | Owns |
|---|---|---|
| Pipeline | [pipeline.md](pipeline.md) | Stage order and the authoritative stage docs under `pipeline/` |
| Reference | [reference/](reference/) | Domain objects, cross-cutting concepts, estimation, inference routing, and execution semantics |
| Guides | [guides/dev_setup.md](guides/dev_setup.md) | Operator and contributor workflows |

## Navigate by Domain Object

- [`LatentModel`](reference/latent-model/constructs-and-edges.md): construct ontology, edge legality, and lag semantics
- [`MeasurementModel`](reference/measurement-model/indicators.md): indicators, extraction semantics, support windows, aggregation, and `model_clock`
- [`CausalSpec`](reference/causal-spec/identifiability.md): identifiability and the Stage 1b handoff contract
- [`ModelSpec`](reference/model-spec/parameters-likelihoods-and-priors.md): parameter roles, likelihoods, and prior elicitation

## Cross-Cutting References

- [Pipeline dimensions](reference/pipeline-dimensions.md): artifacts, assumptions, temporal semantics, scope, and assurance surfaces
- [Compilation](reference/compilation.md): how Stage 4 outputs become an executable SSM
- [Estimation](reference/estimation.md): CT-SDE formulation, discretization, likelihood
- [Stage 6 intervention analysis](pipeline/06-intervention-analysis.md): do-operator semantics, trajectory simulation, and interpretation guidance
- [Inference routing](reference/inference-routing.md): method selection and structural routing
- [Control-flow semantics](reference/pipeline-dimensions.md#7-control-flow-semantics): dependency DAG, resume, gating
- [Persistence and exposure](reference/pipeline-dimensions.md#8-persistence-and-exposure-boundary): internal, web, and snapshot surfaces

## Terminology Note

The term `structural` is historically shared by the SEM and SCM traditions, but it points to different layers of a model.

- In SEM, the `structural model` is the part of the model that specifies directional relations among endogenous variables, in contrast to the `measurement model`.
- In SCM, `structural equations` are the assignment mechanisms `X_i = f_i(Pa_i, U_i)` for endogenous variables.

This project separates them:

| Description | Domain primitive | Owner stage |
|---|---|---|
| The latent-to-latent DAG proposed from theory | [`LatentModel`](reference/latent-model/constructs-and-edges.md) | [Stage 1a](pipeline/01a-latent-model.md) |
| The construct-to-observed mapping | [`MeasurementModel`](reference/measurement-model/indicators.md) | [Stage 1b](pipeline/01b-measurement-identifiability.md) |
| The combined latent, measurement, and identifiability handoff | [`CausalSpec`](reference/causal-spec/identifiability.md) | [Stage 1b](pipeline/01b-measurement-identifiability.md) |
| The equations, likelihoods, priors, and parameterization used for fitting | [`ModelSpec`](reference/model-spec/parameters-likelihoods-and-priors.md) | [Stage 4](pipeline/04-model-specification-priors.md) |

## Route by Question

| Question | Open first | Then |
|---|---|---|
| What does each stage do? | [pipeline.md](pipeline.md) | The relevant file under `pipeline/` |
| Which stage owns an artifact? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md#artifact-index) | The owner stage doc |
| What objects move through the pipeline? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md) | [pipeline.md](pipeline.md) |
| What does one of the four domain primitives mean? | The matching page under `reference/` | The subordinate primitive page that owns the detail |
| What should uploaded data look like? | [guides/data_contract.md](guides/data_contract.md) | [pipeline/00-ingestion.md](pipeline/00-ingestion.md) |
| Why do the docs say `latent model`, `topological structure`, and `functional specification` instead of `structural model`? | [Terminology note](#terminology-note) | The relevant primitive or stage doc |
| How does time work across extraction, fitting, and interventions? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md#temporal-semantics) | [reference/estimation.md](reference/estimation.md) |
| What gets validated where? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md#assurance-surface) | [pipeline.md](pipeline.md) |
| How do resume, replay, overrides, and gates work? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md#7-control-flow-semantics) | [pipeline.md](pipeline.md) |
| What gets persisted, restored, or exposed to the web? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md#8-persistence-and-exposure-boundary) | [pipeline.md](pipeline.md) |
