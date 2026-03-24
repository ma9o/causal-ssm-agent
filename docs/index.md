# Documentation Index

## Top-Level Areas

| Family | First stop | Owns |
|---|---|---|
| Pipeline | [pipeline.md](pipeline.md) | Stage order and the authoritative stage docs under `pipeline/` |
| Reference | [reference/](reference/) | Domain objects, cross-cutting concepts, estimation, inference routing, and execution semantics |
| Guides | [guides/dev_setup.md](guides/dev_setup.md) | Operator and contributor workflows |

## Navigate by Domain Object

- [`LatentModel`](reference/latent-model/constructs-and-edges.md): construct ontology, edge legality, and lag semantics
- [`MeasurementModel`](reference/measurement-model/indicators.md): indicators, extraction semantics, windows, aggregation, and `model_clock`
- [`CausalSpec`](reference/causal-spec/identifiability.md): identifiability and the Stage 1b handoff contract
- [`ModelSpec`](reference/model-spec/parameters-likelihoods-and-priors.md): parameter roles, likelihoods, and prior elicitation

## Cross-Cutting References

- [Pipeline dimensions](reference/pipeline-dimensions.md): artifacts, assumptions, temporal semantics, scope, and assurance surfaces
- [Terminology](reference/terminology.md): SEM vs SCM naming conventions
- [Compilation](reference/compilation.md): how Stage 4 outputs become an executable SSM
- [Estimation](reference/estimation.md): CT-SDE formulation, discretization, likelihood
- [Counterfactual inference](reference/counterfactual-inference.md): do-operator, forward simulation, interpretation guidance
- [Inference routing](reference/inference-routing.md): method selection and structural routing
- [Execution and replay](reference/execution-and-replay.md): dependency DAG, resume, gating
- [Persistence and exposure](reference/persistence-and-exposure.md): internal, web, and snapshot surfaces

## Route by Question

| Question | Open first | Then |
|---|---|---|
| What does each stage do? | [pipeline.md](pipeline.md) | The relevant file under `pipeline/` |
| Which stage owns an artifact? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md#artifact-index) | The owner stage doc |
| What objects move through the pipeline? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md) | [pipeline.md](pipeline.md) |
| What does one of the four domain primitives mean? | The matching page under `reference/` | The subordinate primitive page that owns the detail |
| What should uploaded data look like? | [guides/data_contract.md](guides/data_contract.md) | [pipeline/00-ingestion.md](pipeline/00-ingestion.md) |
| Why do the docs say `latent model`, `topological structure`, and `functional specification` instead of `structural model`? | [reference/terminology.md](reference/terminology.md) | The relevant primitive or stage doc |
| How does time work across extraction, fitting, and interventions? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md#temporal-semantics) | [reference/estimation.md](reference/estimation.md) |
| What gets validated where? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md#assurance-surface) | [pipeline.md](pipeline.md) |
| How do resume, replay, overrides, and gates work? | [reference/execution-and-replay.md](reference/execution-and-replay.md) | [pipeline.md](pipeline.md) |
| What gets persisted, restored, or exposed to the web? | [reference/persistence-and-exposure.md](reference/persistence-and-exposure.md) | [pipeline.md](pipeline.md) |

**Benchmarks:**

- `../apps/data-pipeline/benchmarks/results.md` for inference method parameter recovery results
