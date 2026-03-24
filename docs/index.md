# Documentation Index

## Top-Level Areas

| Family | First stop | Owns |
|---|---|---|
| Pipeline | [pipeline.md](pipeline.md) | Stage order and the authoritative stage docs under `pipeline/` |
| Concepts | [concepts/pipeline-dimensions.md](concepts/pipeline-dimensions.md) | Cross-cutting maps, terminology, assumptions, and artifact-routing aids |
| Primitives | The matching primitive index under `primitives/` | The domain semantics of `LatentModel`, `MeasurementModel`, `CausalSpec`, and `ModelSpec` |
| Model runtime | [model-runtime/handoff-map.md](model-runtime/handoff-map.md) | How Stage 4 outputs become executable inference |
| Runtime | [runtime/execution-and-replay.md](runtime/execution-and-replay.md) | Replay, persistence, and web-exposure behavior |
| Guides | [guides/dev_setup.md](guides/dev_setup.md) | Operator and contributor workflows |

## Navigate by Primitive

- [`LatentModel`](primitives/latent-model/index.md): construct ontology, edge legality, and lag semantics
- [`MeasurementModel`](primitives/measurement-model/index.md): indicators, extraction semantics, windows, aggregation, and `model_clock`
- [`CausalSpec`](primitives/causal-spec/index.md): identifiability and the Stage 1b handoff contract
- [`ModelSpec`](primitives/model-spec/index.md): functional specification, parameter roles, likelihoods, and prior elicitation

## Route by Question

| Question | Open first | Then |
|---|---|---|
| What does each stage do? | [pipeline.md](pipeline.md) | The relevant file under `pipeline/` |
| Which stage owns an artifact? | [concepts/artifact-index.md](concepts/artifact-index.md) | The owner stage doc |
| What objects move through the pipeline? | [concepts/pipeline-dimensions.md](concepts/pipeline-dimensions.md) | [pipeline.md](pipeline.md) |
| What does one of the four domain primitives mean? | The matching page under `primitives/` | The subordinate primitive page that owns the detail |
| What should uploaded data look like? | [guides/data_contract.md](guides/data_contract.md) | [pipeline/00-ingestion.md](pipeline/00-ingestion.md) |
| Why do the docs say `latent model`, `topological structure`, and `functional specification` instead of `structural model`? | [concepts/causal-modeling-terminology.md](concepts/causal-modeling-terminology.md) | The relevant primitive or stage doc |
| How does time work across extraction, fitting, and interventions? | [concepts/scope-and-timescales.md](concepts/scope-and-timescales.md) | [concepts/pipeline-dimensions.md](concepts/pipeline-dimensions.md) and [model-runtime/estimation.md](model-runtime/estimation.md) |
| What gets validated where? | [concepts/pipeline-dimensions.md](concepts/pipeline-dimensions.md) | [pipeline.md](pipeline.md) and [primitives/model-spec/functional-specification.md](primitives/model-spec/functional-specification.md) |
| How do resume, replay, overrides, and gates work? | [runtime/execution-and-replay.md](runtime/execution-and-replay.md) | [pipeline.md](pipeline.md) |
| What gets persisted, restored, or exposed to the web? | [runtime/persistence-and-exposure.md](runtime/persistence-and-exposure.md) | [pipeline.md](pipeline.md) |

**Benchmarks:**

- `../apps/data-pipeline/benchmarks/results.md` for inference method parameter recovery results
