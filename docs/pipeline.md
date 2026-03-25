# Pipeline Overview

The authoritative definition of each pipeline artifact lives in the stage doc that introduces it. Stage order is only one view of the system — for cross-cutting lenses see [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md).

## Stage Map

| Stage | Name | Primary artifact | Modality | Interactive | Stop condition | File |
|---|---|---|---|---|---|---|
| 0 | Agentic Data Ingestion | [`Raw dataframe`](pipeline/00-ingestion.md#raw-dataframe) | Semantic | No | None | [pipeline/00-ingestion.md](pipeline/00-ingestion.md) |
| 1a | Latent Model Proposal | `LatentModel` | Semantic | Yes | None | [pipeline/01a-latent-model.md](pipeline/01a-latent-model.md) |
| 1b | Measurement Model and Identifiability | `CausalSpec` | Semantic | Yes | Stops if no identifiable treatments remain | [pipeline/01b-measurement-identifiability.md](pipeline/01b-measurement-identifiability.md) |
| 2 | Indicator Extraction | Observation rows | Hybrid | No | Stops if no observation rows are extracted | [pipeline/02-indicator-extraction.md](pipeline/02-indicator-extraction.md) |
| 3 | Extraction Validation | Indicator audits | Computed | No | Stops on validation errors | [pipeline/03-extraction-validation.md](pipeline/03-extraction-validation.md) |
| 4 | Model Specification and Prior Elicitation | `ModelSpec` + priors | Semantic | Yes | None | [pipeline/04-model-specification-priors.md](pipeline/04-model-specification-priors.md) |
| 4b | Parametric Identifiability Diagnostics | `ParametricIdResult` + inference structure | Computed | No | None (`warn` only) | [pipeline/04b-parametric-identifiability.md](pipeline/04b-parametric-identifiability.md) |
| 5a | SVI Preflight | SVI diagnostics | Computed | No | None (`warn` only) | [pipeline/05a-svi-preflight.md](pipeline/05a-svi-preflight.md) |
| 5b | Inference and Diagnostics | Fitted artifact + diagnostics | Computed | No | Stops if model fitting fails | [pipeline/05b-inference-diagnostics.md](pipeline/05b-inference-diagnostics.md) |
| 6 | Intervention Analysis | Intervention rankings + follow-up trace | Hybrid | Yes | None | [pipeline/06-intervention-analysis.md](pipeline/06-intervention-analysis.md) |

## Cross-Stage Notes

- Execution order is derived from a dependency DAG, not a hard-coded index. See [reference/execution-semantics.md](reference/execution-semantics.md#1-control-flow-semantics).
- Artifact lineage is the main domain spine. See [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md).
- Stage 4 begins the downstream model-runtime path. See [reference/compilation.md](reference/compilation.md).
- Stage 6 is interactive but terminal: follow-up edits persist in place and do not replay downstream stages. See [reference/execution-semantics.md](reference/execution-semantics.md#1-control-flow-semantics).
