# Pipeline Overview

This is the stage-ordered map of the causal inference pipeline. Use it when the question is "what happens next?" or "which stage owns this artifact?"

Stage order is only one view of the system. The authoritative definition of each pipeline artifact lives in the stage doc that introduces it. If you know an artifact name but not its owner stage, use [concepts/artifact-index.md](concepts/artifact-index.md). If you want the domain semantics of one of the four main primitives, use [primitives/latent-model/index.md](primitives/latent-model/index.md), [primitives/measurement-model/index.md](primitives/measurement-model/index.md), [primitives/causal-spec/index.md](primitives/causal-spec/index.md), or [primitives/model-spec/index.md](primitives/model-spec/index.md). For cross-cutting domain maps, see [concepts/causal-modeling-terminology.md](concepts/causal-modeling-terminology.md), [concepts/pipeline-dimensions.md](concepts/pipeline-dimensions.md), [concepts/assumptions.md](concepts/assumptions.md), and [concepts/scope-and-timescales.md](concepts/scope-and-timescales.md). For replay, persistence, and web/internal boundaries, see [runtime/execution-and-replay.md](runtime/execution-and-replay.md) and [runtime/persistence-and-exposure.md](runtime/persistence-and-exposure.md).

## Stage Map

| Stage | Name | Primary artifact | Modality | Interactive | Gate | File |
|---|---|---|---|---|---|---|
| 0 | Agentic Data Ingestion | [`Raw dataframe`](pipeline/00-ingestion.md#raw-dataframe) | Semantic | No | None | [pipeline/00-ingestion.md](pipeline/00-ingestion.md) |
| 1a | Latent Model Proposal | `LatentModel` | Semantic | Yes | None | [pipeline/01a-latent-model.md](pipeline/01a-latent-model.md) |
| 1b | Measurement Model and Identifiability | `CausalSpec` | Semantic | Yes | Hard gate | [pipeline/01b-measurement-identifiability.md](pipeline/01b-measurement-identifiability.md) |
| 2 | Indicator Extraction | Observation rows | Hybrid | No | None | [pipeline/02-indicator-extraction.md](pipeline/02-indicator-extraction.md) |
| 3 | Extraction Validation | Indicator audits | Computed | No | None | [pipeline/03-extraction-validation.md](pipeline/03-extraction-validation.md) |
| 4 | Model Specification and Prior Elicitation | `ModelSpec` + priors | Semantic | Yes | None | [pipeline/04-model-specification-priors.md](pipeline/04-model-specification-priors.md) |
| 4b | Parametric Identifiability Diagnostics | `ParametricIdResult` + inference structure | Computed | No | Warning-only | [pipeline/04b-parametric-identifiability.md](pipeline/04b-parametric-identifiability.md) |
| 5a | SVI Preflight | SVI diagnostics | Computed | No | None | [pipeline/05a-svi-preflight.md](pipeline/05a-svi-preflight.md) |
| 5b | Inference and Diagnostics | Fitted artifact + diagnostics | Computed | No | None | [pipeline/05b-inference-diagnostics.md](pipeline/05b-inference-diagnostics.md) |
| 6 | Intervention Analysis | Intervention rankings + follow-up trace | Hybrid | Yes | None | [pipeline/06-intervention-analysis.md](pipeline/06-intervention-analysis.md) |

## Reading Guide

- "What does each stage do?" -> start here, then open the stage file.
- "What object flows between stages?" -> open the stage that introduces it, or use [concepts/artifact-index.md](concepts/artifact-index.md) to locate the owner.
- "What does a domain primitive mean beyond its schema?" -> use the matching page under `primitives/`.
- "What assumptions or timescale rules recur across stages?" -> [concepts/assumptions.md](concepts/assumptions.md) and [concepts/scope-and-timescales.md](concepts/scope-and-timescales.md).
- "Why do these docs avoid the word `structural`?" -> [concepts/causal-modeling-terminology.md](concepts/causal-modeling-terminology.md).
- "How does Stage 4 become something Stage 6 can use?" -> [model-runtime/handoff-map.md](model-runtime/handoff-map.md), then [model-runtime/compilation.md](model-runtime/compilation.md) and [model-runtime/estimation.md](model-runtime/estimation.md).
- "Why did this stage rerun or restore from disk?" -> [runtime/execution-and-replay.md](runtime/execution-and-replay.md).
- "What is persisted versus exposed to the web?" -> [runtime/persistence-and-exposure.md](runtime/persistence-and-exposure.md).

## Cross-Stage Notes

- Execution order is derived from a dependency DAG, not a hard-coded index. See [runtime/execution-and-replay.md](runtime/execution-and-replay.md).
- Artifact lineage is the main domain spine. See [concepts/pipeline-dimensions.md](concepts/pipeline-dimensions.md).
- Stage 4 begins the downstream model-runtime path. See [model-runtime/handoff-map.md](model-runtime/handoff-map.md).
- Stage 6 is interactive but terminal: follow-up edits persist in place and do not replay downstream stages. See [runtime/execution-and-replay.md](runtime/execution-and-replay.md).
