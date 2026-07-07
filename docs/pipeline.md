# Pipeline Overview

The authoritative definition of each pipeline artifact lives in the stage doc that introduces it. Stage order is only one view of the system — for cross-cutting lenses see [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md).

| Stage | Name | Primary artifact | Modality | Interactive | Stop condition | File |
|---|---|---|---|---|---|---|
| 0 | Agentic Data Ingestion | [`Raw dataframe`](pipeline/00-ingestion.md#raw-dataframe) | Semantic | No | None | [pipeline/00-ingestion.md](pipeline/00-ingestion.md) |
| 1a | Latent Structure Proposal | `LatentStructure` | Semantic | Yes | None | [pipeline/01a-latent-structure.md](pipeline/01a-latent-structure.md) |
| 1b | Measurement Structure and Identifiability | `CausalDesign` | Semantic | Yes | Stops if no identifiable treatments remain | [pipeline/01b-measurement-structure-identifiability.md](pipeline/01b-measurement-structure-identifiability.md) |
| 2 | Indicator Extraction | `ObservationRecord`s | Hybrid | No | Stops if no `ObservationRecord`s are extracted | [pipeline/02-indicator-extraction.md](pipeline/02-indicator-extraction.md) |
| 3 | Extraction Validation | Indicator audits | Computed | No | Stops on validation errors | [pipeline/03-extraction-validation.md](pipeline/03-extraction-validation.md) |
| 4 | Statistical Model Specification and Prior Elicitation | `StatisticalModelSpec` + priors | Semantic | Yes | None | [pipeline/04-statistical-model-specification-priors.md](pipeline/04-statistical-model-specification-priors.md) |
| 5b | Inference and Diagnostics | Fitted artifact + diagnostics | Computed | No | Stops if model fitting fails | [pipeline/05b-inference-diagnostics.md](pipeline/05b-inference-diagnostics.md) |
| 6 | Intervention Analysis | Intervention rankings + follow-up trace | Hybrid | Yes | None | [pipeline/06-intervention-analysis.md](pipeline/06-intervention-analysis.md) |
