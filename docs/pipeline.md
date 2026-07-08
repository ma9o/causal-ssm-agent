# Pipeline Overview

The authoritative definition of each pipeline artifact lives in the artifact
doc that introduces it. Execution order is a property of the artifact DAG served
by `GET /api/machine`; for cross-cutting lenses see
[reference/pipeline-dimensions.md](reference/pipeline-dimensions.md).

| Run artifact / derived view | Name | Primary artifact | Modality | Interactive | Stop condition | File |
|---|---|---|---|---|---|---|
| `raw_data` | Agentic Data Ingestion | [`Raw dataframe`](pipeline/ingestion.md#raw-dataframe) | Semantic | No | None | [pipeline/ingestion.md](pipeline/ingestion.md) |
| `latent_structure` | Latent Structure Proposal | `LatentStructure` | Semantic | Yes | None | [pipeline/latent-structure.md](pipeline/latent-structure.md) |
| `measurement_structure` | Measurement Structure and Identifiability | `CausalDesign` | Semantic | Yes | Stops if no identifiable treatments remain | [pipeline/measurement-structure.md](pipeline/measurement-structure.md) |
| `measurements` | Indicator Extraction | `ObservationRecord`s | Hybrid | No | Stops if no `ObservationRecord`s are extracted | [pipeline/extraction.md](pipeline/extraction.md) |
| `validation_report` | Extraction Validation | Indicator audits | Computed | No | Stops on validation errors | [pipeline/extraction-validation.md](pipeline/extraction-validation.md) |
| `statistical_model_spec` | Statistical Model Specification and Prior Elicitation | `StatisticalModelSpec` + priors | Semantic | Yes | None | [pipeline/statistical-model-spec.md](pipeline/statistical-model-spec.md) |
| `posterior` | Inference and Diagnostics | Fitted artifact + diagnostics | Computed | No | Stops if model fitting fails | [pipeline/inference.md](pipeline/inference.md) |
| `baseline_report` | Intervention Analysis | Intervention rankings + follow-up trace | Hybrid | Yes | None | [pipeline/analysis.md](pipeline/analysis.md) |
