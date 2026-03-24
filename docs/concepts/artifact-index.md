# Artifact Index

## Core Artifacts

| Artifact | Authoritative definition | Introduced in | One-line role | Used downstream |
|---|---|---|---|---|
| Raw dataframe | [Stage 0: Raw Dataframe](../pipeline/00-ingestion.md#raw-dataframe) | Stage 0 | Normalized source dataframe with typed columns and column descriptions | 1b, 2 |
| `LatentModel` | [Stage 1a: `LatentModel`](../pipeline/01a-latent-model.md#latent-model) | Stage 1a | Theoretical causal DAG over constructs | 1b, 6 |
| Measurement model | [Stage 1b: Measurement Model](../pipeline/01b-measurement-identifiability.md#measurement-model) | Stage 1b | Mapping from constructs to observed indicators | 2, 4, 6 |
| `CausalSpec` | [Stage 1b: `CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec) | Stage 1b | Combined latent, measurement, and identifiability payload | 2, 3, 4, 6 |
| `IdentifiabilityStatus` | [Stage 1b: `IdentifiabilityStatus`](../pipeline/01b-measurement-identifiability.md#identifiabilitystatus) | Stage 1b | Treatment-level causal-identifiability result | 1b gate, 6 |
| Observation row | [Stage 2: Observation Row](../pipeline/02-indicator-extraction.md#observation-row) | Stage 2 | Canonical extracted indicator datum with support-window semantics | 3, fitting inputs |
| Model-ready data | [Stage 2: Model-Ready Data](../pipeline/02-indicator-extraction.md#model-ready-data) | Stage 2 | Encoded observation table used by fitting backends | 4, 4b, 5a, 5b, 6 |
| `IndicatorAudit` | [Stage 3: `IndicatorAudit`](../pipeline/03-extraction-validation.md#indicatoraudit) | Stage 3 | Empirical profile plus validation findings for one indicator | Stage 4, web diagnostics |
| `ModelSpec` | [Stage 4: `ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) | Stage 4 | Functional specification for fitting | 4b, compilation, 5a, 5b |
| `PriorProposal` | [Stage 4: `PriorProposal`](../pipeline/04-model-specification-priors.md#priorproposal) | Stage 4 | User-facing prior proposal for one parameter | compilation, prior predictive checks |
| `ParametricIdResult` | [Stage 4b: `ParametricIdResult`](../pipeline/04b-parametric-identifiability.md#parametricidresult) | Stage 4b | Pre-fit recoverability diagnostics | web diagnostics, model review |
| `InferenceStructureResult` | [Stage 4b: `InferenceStructureResult`](../pipeline/04b-parametric-identifiability.md#inferencestructureresult) | Stage 4b | Likelihood-path and routing summary | web diagnostics, method introspection |
| `FittedArtifact` | [Stage 5b: `FittedArtifact`](../pipeline/05b-inference-diagnostics.md#fittedartifact) | Stage 5b | Persisted fitted runtime object used downstream | 6, resume |
| `TreatmentEffect` | [Stage 6: `TreatmentEffect`](../pipeline/06-intervention-analysis.md#treatmenteffect) | Stage 6 | Ranked causal-effect summary for one treatment | web app, final intervention analysis |
| `LLMTrace` | Stage doc that emits it | Several semantic stages | Conversation and tool-call history for an interactive or agentic stage | web app, refinement history |
