# Artifact Glossary

This glossary names the main artifacts that move through the pipeline. It is the shortest path for answering "what is this object?" before reading the stage or modeling docs.

For the cross-cutting pipeline map, see [pipeline_dimensions.md](pipeline_dimensions.md). For execution and persistence behavior, see [runtime_semantics.md](runtime_semantics.md).

## Core Pipeline Artifacts

| Artifact | Produced in | Meaning | Consumed by |
|---|---|---|---|
| Research question | Pipeline request | Natural-language causal query that anchors the run | 1a, 1b, 2, 4, 6 |
| `LatentModel` | Stage 1a | Theoretical causal DAG over constructs | 1b, 6 |
| Measurement model | Stage 1b | Mapping from constructs to observed indicators, including extraction and aggregation semantics | 1b, 2, 4, 6 |
| `CausalSpec` | Stage 1b | Combined latent model, measurement model, and identifiability status | 2, 3, 4, compilation, 6 |
| Observation row | Stage 2 | Canonical extracted indicator datum with indicator name, value, anchor time, and support window | 3, fitting inputs |
| Model-ready data | Stage 2 | Encoded observation table used by fitting backends | 4, 4b, 5a, 5b, 6 |
| `ModelSpec` | Stage 4 | Functional specification of parameters, roles, and likelihood choices | 4b, compilation, 5a, 5b |
| `PriorProposal` | Stage 4 | User-facing prior specification for one parameter | compilation, prior predictive checks |
| `ParametricIdResult` | Stage 4b | Pre-fit recoverability diagnostics for the chosen functional specification | web diagnostics, model review |
| `InferenceStructureResult` | Stage 4b | Likelihood-path and routing summary for the compiled model | web diagnostics, method introspection |
| Fitted artifact | Stage 5b | Persisted runtime object holding fitted model state and post-fit diagnostics | 6, resume |
| `TreatmentEffect` | Stage 6 | Ranked causal effect summary for one treatment | web app, final intervention analysis |
| `LLMTrace` | Several semantic stages | Conversation and tool-call trace for an interactive or agentic stage | web app, refinement history |

## Closely Related Terms

### `LatentModel` vs measurement model

- `LatentModel` is the theoretical causal graph over constructs.
- The measurement model says how those constructs are observed in data.

The first answers "what causes what?" The second answers "how do we see it?"

### `CausalSpec` vs `ModelSpec`

- `CausalSpec` is still at the causal-and-measurement level.
- `ModelSpec` is the statistical implementation choice used for fitting.

`CausalSpec` says which constructs, indicators, and treatment-outcome relationships are in play. `ModelSpec` says which distributions, parameters, constraints, and priors realize that model in the estimation backend.

### Observation rows vs model-ready data

- Observation rows are canonical extracted facts with explicit support-window semantics.
- Model-ready data is the encoded numerical form used by the SSM runtime.

The former is closer to the extraction contract; the latter is closer to the fitting contract.

### Fitted artifact vs web payload

- The fitted artifact is the heavyweight internal runtime object persisted for Stage 5b and used by Stage 6.
- The web payload is the contract-validated JSON projection exposed to the frontend.

They are related, but they are not interchangeable.

## Reading Guide

Use this document when you need to disambiguate names quickly:

- "What is `CausalSpec`?" -> this document, then [../pipeline_stages.md](../pipeline_stages.md)
- "How is `ModelSpec` different from the causal model?" -> this document, then [../modeling/functional_spec.md](../modeling/functional_spec.md)
- "What object does compilation consume?" -> this document, then [../modeling/compilation.md](../modeling/compilation.md)
- "What is persisted vs exposed to the web?" -> this document, then [runtime_semantics.md](runtime_semantics.md)
