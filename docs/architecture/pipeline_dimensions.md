# Pipeline Dimensions

This document is the cross-cutting map of the pipeline. [pipeline_stages.md](../pipeline_stages.md) is still the canonical stage-by-stage reference, but stage order is only one way to understand the system. For short definitions of the main pipeline objects, see [artifact_glossary.md](artifact_glossary.md).

The more useful view for design, implementation, and documentation is a small set of orthogonal dimensions that recur across stages.

## 1. Artifact Lineage

The most important dimension is the sequence of domain objects the pipeline produces and refines.

| Layer | Primary artifact | Produced in | Purpose |
|---|---|---|---|
| Research intent | Natural-language question | Pipeline request | Declares the causal query |
| Theoretical causal structure | `LatentModel` | Stage 1a | Defines constructs, edges, outcome, and candidate treatments |
| Measurement + identification | `CausalSpec` | Stage 1b | Binds constructs to indicators and records identifiability |
| Observational evidence | Raw and model-ready observation rows | Stage 2 | Converts source data into time-indexed indicator values |
| Data quality surface | Indicator audits and dataset issues | Stage 3 | Describes whether extracted observations are usable |
| Functional specification | `ModelSpec` + priors | Stage 4 | Chooses likelihoods, parameters, and prior beliefs |
| Parametric recoverability | `ParametricIdResult` + inference structure | Stage 4b | Checks whether the functional specification is plausibly estimable |
| Approximate fit preflight | SVI diagnostics | Stage 5a | Cheap sanity check before expensive fitting |
| Fitted model artifact | Persisted fitted result + diagnostics | Stage 5b | Holds posterior inference outputs used downstream |
| Causal decision surface | Intervention rankings and interactive simulations | Stage 6 | Answers rung-2 and rung-3 causal questions |

This artifact lineage is the main spine of the pipeline. Most other distinctions are orthogonal metadata layered on top of it.

## 2. Temporal Semantics

Time appears in several distinct places. They should not be collapsed into a single notion of "granularity."

| Temporal layer | Lives in | Meaning |
|---|---|---|
| Construct granularity | Latent model / measurement model | The causal timescale of a construct such as hourly, daily, or weekly |
| Observation window | Measurement model | The support interval over which an indicator is measured or aggregated |
| Anchor time | Stage 2 observation rows | The timestamp assigned to the extracted indicator value |
| Inter-observation interval `dt` | Estimation runtime | The elapsed time used to discretize the continuous-time model |
| Intervention horizon | Stage 6 queries | How far forward a trajectory intervention is projected |

Important consequence: Stage 2 windowing, Stage 4 causal timescale choices, Stage 5 discretization, and Stage 6 intervention forecasts are all temporal, but they are not the same temporal decision.

See [modeling/scope.md](../modeling/scope.md) for construct granularity and cross-timescale rules, and [modeling/estimation.md](../modeling/estimation.md) for CT-to-DT discretization.

## 3. Execution Modality

Stages differ in how work is performed, independently of what artifact they produce.

| Modality | Meaning | Stages |
|---|---|---|
| Semantic | LLM-driven reasoning or extraction is the primary engine | 0, 1a, 1b, 4 |
| Computed | Deterministic or numerical backend logic is the primary engine | 3, 4b, 5a, 5b |
| Hybrid | Semantic and computed paths both matter | 2, 6 |

Two additional orthogonal questions matter here:

- `Interactive vs non-interactive`: 1a, 1b, 4, and 6 expose a user-facing refinement surface.
- `Single-shot vs multi-turn`: 1a and 1b are single-conversation validation loops; Stage 4 and Stage 6 are multi-turn agentic surfaces.

This dimension answers "how does the stage run?" not "what does the stage mean?"

## 4. Assurance Surface

The pipeline has several kinds of checks. They target different failure modes and should be documented separately.

| Assurance target | Stage | Question being answered |
|---|---|---|
| Causal identifiability | 1b | Is the target treatment -> outcome effect identified from the latent + measurement assumptions? |
| Extraction and data quality | 3 | Are the observed indicator series usable and coherent? |
| Parametric identifiability | 4b | Can the chosen functional specification plausibly recover its parameters from the available data? |
| Cheap prefit sanity | 5a | Does a fast approximate fit immediately reveal gross pathologies? |
| Post-fit diagnostics | 5b | Does the fitted model behave well under posterior diagnostics and predictive checks? |

These are all "validation" in a loose sense, but each validates a different object:

- Stage 1b validates the causal question under the topological structure.
- Stage 3 validates the extracted evidence.
- Stage 4b validates the functional specification before fitting.
- Stage 5b validates the fitted model after inference.

## 5. Control-Flow Semantics

The runtime treats stages differently with respect to replay, overrides, and gating.

| Property | Meaning | Current stages |
|---|---|---|
| `Interactive` | User can refine or follow up through the web surface | 1a, 1b, 4, 6 |
| `Override-eligible` | Pipeline can accept a user-supplied replacement payload for the stage | 1a, 1b, 4 |
| `Hard gate` | Failure can halt downstream execution unless explicitly overridden | 1b |
| `Warning-only gate` | Failure is reported but does not halt the pipeline | 4b |
| `Always recompute on resume` | Stage is intentionally not restored from checkpoint | 5a |
| `Terminal in-place persistence` | Interactive changes persist in the current stage rather than replaying downstream stages | 6 |

This dimension is implemented in the stage registry and pipeline runtime, not in the stage payload schemas alone.

See:

- [pipeline_stages.md](../pipeline_stages.md) for the stage-facing description
- [apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py) for the executable source of truth
- [apps/data-pipeline/src/causal_ssm_agent/flows/pipeline.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/pipeline.py) for replay and resume orchestration

## 6. Persistence and Exposure Boundary

Each stage has up to three distinct persistence surfaces:

| Surface | What it contains | Consumer |
|---|---|---|
| Internal stage result | Full runtime payload, including private fields and heavyweight objects | Downstream pipeline stages |
| Public web payload | JSON-serializable subset validated by stage contracts | Web app and API routes |
| Heavy artifact | Parquet or pickle sidecar files | Resume, exploration, or downstream numerical stages |

Examples:

- Stage 0 persists raw ingested data as parquet.
- Stage 2 persists both raw observation rows and model-ready encoded data as parquet.
- Stage 5b persists the fitted result as a pickle artifact.
- All stages persist validated JSON for the web layer.

This boundary matters because the web payload is not the same thing as the full runtime result. Internal fields prefixed with `_` are stripped from the public payload, while snapshots preserve the full state for resume.

See [pipeline_stages.md](../pipeline_stages.md) for the public summary and [apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py) for the concrete persistence mechanics.

## Reading Guide

Use the docs in this order depending on the question:

- "What objects flow through the pipeline?" -> this document, then [pipeline_stages.md](../pipeline_stages.md)
- "How does time work?" -> this document, then [modeling/scope.md](../modeling/scope.md) and [modeling/estimation.md](../modeling/estimation.md)
- "What gets checked where?" -> this document, then [modeling/functional_spec.md](../modeling/functional_spec.md) and [pipeline_stages.md](../pipeline_stages.md)
- "How does fitting choose an inference method?" -> [modeling/inference-strategies.md](../modeling/inference-strategies.md)
- "What gets saved, restored, or exposed to the web?" -> [runtime_semantics.md](runtime_semantics.md), then [pipeline_stages.md](../pipeline_stages.md)
- "How do replay, overrides, gates, and terminal Stage 6 persistence work?" -> [runtime_semantics.md](runtime_semantics.md)
