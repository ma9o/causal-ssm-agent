# Pipeline Dimensions

The more useful view for design, implementation, and documentation is a small set of orthogonal dimensions that recur across stages.

## 1. Artifact Lineage

The most important dimension is the sequence of domain objects the pipeline produces and refines.

| Layer | Primary artifact | Produced in | Purpose |
|---|---|---|---|
| Research intent | Natural-language question | Pipeline request | Declares the causal query |
| Theoretical causal structure | [`LatentModel`](../pipeline/01a-latent-model.md#latentmodel) | Stage 1a | Defines constructs, edges, and the designated outcome; candidate treatments are derived from the graph |
| Measurement + identification | [`CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec) | Stage 1b | Binds constructs to indicators and records identifiability |
| Observational evidence | Raw and model-ready observation rows | Stage 2 | Converts source data into time-indexed indicator values |
| Data quality surface | Indicator audits and dataset issues | Stage 3 | Describes whether extracted observations are usable |
| Functional specification | [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) + priors | Stage 4 | Chooses likelihoods, parameters, and prior beliefs |
| Parametric recoverability | [`ParametricIdResult`](../pipeline/04b-parametric-identifiability.md#parametricidresult) + inference structure | Stage 4b | Checks whether the functional specification is plausibly estimable |
| Approximate fit preflight | SVI diagnostics | Stage 5a | Cheap sanity check before expensive fitting |
| Fitted model artifact | Persisted fitted result + diagnostics | Stage 5b | Holds posterior inference outputs used downstream |
| Causal decision surface | Intervention rankings and interactive simulations | Stage 6 | Answers rung-2 and rung-3 causal questions |

This artifact lineage is the main spine of the pipeline. Most other distinctions are orthogonal metadata layered on top of it.

### Artifact Index

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

## 2. Scope Boundary

In scope:

- time-varying constructs with optional time-invariant covariates
- explicit measurement definitions for every construct
- causal reasoning that can stop at structure when numeric identification is not justified

Out of scope:

- trajectory estimation for unmeasured constructs; every construct must have at least one indicator
- user-facing bidirected-edge representations instead of explicit latent confounder nodes

Latent state filtering is used internally for likelihood computation, but the framework's outputs are causal effect estimates rather than state-trajectory products.

## 3. Temporal Semantics

Time appears in several distinct places. They should not be collapsed into a single notion of "granularity."

| Temporal layer | Lives in | Meaning |
|---|---|---|
| Construct granularity | Latent model / measurement model | The causal timescale of a construct such as hourly, daily, or weekly |
| Observation window | Measurement model | The support interval over which an indicator is measured or aggregated |
| Anchor time | Stage 2 observation rows | The timestamp assigned to the extracted indicator value |
| Inter-observation interval `dt` | Estimation runtime | The elapsed time used to discretize the continuous-time model |
| Intervention horizon | Stage 6 queries | How far forward a trajectory intervention is projected |

Important consequence: Stage 2 windowing, Stage 4 causal timescale choices, Stage 5 discretization, and Stage 6 intervention forecasts are all temporal, but they are not the same temporal decision.

The detailed time semantics live with the primitives that own them. Identifiability is checked by y0 in Stage 1b rather than enforced at the schema level.

| Question | Primary owner | Detail page |
|---|---|---|
| What is a construct, and which edges are legal between constructs? | LatentModel | [constructs-and-edges.md](latent-model/constructs-and-edges.md) |
| How do lag rules work at the construct level? | LatentModel | [constructs-and-edges.md#temporal-semantics](latent-model/constructs-and-edges.md#temporal-semantics) |
| How do indicators define support windows, aggregation, and `model_clock`? | MeasurementModel | [indicators.md#observation-windows-and-model-clock](measurement-model/indicators.md#observation-windows-and-model-clock) |
| How does temporal unrolling affect causal identification? | CausalSpec | [identifiability.md](causal-spec/identifiability.md) |
| How is elapsed `dt` used in continuous-to-discrete runtime transitions? | Runtime estimation | [estimation.md](estimation.md) |

See [estimation.md](estimation.md) for CT-to-DT discretization.

## 4. Execution Modality

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

## 5. Assurance Surface

The pipeline has several kinds of checks. They target different failure modes and should be documented separately.

| Assurance target | Stage | Question being answered |
|---|---|---|
| Causal identifiability | 1b | Is the target treatment -> outcome effect identified from the latent + measurement assumptions? |
| Extraction and data quality | 3 | Are the observed indicator series usable and coherent? |
| Parametric identifiability | 4b | Can the chosen functional specification plausibly recover its parameters from the available data? |
| Cheap prefit sanity | 5a | Does a fast approximate fit immediately reveal gross pathologies? |
| Post-fit diagnostics | 5b | Does the fitted model behave well under posterior diagnostics and predictive checks? |

## 6. Assumption Map

| Assumption | Primary owner primitive | Main consumers | Detail page |
|---|---|---|---|
| A1. Reflective measurement model | MeasurementModel | Stages 1b, 4 | [measurement-model/assumptions.md](measurement-model/assumptions.md) |
| A3. Markov property for temporal dynamics | LatentModel | Stages 1a, 4, runtime | [latent-model/assumptions.md](latent-model/assumptions.md) |
| A3a. Latent confounders have bounded temporal reach | CausalSpec | Stage 1b | [causal-spec/identifiability.md](causal-spec/identifiability.md) |
| A4. Acyclicity within time slice | LatentModel | Stages 1a, 1b | [latent-model/assumptions.md](latent-model/assumptions.md) |
| A4b. Endogenous time-varying directed effects are drift-mediated | LatentModel | Stages 1a, 4, runtime | [latent-model/assumptions.md](latent-model/assumptions.md) |
| A5. Time-invariant latents as subject-level static states | LatentModel | Stage 1a, runtime | [latent-model/assumptions.md](latent-model/assumptions.md) |
| A6. Measurement error handling depends on indicator count | MeasurementModel | Stages 1b, 4 | [measurement-model/assumptions.md](measurement-model/assumptions.md) |
| A7. Measurement model identification enables causal identification | CausalSpec | Stage 1b | [causal-spec/identifiability.md](causal-spec/identifiability.md) |
| A8. Indicator residuals are temporally independent | MeasurementModel | Stages 1b, 4, runtime | [measurement-model/assumptions.md](measurement-model/assumptions.md) |
| A9. Single-indicator constructs absorb measurement error | MeasurementModel | Stages 1b, 4 | [measurement-model/assumptions.md](measurement-model/assumptions.md) |

<!-- A2 is intentionally absent. It was removed during an early revision; numbering is kept stable to avoid breaking cross-references in code and other docs. -->

## 7. Control-Flow Semantics

Execution order is not hard-coded. It is derived from a dependency DAG declared in the stage registry, where each stage declares `stage_id`, `depends_on`, `contract`, `bind_inputs`, `runner`, optional gate behavior, and optional restore/persist/finalize behavior through a materializer. The runtime computes a topological order from `depends_on` and folds over that order.

| Property | Meaning | Current stages |
|---|---|---|
| `Interactive` | User can refine or follow up through the web surface | 1a, 1b, 4, 6 |
| `Override-eligible` | Pipeline can accept a user-supplied replacement payload for the stage | 1a, 1b, 4 |
| `Hard gate` | Failure can halt downstream execution unless explicitly overridden | 1b |
| `Warning-only gate` | Failure is reported but does not halt the pipeline | 4b |
| `Always recompute on resume` | Stage is intentionally not restored from checkpoint | 5a |
| `Terminal in-place persistence` | Interactive changes persist in the current stage rather than replaying downstream stages | 6 |

### Resume Semantics

Resume restores earlier dependencies and re-executes only the requested window:

1. Resolve `start_stage` and `end_stage`.
2. Restore earlier dependencies from snapshot or reconstructed artifacts.
3. Execute only stages inside the requested window.
4. Persist fresh web payloads and snapshots for stages that reran.

Important cases: most stages restore normally from persisted state; Stage 5a is intentionally never restored; Stages 0, 2, 4, 4b, and 5b use artifact-backed restore logic.

### Question and Context Resolution

The natural-language research question is materialized to `data/{workspace_id}/query.txt`. This lets fresh runs start from web-submitted text while resume runs can reload the same question without resubmission.

### Sources

- [../pipeline.md](../pipeline.md) for the stage-facing description
- [apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py) for the executable source of truth
- [apps/data-pipeline/src/causal_ssm_agent/flows/pipeline.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/pipeline.py) for replay and resume orchestration

## 8. Persistence and Exposure Boundary

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

See [../pipeline.md](../pipeline.md) for the public summary and [apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py) for the concrete persistence mechanics.
