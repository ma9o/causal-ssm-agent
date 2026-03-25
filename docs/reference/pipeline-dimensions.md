# Pipeline Dimensions

This page is the cross-stage map of the system. It should explain what recurs across many stages, not re-own the artifacts that stage docs already define.

## Artifact Lineage

The main domain spine is the sequence of artifacts the pipeline produces and refines.

| Layer | Primary artifact | Produced in | Owner doc | Purpose |
|---|---|---|---|---|
| Research intent | Natural-language question | Pipeline request | [pipeline.md](../pipeline.md) | Declares the causal query |
| Theoretical causal structure | `LatentModel` | Stage 1a | [../pipeline/01a-latent-model.md](../pipeline/01a-latent-model.md) | Defines constructs, edges, and the designated outcome |
| Measurement and identification | `CausalSpec` | Stage 1b | [../pipeline/01b-measurement-identifiability.md](../pipeline/01b-measurement-identifiability.md) | Binds constructs to indicators and records identifiability |
| Observational evidence | `ObservationRecord`s and the encoded observation table (`data_for_model`) | Stage 2 | [../pipeline/02-indicator-extraction.md](../pipeline/02-indicator-extraction.md) | Converts source data into time-indexed indicator values |
| Data-quality surface | `IndicatorAudit` | Stage 3 | [../pipeline/03-extraction-validation.md](../pipeline/03-extraction-validation.md) | Describes whether extracted observations are usable |
| Functional specification | `ModelSpec` plus priors | Stage 4 | [../pipeline/04-model-specification-priors.md](../pipeline/04-model-specification-priors.md) | Chooses likelihoods, parameters, and prior beliefs |
| Parametric identification diagnostics | `ParametricIdResult` plus inference structure | Stage 4b | [../pipeline/04b-parametric-identifiability.md](../pipeline/04b-parametric-identifiability.md) | Runs conservative degrees-of-freedom, local-identification, and practical-identifiability checks |
| Variational pre-fit diagnostic | SVI diagnostics | Stage 5a | [../pipeline/05a-svi-preflight.md](../pipeline/05a-svi-preflight.md) | Lightweight approximate fit before expensive inference |
| Fitted runtime artifact | `FittedArtifact` plus diagnostics | Stage 5b | [../pipeline/05b-inference-diagnostics.md](../pipeline/05b-inference-diagnostics.md) | Holds posterior inference outputs used downstream |
| Interventional and counterfactual effect summaries | `TreatmentEffect` plus follow-up simulations | Stage 6 | [../pipeline/06-intervention-analysis.md](../pipeline/06-intervention-analysis.md) | Answers interventional (`do`) and counterfactual queries |

## Temporal Semantics

Time appears in several different places. They should not be collapsed into a single notion of granularity.

| Temporal layer | Primary owner | Meaning |
|---|---|---|
| Construct lag semantics | [Stage 1a](../pipeline/01a-latent-model.md) | How construct-to-construct effects are encoded across model-clock ticks |
| Observation window | [Stage 1b](../pipeline/01b-measurement-identifiability.md#observation-windows-and-model-clock) | The support interval over which an indicator is measured or aggregated |
| Anchor time | [Stage 2](../pipeline/02-indicator-extraction.md#observationrecord) | The timestamp attached to the extracted indicator datum |
| Inter-observation interval `dt` | [estimation.md](estimation.md) | Elapsed time used to discretize the continuous-time model |
| Intervention horizon | [Stage 6](../pipeline/06-intervention-analysis.md) | How far forward a trajectory intervention is projected |

## Execution Modality

Stages differ in how work is performed, independently of what artifact they produce.

| Modality | Meaning | Stages |
|---|---|---|
| Semantic | LLM-driven reasoning or extraction is the primary engine | 0, 1a, 1b, 4 |
| Computed | Deterministic or numerical backend logic is the primary engine | 3, 4b, 5a, 5b |
| Hybrid | Semantic and computed paths both matter | 2, 6 |

Two additional orthogonal questions matter here:

- `Interactive vs non-interactive`: 1a, 1b, 4, and 6 expose a user-facing refinement surface.
- `Single-shot vs multi-turn`: 1a and 1b are single-conversation validation loops; Stage 4 and Stage 6 are multi-turn agentic surfaces.

## Assurance Surface

The pipeline has several kinds of checks. They target different failure modes and should not be conflated.

| Assurance target | Stage | Question being answered |
|---|---|---|
| Causal identifiability | 1b | Is the treatment-to-outcome effect identified from the latent and measurement assumptions? |
| Extraction and data quality | 3 | Are the observed indicator series usable and coherent? |
| Parametric identifiability | 4b | Does the chosen parameterization pass conservative degrees-of-freedom, local-identification, and practical-identifiability checks? |
| Variational pre-fit diagnostic | 5a | Does a lightweight approximate fit immediately reveal gross pathologies? |
| Post-fit diagnostics | 5b | Does the fitted model behave well under posterior diagnostics and predictive checks? |

## Assumption Map

| Assumption | Primary owner | Main consumers | Detail page |
|---|---|---|---|
| A1. Reflective measurement model | MeasurementModel | Stages 1b, 3, 4 | [measurement-model/assumptions.md](measurement-model/assumptions.md) |
| A3. Markov property for temporal dynamics | LatentModel | Stages 1a, 4, runtime | [latent-model/assumptions.md](latent-model/assumptions.md) |
| A3a. Latent confounders have bounded temporal reach | CausalSpec identifiability | Stage 1b | [causal-spec/identifiability.md](causal-spec/identifiability.md) |
| A4. Acyclicity within time slice | LatentModel | Stages 1a, 1b | [latent-model/assumptions.md](latent-model/assumptions.md) |
| A4b. Endogenous time-varying directed effects are drift-mediated | LatentModel | Stages 1a, 4, runtime | [latent-model/assumptions.md](latent-model/assumptions.md) |
| A5. Time-invariant latents as subject-level static states | LatentModel | Stage 1a, runtime | [latent-model/assumptions.md](latent-model/assumptions.md) |
| A6. Measurement error handling depends on indicator count | MeasurementModel | Stages 1b, 4 | [measurement-model/assumptions.md](measurement-model/assumptions.md) |
| A7. Measurement model identification enables causal identification | CausalSpec identifiability | Stage 1b | [causal-spec/identifiability.md](causal-spec/identifiability.md) |
| A8. Indicator residuals are temporally independent | MeasurementModel | Stages 1b, 4, runtime | [measurement-model/assumptions.md](measurement-model/assumptions.md) |
| A9. Single-indicator constructs absorb measurement error | MeasurementModel | Stages 1b, 4 | [measurement-model/assumptions.md](measurement-model/assumptions.md) |
