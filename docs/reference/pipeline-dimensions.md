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
| Fitted runtime artifact | `FittedArtifact` plus diagnostics | Stage 5b | [../pipeline/05b-inference-diagnostics.md](../pipeline/05b-inference-diagnostics.md) | Holds posterior inference outputs used downstream |
| Interventional and counterfactual effect summaries | `TreatmentEffect` plus follow-up simulations | Stage 6 | [../pipeline/06-intervention-analysis.md](../pipeline/06-intervention-analysis.md) | Answers interventional (`do`) and counterfactual queries |

## Temporal Semantics

Time appears in five distinct roles across the pipeline. They answer different questions and should not be collapsed into a single notion of granularity.

| Concept | Primary owner | What it answers |
|---|---|---|
| **`model_clock`** | [Stage 1b](../pipeline/01b-measurement-identifiability.md#observation_window-and-model_clock) | What is the shared tick width used for extraction, discretization, and the default lag unit? A global setting (e.g. `"1d"`) that aligns all indicators onto a common grid. |
| **`observation_window`** | [Stage 1b](../pipeline/01b-measurement-identifiability.md#observation_window-and-model_clock) | Over what support interval is a single indicator value measured or aggregated? May differ per indicator (e.g. daily mood vs. weekly incident count) as long as windows align back onto the `model_clock`. |
| **`anchor_time`** | [Stage 2](../pipeline/02-indicator-extraction.md#observationrecord) | Which timestamp attaches the extracted value to the latent grid? Derived from the indicator's [`anchor_policy`](../pipeline/01b-measurement-identifiability.md#derived-observation-semantics) — usually `support_end` for interval summaries, `support_start` for `first`. |
| **`dt`** | [estimation.md](estimation.md#2-discretization-ct-to-dt) | What is the elapsed time between consecutive observations used to discretize the continuous-time SDE? Computed from successive `anchor_time` values; drives `A_d = exp(A·dt)` and the discrete process noise. |
| **Intervention horizon** | [Stage 6](../pipeline/06-intervention-analysis.md) | How far forward is a trajectory intervention projected? Default 30 days, discretized at the `model_clock` step, yielding snapshots at 1 d, 7 d, and 30 d plus peak effect and time-to-peak. |

### Worked example: one observation through the pipeline

Consider a study with `model_clock = "1d"` and an indicator *daily mean mood* (`aggregation = mean`, `observation_window = "1d"`).

| Stage | What happens | Temporal artifact |
|---|---|---|
| **1b** | The measurement model declares `aggregation = mean` → `support_kind = interval`, `anchor_policy = support_end`. | `observation_window = "1d"` committed |
| **2** | The extractor averages mood values from 2025-03-01 00:00 to 2025-03-02 00:00, producing value 6.2. | `ObservationRecord(anchor_time = 2025-03-02, support_start = 2025-03-01, support_end = 2025-03-02)` |
| **5** | The previous observation was anchored at 2025-03-01; the estimator computes `dt = 1.0 day` and discretizes: `A_d = exp(A · 1.0)`. | `dt = 1.0` day feeds the Kalman/PF step |
| **6** | After fitting, an intervention `do(exercise = baseline+1)` is simulated forward 30 days at 1-day steps from the baseline steady state, producing `TemporalEffect(effect_1d, effect_7d, effect_30d, peak_effect, time_to_peak_days)`. | Horizon = 30 d at `model_clock` resolution |

The key invariant: `model_clock` sets the resolution; `observation_window` says how much real-world time each datum summarizes; `anchor_time` places it on the grid; `dt` discretizes the SDE between grid points; the intervention horizon projects the fitted model forward on that same grid.

## Assurance Surface

The pipeline has several kinds of checks. They target different failure modes and should not be conflated.

| Assurance target | Stage | Question being answered |
|---|---|---|
| Causal identifiability | 1b | Is the treatment-to-outcome effect identified from the latent and measurement assumptions? |
| Extraction and data quality | 3 | Are the observed indicator series usable and coherent? |
| Parametric identifiability | 4b | Does the chosen parameterization pass conservative degrees-of-freedom, local-identification, and practical-identifiability checks? |
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
