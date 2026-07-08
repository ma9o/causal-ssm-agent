# Intervention Analysis

| Modality | Interactive | Produces |
|---|---|---|
| Hybrid | Yes | [`TreatmentEffect`](#treatmenteffect) list, interactive simulation tools |

Applies steady-state interventional-effect and trajectory-simulation semantics to the [`posterior` transition fitted model](inference.md#fittedartifact), ranks treatments by causal effect size, generates LLM commentary, and exposes three interactive tools for follow-up interventional (rung 2) and counterfactual (rung 3) queries in Pearl's ladder of causation[^pearl2019] [^pearl2009]. This is the terminal transition; interactive edits persist in place with no downstream replay.

## Inputs

| Input | Source | Description |
|---|---|---|
| `fitted_artifact` | [`posterior` transition](inference.md) | [`FittedArtifact`](inference.md#fittedartifact) with posterior samples, runtime builder, observation times, PPC result, and power-scaling result |
| `causal_design` | [`measurement_structure` transition](measurement-structure.md) | [`CausalDesign`](measurement-structure.md#causaldesign) with identifiability status, measurement structure, and outcome construct designation |
| `question` | User | Original research question for grounding the opening commentary |

`posterior` transition provided the posterior and diagnostics; `measurement_structure` transition provided the identifiability verdicts. `baseline_report` transition is the first point where posterior samples are translated into causal decision quantities.

## Process

`baseline_report` transition runs in two phases: a deterministic intervention computation that produces the baseline ranking, followed by a single LLM generation that produces opening commentary. After completion the transition exposes three interactive tools for follow-up exploration.

```mermaid
flowchart LR
    B[Baseline\nranking] --> C[LLM commentary] --> T([Interactive tools])
```

**Baseline ranking:** For each treatment that remains after the [`measurement_structure` transition identifiability screen](measurement-structure.md), the transition computes a steady-state interventional effect under `do(treatment = baseline + 1)`. For each posterior draw the baseline steady state η\* solves the vector-field root `f(η*) = 0` (numerically, via Levenberg-Marquardt); for the [affine special case](../reference/estimation.md#1-ct-sde-formulation) this reduces to η\* = −**A**⁻¹**c** for [drift matrix **A** and continuous intercept **c**](../reference/estimation.md#1-ct-sde-formulation). An intervention clamps the treatment equation and re-solves the modified system, comparing the intervened and baseline outcome values. The default `do(treatment = baseline + 1)` is vmapped over all posterior draws to produce the full posterior treatment-effect distribution.

- *Temporal forward simulation:* When temporal information is available (either from the [`model_clock`](measurement-structure.md#observation_window-and-model_clock) or the median observed timestep), the transition also runs a 30-day forward simulation for each treatment, discretizing the continuous-time system from the baseline steady state with the treatment clamped at each step. The mean trajectory across posterior draws is summarized into a [`TemporalEffect`](#temporaleffect) with 1-day, 7-day, and 30-day snapshots plus peak effect and time-to-peak.
- *Manifest-level decomposition:* When posterior draws of the [loading matrix](../reference/estimation.md#1-ct-sde-formulation) λ are available, the transition projects each treatment's outcome-level effect through the loadings to produce per-manifest effects: `manifest_effect[i] = λ[i, outcome_idx] × effect_mean`.
- *Ranking:* Treatments are sorted by |mean(`posterior_draws`)| descending.

**LLM commentary:** A single LLM generation receives the top-5 ranked effects, any diagnostic warnings, excluded non-identifiable treatments, and a summary of the follow-up capabilities. The LLM produces plain Markdown commentary for the user, persisted as `final_summary`.

**Interactive tools:** After the baseline ranking completes, `baseline_report` transition exposes three read-only tools for follow-up exploration within the same conversation.

- *`get_model_info`:* Returns a structured read-only summary of the fitted model and its diagnostics. An optional `names` filter restricts the response to specific constructs or indicators.
- *`simulate_intervention`:* Runs an interventional query on rung 2 of Pearl's ladder[^pearl2019], generalizing the baseline ranking to arbitrary intervention values, trajectory horizons, and manifest projections. Returns a posterior summary (mean, median, 95% CI, `prob_positive`) with any relevant [PPC](inference.md#ppcresult) or [power-scaling](inference.md#powerscalingresult) warnings.
- *`simulate_counterfactual`:* Runs a counterfactual query on rung 3 of Pearl's ladder[^pearl2019], conditioning on observed data before asking "what would have happened if we had intervened?" The caller specifies the evidence boundary (`start` — either an observed `time_index` or an ISO-8601 `time`, not both, defaulting to the final retained fitted latent state), the treatment and intervention mode, and the estimand (`"end_state"` or `"trajectory"` with `horizon_days` and projection level). Computation follows the standard abduction-action-prediction procedure in Pearl, Glymour, and Jewell (2016)[^pearl2016].
  - *Abduction:* recovers the latent state at the evidence boundary
  - *Forward simulation:* from the abducted state, simulates both a baseline path and a counterfactual path (treatment clamped)
  - *Output:* reports the difference as the causal effect with posterior summary, effect trajectory, and abduction warnings

### Example

For a study of agricultural practices and crop yield where `latent_structure` transition posited constructs `Irrigation Frequency`, `Soil Nitrogen`, `Pest Pressure`, and `Crop Yield`, `baseline_report` transition might rank `Soil Nitrogen` first with mean(posterior_draws)=+0.38 and a temporal peak at 12 days (`peak_effect=+0.41`), while `Pest Pressure` ranks second with mean(posterior_draws)=−0.22. Any [power-scaling](inference.md#powerscalingresult) or [PPC](inference.md#ppcresult) warnings from `posterior` transition are referenced in the LLM commentary.

## Outputs

| Output | Type | Description |
|---|---|---|
| `intervention_results` | list\[[`TreatmentEffect`](#treatmenteffect)\] | Treatments ranked by \|mean(`posterior_draws`)\| descending |
| `final_summary` | `str` \| null | LLM-generated opening commentary |

### `TreatmentEffect`

| Field | Type | Description |
|---|---|---|
| `treatment` | `str` | Construct name of the treatment |
| `posterior_draws` | `list[float]` \| null | Full posterior distribution of the treatment effect (one draw per posterior sample); consumers derive effect_size=mean(draws) and P(>0)=mean(draws>0) |
| `temporal` | [`TemporalEffect`](#temporaleffect) \| null | Forward-simulation summary at 1-day, 7-day, and 30-day horizons |
| `manifest_effects` | `dict[str, float]` \| null | Per-manifest outcome decomposition via [loading-matrix](../reference/estimation.md#1-ct-sde-formulation) projection; keys are manifest names, values are `λ[manifest, outcome] × effect_mean` |

Identifiability status, [PPC warnings](inference.md#ppcresult), and [power-scaling diagnostics](inference.md#powerscalingresult) are not duplicated here — consumers derive them from [`measurement_structure` transition](measurement-structure.md#identifiabilitystatus) and [`posterior` transition](inference.md) outputs respectively.

### `TemporalEffect`

| Field | Type | Description |
|---|---|---|
| `effect_1d` | `float` | Effect magnitude at 1 day post-intervention |
| `effect_7d` | `float` | Effect magnitude at 7 days |
| `effect_30d` | `float` | Effect magnitude at 30 days (or at the horizon boundary if shorter) |
| `peak_effect` | `float` | Maximum absolute effect reached during the trajectory |
| `time_to_peak_days` | `float` | Days from intervention onset to peak effect |

[^pearl2009]: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Bibliography entry](../reference/bibliography.md)
[^pearl2019]: Pearl, J. (2019). The Seven Tools of Causal Inference, with Reflections on Machine Learning. *Communications of the ACM*, 62(3), 54–60. [Bibliography entry](../reference/bibliography.md)
[^pearl2016]: Pearl, J., Glymour, M., & Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley. [Bibliography entry](../reference/bibliography.md)
