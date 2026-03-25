# Stage 6: Intervention Analysis

| Modality | Interactive | Produces |
|---|---|---|
| Hybrid | Yes | [`TreatmentEffect`](#treatmenteffect) list, interactive simulation tools |

Applies steady-state and trajectory intervention semantics to the [Stage 5b fitted model](05b-inference-diagnostics.md#fittedartifact), ranks treatments by causal effect size, generates LLM commentary, and exposes three interactive tools for follow-up rung-2 and rung-3 simulations. This is the terminal stage—interactive edits persist in place with no downstream replay.

## Inputs

| Input | Source | Description |
|---|---|---|
| `fitted_artifact` | [Stage 5b](05b-inference-diagnostics.md) | [`FittedArtifact`](05b-inference-diagnostics.md#fittedartifact) with posterior samples, runtime builder, observation times, PPC result, and power-scaling result |
| `latent_model` | [Stage 1a](01a-latent-model.md) | [`LatentModel`](01a-latent-model.md#latentmodel) from which the outcome construct name is derived |
| `causal_spec` | [Stage 1b](01b-measurement-identifiability.md) | [`CausalSpec`](01b-measurement-identifiability.md#causalspec) with identifiability status and measurement model |
| `question` | User | Original research question for grounding the opening commentary |

Stage 5b provided the posterior and diagnostics; Stage 1b provided the identifiability verdicts. Stage 6 is the first point where posterior samples are translated into causal decision quantities.

## Process

Stage 6 runs in two phases: a deterministic intervention computation that produces the baseline ranking, followed by a single LLM generation that produces opening commentary. After completion the stage exposes three interactive tools for follow-up exploration.

```mermaid
flowchart LR
    B[Baseline\nranking] --> A[Artifact\nassembly] --> C[LLM commentary] --> T([Interactive tools])
```

**Baseline intervention ranking:** For each treatment that remains after the [Stage 1b identifiability filter](01b-measurement-identifiability.md), the stage computes the steady-state do-operator. For a draw with [drift matrix **A** and continuous intercept **c**](../reference/estimation.md#1-ct-sde-formulation), the baseline steady state is η\* = −**A**⁻¹**c**; an intervention clamps the treatment equation, solves the modified linear system, and compares the intervened and baseline outcome values. The default `do(treatment = baseline + 1)` is vmapped over all posterior draws to produce the full posterior treatment-effect distribution.

**Temporal forward simulation:** When temporal information is available (either from the [`model_clock`](01b-measurement-identifiability.md#observation_window-and-model_clock) or the median observed timestep), the stage also runs a 30-day forward simulation for each treatment. This discretizes the continuous-time system, starts from the baseline steady state, clamps the treatment at each step, and records the outcome trajectory. The mean trajectory across posterior draws is summarized into a [`TemporalEffect`](#temporaleffect) with 1-day, 7-day, and 30-day snapshots plus peak effect and time-to-peak.

**Manifest-level decomposition:** When posterior draws of the [loading matrix](../reference/estimation.md#1-ct-sde-formulation) λ are available, the stage projects each treatment's outcome-level effect through the loadings to produce per-manifest effects: `manifest_effect[i] = λ[i, outcome_idx] × effect_mean`.

**Ranking:** Treatments are sorted by |`effect_size`| descending.

**LLM commentary:** A single LLM generation receives the top-5 ranked effects, any diagnostic warnings, excluded non-identifiable treatments, and a summary of the follow-up capabilities. The LLM produces plain Markdown commentary for the user, persisted as `final_summary`.

**Interactive tools:** After the baseline ranking completes, Stage 6 exposes three read-only tools for follow-up exploration within the same conversation.

**`get_model_info`:** Returns a structured read-only summary of the fitted model and its diagnostics. An optional `names` filter restricts the response to specific constructs or indicators.

**`simulate_intervention`:** Runs a Pearl rung-2 interventional simulation, generalizing the baseline ranking to arbitrary intervention values, trajectory horizons, and manifest projections. Returns a posterior summary (mean, median, 95% CI, `prob_positive`) with any relevant [PPC](05b-inference-diagnostics.md#ppcresult) or [power-scaling](05b-inference-diagnostics.md#powerscalingresult) warnings.

**`simulate_counterfactual`:** Runs a Pearl rung-3 counterfactual forecast—conditioning on observed data before asking "what would have happened if we had intervened?" The caller specifies an evidence window (optional ISO-8601 `start_time`/`end_time` bounds, defaulting to the full observed range), the treatment and intervention mode, and the estimand (`"end_state"` or `"trajectory"` with `horizon_days` and projection level).

- *Abduction*: recovers the latent state at the evidence boundary
- *Forward simulation*: from the abducted state, simulates both a baseline path and a counterfactual path (treatment clamped)
- *Output*: reports the difference as the causal effect with posterior summary, effect trajectory, and abduction warnings

### Example

For a study of agricultural practices and crop yield where Stage 1a posited constructs `Irrigation Frequency`, `Soil Nitrogen`, `Pest Pressure`, and `Crop Yield`, Stage 6 might rank `Soil Nitrogen` first with `effect_size=+0.38`, `prob_positive=0.96`, and a temporal peak at 12 days (`peak_effect=+0.41`), while `Pest Pressure` ranks second with `effect_size=−0.22` and a prior-sensitivity warning because its drift cross-lag was classified as `prior_dominated`.

## Outputs

| Output | Type | Description |
|---|---|---|
| `intervention_results` | list\[[`TreatmentEffect`](#treatmenteffect)\] | Treatments ranked by \|`effect_size`\| descending |
| `saved_scenarios` | list\[`SavedScenario`\] \| null | Follow-up simulations saved during the interactive session (label, query, and LLM summary per scenario) |
| `final_summary` | `str` \| null | LLM-generated opening commentary |

### `TreatmentEffect`

| Field | Type | Description |
|---|---|---|
| `treatment` | `str` | Construct name of the treatment |
| `effect_size` | `float` \| null | Posterior mean treatment effect on the outcome under the default unit intervention; null if the model could not estimate this effect |
| `posterior_draws` | `list[float]` \| null | Full posterior distribution of the treatment effect (one draw per posterior sample) |
| `prob_positive` | `float` \| null | P(effect > 0)—posterior probability that the effect is positive |
| `identifiable` | `bool` | Whether this treatment-to-outcome effect is [causally identifiable](01b-measurement-identifiability.md#identifiabilitystatus) |
| `ppc_warnings` | list\[[`PPCWarning`](05b-inference-diagnostics.md#ppcresult)\] \| null | Posterior predictive check warnings relevant to this treatment |
| `prior_sensitivity_warning` | `str` \| null | Free-text warning when [power-scaling](05b-inference-diagnostics.md#powerscalingresult) classifies this treatment's drift parameters as `prior_dominated` |
| `temporal` | [`TemporalEffect`](#temporaleffect) \| null | Forward-simulation summary at 1-day, 7-day, and 30-day horizons |
| `manifest_effects` | `dict[str, float]` \| null | Per-manifest outcome decomposition via [loading-matrix](../reference/estimation.md#1-ct-sde-formulation) projection; keys are manifest names, values are `λ[manifest, outcome] × effect_mean` |

### `TemporalEffect`

| Field | Type | Description |
|---|---|---|
| `effect_1d` | `float` | Effect magnitude at 1 day post-intervention |
| `effect_7d` | `float` | Effect magnitude at 7 days |
| `effect_30d` | `float` | Effect magnitude at 30 days (or at the horizon boundary if shorter) |
| `peak_effect` | `float` | Maximum absolute effect reached during the trajectory |
| `time_to_peak_days` | `float` | Days from intervention onset to peak effect |
