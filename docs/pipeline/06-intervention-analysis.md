# Stage 6: Intervention Analysis

| Type | Interactive | Gate | Produces |
|---|---|---|---|
| llm+intervention | Yes | No | [`TreatmentEffect`](#treatmenteffect) list, interactive simulation tools |

Applies [do-operator](../reference/counterfactual-inference.md) interventions to the [Stage 5b fitted model](05b-inference-diagnostics.md#fittedartifact), ranks treatments by causal effect size, generates LLM commentary, and exposes three interactive tools for follow-up rung-2 and rung-3 simulations. This is the terminal stage—interactive edits persist in place with no downstream replay.

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage5b.result` | [Stage 5b](05b-inference-diagnostics.md) | Pickled [`FittedArtifact`](05b-inference-diagnostics.md#fittedartifact) containing posterior samples, runtime builder, observation times, PPC result, and power-scaling result |
| `stage1a.result` | [Stage 1a](01a-latent-model.md) | [`LatentModel`](01a-latent-model.md#latent-model) from which the outcome construct name is derived |
| `stage1b.result` | [Stage 1b](01b-measurement-identifiability.md) | [`CausalSpec`](01b-measurement-identifiability.md#causalspec) including identifiability status and measurement model |
| `stage1b_gate.result` | [Stage 1b](01b-measurement-identifiability.md) gate | Filtered treatment list with non-identifiable treatments removed |
| `question` | Pipeline request | Original research question for grounding the opening commentary |

Stage 5b provided the posterior and diagnostics; Stage 1b provided the identifiability verdicts. Stage 6 is the first point where posterior samples are translated into causal decision quantities.

## Process

Stage 6 runs in two phases: a deterministic intervention computation that produces the baseline ranking, followed by a single LLM generation that produces opening commentary. After completion the stage exposes three interactive tools for follow-up exploration.

**Baseline intervention ranking.** For each treatment that passed the [Stage 1b identifiability gate](01b-measurement-identifiability.md), the stage computes the steady-state causal effect of a unit intervention `do(treatment = baseline + 1)`. For each posterior draw of the drift matrix **A** and continuous intercept **c**, the baseline latent state is η\* = −**A**⁻¹**c**; the intervened state solves the modified linear system where the treatment row is clamped to η\*treatment + 1. The per-draw effect on the outcome is the difference between the intervened and baseline outcome values. The stage vmaps this computation over all posterior draws, producing a full posterior distribution of the treatment effect.

**Temporal forward simulation.** When temporal information is available (either from the [`model_clock`](01b-measurement-identifiability.md#measurement-model) or the median observed timestep), the stage also runs a 30-day forward simulation for each treatment. This discretizes the continuous-time system, starts from the baseline steady state, clamps the treatment at each step, and records the outcome trajectory. The mean trajectory across posterior draws is summarized into a [`TemporalEffect`](#temporaleffect) with 1-day, 7-day, and 30-day snapshots plus peak effect and time-to-peak.

**Manifest-level decomposition.** When posterior draws of the loading matrix λ are available, the stage projects each treatment's outcome-level effect through the loadings to produce per-manifest effects: `manifest_effect[i] = λ[i, outcome_idx] × effect_mean`. Only point-like manifests (those with `support_kind` of `null` or `"point"`) are included; interval-summary manifests are excluded from this simple projection.

**Diagnostic attachment.** Two upstream diagnostic signals are attached per treatment:

- *PPC warnings*: if [Stage 5b PPC](05b-inference-diagnostics.md#ppcresult) ran and produced warnings, the stage identifies which manifest variables are relevant to each treatment (via the loading matrix connecting treatment and outcome indices to manifests) and attaches only the relevant subset.
- *Prior-sensitivity warnings*: if [power-scaling analysis](05b-inference-diagnostics.md#powerscalingresult) classified any drift parameters as `prior_dominated`, the stage flags treatments whose drift parameters are among them.

**Ranking and artifact assembly.** Treatments are sorted by |`effect_size`| descending. The ranked list is persisted as a Prefect table artifact and as the `intervention_results` field of the stage payload.

**LLM commentary.** A single LLM generation receives the top-5 ranked effects, any diagnostic warnings, excluded non-identifiable treatments, and a summary of the follow-up capabilities. The LLM produces plain Markdown commentary for the user, persisted as `final_summary`.

**Outcome classification.** The stage sets `outcome` to `"warn"` if any treatment carries PPC warnings or a prior-sensitivity warning. Otherwise `outcome` is `"success"`.

## Interactive Tools

After the baseline ranking completes, Stage 6 exposes three read-only tools for follow-up exploration within the same conversation.

### `get_model_info`

Returns a structured read-only summary of the fitted model. The caller selects which sections to include:

| Section | Contents |
|---|---|
| `overview` | Outcome name, identifiable treatments, latent/manifest counts, inference method, observed time range |
| `variables` | Per-construct details (name, description, role, temporal status) and per-indicator details (measurement dtype, support kind, observation window) |
| `measurement` | Model clock and manifest variable names |
| `identifiability` | Identifiable treatments and non-identifiable treatments with blocking confounders |
| `diagnostics` | PPC warning count, power-scaling issues, and inference structure from [Stage 4b](04b-parametric-identifiability.md) |
| `baseline_effects` | Per-treatment effect size, `prob_positive`, and diagnostic warning counts from the baseline ranking |
| `capabilities` | Summary of available intervention and counterfactual modes |

An optional `names` filter restricts the response to specific constructs or indicators.

### `simulate_intervention`

Runs a [Pearl rung-2](../reference/counterfactual-inference.md) interventional simulation on the fitted generative model. The caller specifies:

- **Action**: an [`InterventionAction`](#interventionaction) naming the treatment, the mode (`"set"` to clamp to an absolute value, `"shift"` to add a delta to baseline), and the corresponding value or amount.
- **Query**: the estimand (`"steady_state"` for the long-run equilibrium effect, `"trajectory"` for a forward simulation over `horizon_days`), and the projection level (`"latent"`, `"manifest"`, or `"both"`).

For steady-state queries, the tool computes the intervened steady state per posterior draw and returns a posterior summary (mean, median, 95% CI, `prob_positive`). For trajectory queries, it additionally returns the day-by-day effect path and a [`TemporalEffect`](#temporaleffect) summary. Manifest projections decompose the latent outcome effect through the loading matrix. Any PPC or prior-sensitivity warnings from the baseline ranking are attached.

### `simulate_counterfactual`

Runs a [Pearl rung-3](../reference/counterfactual-inference.md) counterfactual forecast. This conditions on actually observed data before asking "what would have happened if we had intervened?" The caller specifies:

- **Evidence**: an observed history window defined by optional ISO-8601 `start_time` and `end_time` bounds over the fitted observation period. Defaults to the full observed range.
- **Action**: same [`InterventionAction`](#interventionaction) as rung-2 simulations.
- **Query**: the estimand (`"end_state"` or `"trajectory"`), `horizon_days`, and projection level.

The tool first performs abduction—recovering the latent state at the evidence boundary. It prefers a Kalman smoother on posterior-mean parameters applied to the evidence window; when the smoother is unavailable (non-Gaussian emissions), it falls back to a least-squares pseudoinverse of the observation model at the final evidence timepoint. From the abducted state, it forward-simulates both a baseline path (no intervention) and a counterfactual path (treatment clamped) over the horizon, and reports their difference as the causal effect. The return includes the evidence metadata, conditioning method, baseline and counterfactual forecast means, posterior summary, effect trajectory, and any warnings from the abduction step.

## Outputs

| Output | Type | Description |
|---|---|---|
| `intervention_results` | list\[[`TreatmentEffect`](#treatmenteffect)\] | Treatments ranked by \|`effect_size`\| descending |
| `saved_scenarios` | list\[[`SavedScenario`](#savedscenario)\] \| null | Follow-up simulations saved during the interactive session |
| `final_summary` | `str` \| null | LLM-generated opening commentary |

The contract also exposes `outcome` (`"success"` or `"warn"`) and `llm_trace` inherited from the base LLM stage contract.

## Definitions

### TreatmentEffect

`TreatmentEffect` is the final causal-decision object for one treatment—the authoritative output of the pipeline. It packages the posterior effect estimate, its uncertainty, the identifiability verdict, upstream diagnostic warnings, and optional temporal and manifest-level decompositions.

| Field | Type | Description |
|---|---|---|
| `treatment` | `str` | Construct name of the treatment |
| `effect_size` | `float` \| null | Posterior mean of do(treatment = baseline + 1) on the outcome; null if the model could not estimate this effect |
| `posterior_draws` | `list[float]` \| null | Full posterior distribution of the treatment effect (one draw per posterior sample) |
| `prob_positive` | `float` \| null | P(effect > 0)—posterior probability that the effect is positive |
| `identifiable` | `bool` | Whether this treatment-to-outcome effect is [causally identifiable](01b-measurement-identifiability.md#identifiabilitystatus) |
| `ppc_warnings` | `list[PPCWarning]` \| null | [Posterior predictive check](05b-inference-diagnostics.md#ppcresult) warnings for manifest variables relevant to this treatment's causal pathway |
| `prior_sensitivity_warning` | `str` \| null | Free-text warning when [power-scaling](05b-inference-diagnostics.md#powerscalingresult) classifies this treatment's drift parameters as `prior_dominated` |
| `temporal` | [`TemporalEffect`](#temporaleffect) \| null | Forward-simulation summary at 1-day, 7-day, and 30-day horizons |
| `manifest_effects` | `dict[str, float]` \| null | Per-manifest outcome decomposition via loading-matrix projection; keys are manifest names, values are `λ[manifest, outcome] × effect_mean` |

### TemporalEffect

Forward-simulation decomposition of a treatment effect over time.

| Field | Type | Description |
|---|---|---|
| `effect_1d` | `float` | Effect magnitude at 1 day post-intervention |
| `effect_7d` | `float` | Effect magnitude at 7 days |
| `effect_30d` | `float` | Effect magnitude at 30 days (or at the horizon boundary if shorter) |
| `peak_effect` | `float` | Maximum absolute effect reached during the trajectory |
| `time_to_peak_days` | `float` | Days from intervention onset to peak effect |

### SavedScenario

A follow-up simulation saved during the interactive session.

| Field | Type | Description |
|---|---|---|
| `label` | `str` | User-assigned or auto-generated label for the scenario |
| `query` | `str` | The tool call or natural-language query that produced this scenario |
| `summary` | `str` \| null | LLM-generated summary of the scenario result |

### InterventionAction

Specifies the do-operator action for both rung-2 and rung-3 simulations.

| Field | Type | Description |
|---|---|---|
| `variable` | `str` | Latent construct to intervene on |
| `mode` | `"set"` \| `"shift"` | `"set"` clamps the construct to an absolute latent-space value; `"shift"` adds a delta to the baseline |
| `value` | `float` \| null | Required when `mode="set"`: the absolute value to clamp to |
| `amount` | `float` \| null | Required when `mode="shift"`: the additive delta from baseline |

Example: for a study of agricultural practices and crop yield where Stage 1a posited constructs `Irrigation Frequency`, `Soil Nitrogen`, `Pest Pressure`, and `Crop Yield`, Stage 6 might rank `Soil Nitrogen` first with `effect_size=+0.38`, `prob_positive=0.96`, and a `TemporalEffect` showing the effect builds over 12 days (`time_to_peak_days=12.0`, `peak_effect=+0.41`) before settling to steady state. `Pest Pressure` might rank second with `effect_size=−0.22` and a prior-sensitivity warning because its drift cross-lag parameter was classified as `prior_dominated`. A follow-up `simulate_counterfactual` call conditioning on the last 60 days of observed data could then forecast what `Crop Yield` would have been had `Irrigation Frequency` been shifted by +0.5 units, returning both the latent effect trajectory and per-manifest decompositions showing differential impacts on "monthly harvest weight" versus "canopy greenness index."
