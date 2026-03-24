# Stage 6: Intervention Analysis

| Type | Interactive | Gate | Terminal behavior |
|---|---|---|---|
| llm | Yes | No | Interactive edits persist in place; no downstream replay |

Applies do-operator interventions to the fitted model, ranks treatments, and exposes a narrow terminal interactive surface. This page is the authoritative definition of `TreatmentEffect`. The counterfactual math lives in [../model-runtime/estimation.md](../model-runtime/estimation.md), the upstream artifact handoff is summarized in [../model-runtime/handoff-map.md](../model-runtime/handoff-map.md), and the terminal no-replay behavior is defined in [../runtime/execution-and-replay.md](../runtime/execution-and-replay.md).

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage5b.result` | Stage 5b | Fitted model artifact |
| `stage1a.result` | Stage 1a | Outcome name and treatment list |
| `stage1b.result` | Stage 1b | `CausalSpec`, including identifiability status |
| `stage1b_gate.result` | Stage 1b gate | Filtered treatment list with non-identifiable treatments removed |
| `question` | Pipeline request | Optional question for the opening commentary |

## Process

1. Compute the canonical baseline ranking with steady-state interventions `do(treatment = baseline + 1)`.
2. Repeat the intervention over posterior samples to compute posterior draws and `prob_positive`.
3. Attach PPC warnings, prior-sensitivity warnings, temporal effects, and manifest decompositions where available.
4. Rank treatments by absolute effect size.
5. Persist an initial interpretation as `final_summary` and `llm_trace`.
6. Expose read-only tools:
   - `get_model_info`
   - `simulate_intervention`
   - `simulate_counterfactual`

## Outputs

| Output | Type | Description |
|---|---|---|
| `intervention_results` | `list[TreatmentEffect]` | Ranked treatment effects |
| `saved_scenarios` | `list[SavedScenario]` | Optional saved follow-up simulations |
| `final_summary` | `str?` | Persisted Stage 6 interpretation |
| `llm_trace` | `LLMTrace?` | Opening commentary plus follow-up turns |

## Artifact Introduced

### TreatmentEffect

`TreatmentEffect` is the final intervention-analysis object for one treatment. It owns:

- the ranked effect-size summary
- posterior draws for that effect
- positivity probability
- any attached diagnostic warnings
- any temporal or manifest-level decomposition included in the result

This is the authoritative definition of the final causal-decision object emitted by the pipeline.

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `TreatmentEffect` | `{treatment, effect_size, posterior_draws, prob_positive, identifiable, ppc_warnings, prior_sensitivity_warning, temporal, manifest_effects}` | Final intervention-analysis payload |
| `prob_positive` | `P(effect > 0)` | Posterior probability that the effect is positive |
