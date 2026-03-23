# Stage 6: Intervention Analysis

Applies do-operator interventions to the fitted model, ranks treatments, and exposes a narrow terminal interactive surface.

## At a Glance

| Property | Value |
|---|---|
| Type | Hybrid |
| Interactive | Yes |
| Gate | No |
| Terminal behavior | Interactive edits persist in place; no downstream replay |

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

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `TreatmentEffect` | `{treatment, effect_size, posterior_draws, prob_positive, identifiable, ppc_warnings, prior_sensitivity_warning, temporal, manifest_effects}` | Final intervention-analysis payload |
| `prob_positive` | `P(effect > 0)` | Posterior probability that the effect is positive |

## Related Docs

- [../model-runtime/handoff-map.md](../model-runtime/handoff-map.md)
- [../model-runtime/estimation.md](../model-runtime/estimation.md)
- [../runtime/execution-and-replay.md](../runtime/execution-and-replay.md)
- [../concepts/artifact-glossary.md](../concepts/artifact-glossary.md)
