# Stage 1b: Measurement Model and Identifiability

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | Yes | [`CausalSpec`](#causalspec) |

Grounds the [Stage 1a latent model](01a-latent-model.md#latent-model) in observed data by proposing [indicators](../reference/measurement-model/indicators.md) for each construct, then checks whether each treatment-to-outcome effect is [causally identifiable](../reference/causal-spec/identifiability.md).

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | User | Original research question—grounds measurement choices |
| `stage0.result` | [Stage 0](00-ingestion.md) | Ingested dataframe plus column descriptions |
| `stage1a.result` | [Stage 1a](01a-latent-model.md) | Latent model with constructs and edges |

Stage 1a provided theoretical structure without seeing any data. Stage 1b is the first point where the model meets the dataset.

## Process

Stage 1b runs a single LLM conversation that bridges theory and data. The LLM sees the latent model, the research question, and a schema summary of the ingested dataset. The conversation has three phases: indicator proposal grounded by a combined validation tool, identifiability repair if needed, and a self-review pass.

**Forward reasoning from constructs to columns.** For each construct in the latent model, the LLM proposes one or more [indicators](../reference/measurement-model/indicators.md)—observable proxies that operationalize it. Each indicator specifies which source columns it draws from, how to extract it (`extraction_mode`: `"computed"` for deterministic aggregation, `"semantic"` for LLM-based extraction), its [measurement dtype](../reference/measurement-model/indicators.md#measurement-dtype), its [aggregation and support window](../reference/measurement-model/indicators.md#observation-windows-and-model-clock), and the shared [`model_clock`](../reference/measurement-model/indicators.md#observation-windows-and-model-clock) that governs extraction and downstream fitting.

**Validation loop.** The LLM submits its proposal via a `validate_measurement_model` tool call. The tool checks two things directly, then the orchestration layer may run additional deterministic analysis:

- *Schema and compiler constraints*: every outcome construct has at least one indicator, no duplicate operationalizations, indicator references point to valid constructs, `measurement_dtype` and `aggregation` are compatible, and computed indicators have well-formed rules.
- *Causal identifiability*: the tool unrolls the latent graph to two timesteps (justified by [A3a](../reference/causal-spec/identifiability.md#a3a-latent-confounders-have-bounded-temporal-reach)), projects to an internal ADMG, and runs [y0's ID algorithm](../reference/causal-spec/identifiability.md#user-facing-dag-vs-internal-admg-projection) for each treatment-to-outcome pair. If some effects are blocked by an unobserved confounder, the tool reports which confounder is the problem and suggests adding proxy indicators to restore identifiability.
- *Deterministic follow-up analysis*: after a valid `CausalSpec` is captured, the orchestration layer may run additional analysis over unobserved constructs. That analysis is internal bookkeeping; the public stage contract remains the `CausalSpec` plus optional runtime provenance.

On failure the tool returns the specific errors; the LLM revises and resubmits within the same conversation until the tool returns VALID.

**Self-review.** A follow-up prompt asks the LLM to review its validated measurement model for coverage (every time-varying construct has at least one indicator), `how_to_measure` clarity, observation-window semantics, the [pure-indicators assumption](../reference/causal-spec/identifiability.md#a7-measurement-model-identification-enables-causal-identification) (no direct indicator-to-indicator edges), and absence of cumulative or running metrics. If the review surfaces issues, the LLM revises and re-validates before the conversation ends.

**Outcome semantics.** This stage first filters treatment-to-outcome effects. Effects that remain non-identifiable are recorded in `causal_spec.identifiability` and excluded from downstream intervention analysis. The stage then classifies its public outcome:

- `"success"`: every treatment effect is identifiable
- `"warn"`: some treatment effects were filtered out, but at least one identifiable treatment remains
- `"fail"` with `fail_reason = "no_identifiable_treatments"`: no identifiable treatments remain, so the pipeline stops after Stage 1b

## Outputs

| Output | Type | Description |
|---|---|---|
| `causal_spec` | [`CausalSpec`](#causalspec) | Combined latent model, measurement model, and identifiability status |

The public stage payload exposes that artifact directly. It may also include `fail_reason` when the stage stops and `llm_trace` as runtime provenance for the UI. Internal runtime-only fields, such as the filtered identifiable-treatment list used by Stage 6, are not part of the public contract.

## Definitions

### Measurement Model

The `MeasurementModel` nested inside `CausalSpec` has two top-level fields:

| Field | Type | Description |
|---|---|---|
| `indicators` | `list[Indicator]` | Observed indicators attached to constructs. Each `Indicator` carries `name`, `construct_name`, `how_to_measure`, `measurement_dtype`, `aggregation`, `observation_window`, `ordinal_levels`, `source_columns`, and `extraction_mode`. |
| `model_clock` | `str` | Shared observation-window width used for extraction and discretization. |

Indicators are reflective: the construct causes the indicator value, not the reverse. Each indicator's `extraction_mode` is either `"computed"` (a deterministic aggregation that [Stage 2](02-indicator-extraction.md) can evaluate mechanically) or `"semantic"` (requiring an LLM worker to interpret unstructured text).

For indicator semantics, support-window behavior, aggregation rules, and `model_clock`, see [reference/measurement-model/indicators.md](../reference/measurement-model/indicators.md).

### CausalSpec

`CausalSpec` is the Stage 1b handoff object:

| Field | Type | Description |
|---|---|---|
| `latent` | [`LatentModel`](01a-latent-model.md#latent-model) | The validated Stage 1a construct-level graph. |
| `measurement` | [`MeasurementModel`](#measurement-model) | The indicator mapping and model clock introduced here. |
| `identifiability` | [`IdentifiabilityStatus`](#identifiabilitystatus) \| `null` | Treatment-level identifiability results from the Stage 1b checker. |

Downstream stages use `CausalSpec` as the combined causal-and-measurement input to extraction, model specification, and intervention analysis.

### IdentifiabilityStatus

`IdentifiabilityStatus` records which treatment-to-outcome effects are identifiable under the latent and measurement assumptions:

| Field | Type | Description |
|---|---|---|
| `identifiable_treatments` | `dict[str, IdentifiedTreatmentStatus]` | Treatment names mapped to the identification method, estimand, marginalized confounders, and any instruments used. |
| `non_identifiable_treatments` | `dict[str, NonIdentifiableTreatmentStatus]` | Treatment names mapped to the blocking confounders and optional notes. |

For the assumptions behind these results, including A3a and the internal DAG-to-ADMG projection, see [reference/causal-spec/identifiability.md](../reference/causal-spec/identifiability.md).

Example: for a study of developer workload and code quality where Stage 1a posited an unobserved confounder `Organizational Pressure`, Stage 1b might map `Developer Workload` to indicators like "number of open PRs assigned" (computed, count) and "sprint velocity" (computed, mean), map `Review Thoroughness` to "average review comment count per PR" (computed, mean), and add a proxy indicator "manager-reported deadline pressure" (semantic, ordinal) to restore identifiability of the `Organizational Pressure` confounder path.
