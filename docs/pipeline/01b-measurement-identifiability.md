# Stage 1b: Measurement Model and Identifiability

| Type | Interactive | Gate | Produces |
|---|---|---|---|
| llm+grounding | Yes | Yes | [`CausalSpec`](#causalspec) |

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

**Validation loop.** The LLM submits its proposal via a `validate_measurement_model` tool call. The tool checks three things simultaneously:

- *Schema and compiler constraints*: every outcome construct has at least one indicator, no duplicate operationalizations, indicator references point to valid constructs, `measurement_dtype` and `aggregation` are compatible, and computed indicators have well-formed rules.
- *Causal identifiability*: the tool unrolls the latent graph to two timesteps (justified by [A3a](../reference/causal-spec/identifiability.md#a3a-latent-confounders-have-bounded-temporal-reach)), projects to an internal ADMG, and runs [y0's ID algorithm](../reference/causal-spec/identifiability.md#user-facing-dag-vs-internal-admg-projection) for each treatment-to-outcome pair. If some effects are blocked by an unobserved confounder, the tool reports which confounder is the problem and suggests adding proxy indicators to restore identifiability.
- *Marginalization analysis*: once identifiability passes, a deterministic post-processing step identifies which unobserved confounders can be safely ignored because they have no remaining confounding influence.

On failure the tool returns the specific errors; the LLM revises and resubmits within the same conversation until the tool returns VALID.

**Self-review.** A follow-up prompt asks the LLM to review its validated measurement model for coverage (every time-varying construct has at least one indicator), `how_to_measure` clarity, observation-window semantics, the [pure-indicators assumption](../reference/causal-spec/identifiability.md#a7-measurement-model-identification-enables-causal-identification) (no direct indicator-to-indicator edges), and absence of cumulative or running metrics. If the review surfaces issues, the LLM revises and re-validates before the conversation ends.

**Hard gate.** This stage gates the pipeline: if any treatment-to-outcome effect remains non-identifiable after the LLM's revision attempts, the pipeline stops. The gate can be overridden by the user, but the non-identifiability is recorded in the output.

## Outputs

| Output | Type | Description |
|---|---|---|
| `causal_spec` | [`CausalSpec`](#causalspec) | Combined latent model, measurement model, and identifiability status |

The public stage payload exposes that artifact directly. It may also include `gate_overridden` if the hard gate was overridden and `llm_trace` as runtime provenance for the UI.

## Definitions

### Measurement Model

The `MeasurementModel` defines how theoretical constructs are observed in data. It owns:

- the indicator list—each indicator carries `name`, `construct_name`, `how_to_measure`, `measurement_dtype`, `aggregation`, `observation_window`, `source_columns`, and `extraction_mode`
- the `model_clock`—the observation-window width used for extraction and discretization (see [Observation Windows and Model Clock](../reference/measurement-model/indicators.md#observation-windows-and-model-clock))

Indicators are reflective: the construct causes the indicator value, not the reverse. Each indicator's `extraction_mode` is either `"computed"` (a deterministic aggregation that [Stage 2](02-indicator-extraction.md) can evaluate mechanically) or `"semantic"` (requiring an LLM worker to interpret unstructured text).

### CausalSpec

`CausalSpec` is the combined causal-and-measurement handoff object. It packages:

- the [Stage 1a `LatentModel`](01a-latent-model.md#latent-model)
- the `MeasurementModel` introduced here
- the treatment-level [`IdentifiabilityStatus`](#identifiabilitystatus)

Later stages should treat `CausalSpec` as the authoritative answer to "what causal question are we fitting?" and "how is each construct measured?"

### IdentifiabilityStatus

`IdentifiabilityStatus` records which treatment-to-outcome effects are identifiable under the latent and measurement assumptions and which are blocked. For identifiable treatments it includes the identification method (backdoor, frontdoor, or instrumental variable under linearity) and any confounders that were safely marginalized. For non-identifiable treatments it reports the blocking confounders.

Example: for a study of developer workload and code quality where Stage 1a posited an unobserved confounder `Organizational Pressure`, Stage 1b might map `Developer Workload` to indicators like "number of open PRs assigned" (computed, count) and "sprint velocity" (computed, mean), map `Review Thoroughness` to "average review comment count per PR" (computed, mean), and add a proxy indicator "manager-reported deadline pressure" (semantic, ordinal) to restore identifiability of the `Organizational Pressure` confounder path.
