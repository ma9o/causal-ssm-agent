# Stage 1b: Measurement Model and Identifiability

Maps constructs to observable indicators and checks whether each treatment-outcome effect is causally identifiable. This page is the authoritative definition of the measurement model, `CausalSpec`, and `IdentifiabilityStatus`. For the construct-versus-indicator split and timescale rules that govern this stage, see [../concepts/scope-and-timescales.md](../concepts/scope-and-timescales.md). For the identification assumptions behind the y0 check, see [../concepts/assumptions.md](../concepts/assumptions.md).

## At a Glance

| Property | Value |
|---|---|
| Type | Semantic |
| Interactive | Yes |
| Gate | Hard gate |
| Produces | [`CausalSpec`](#causalspec) plus identifiability status |

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | Pipeline request | Grounds measurement choices |
| `stage0.result` | Stage 0 | Ingested dataframe plus column descriptions |
| `stage1a.result` | Stage 1a | Latent model with constructs and edges |

## Process

1. Format the dataset schema as LLM context.
2. Run a single LLM conversation with `validate_measurement_model(measurement_json)`.
3. Validate three things together:
   - schema consistency between indicators and the latent model
   - compiler constraints such as dtype, support, and aggregation validity
   - causal identifiability via y0's ID algorithm under the assumptions summarized in [../concepts/assumptions.md](../concepts/assumptions.md)
4. If identifiability fails, let the model revise indicators or proxies and try again.
5. Assemble the final `CausalSpec`.

## Outputs

| Output | Type | Description |
|---|---|---|
| `causal_spec` | `CausalSpec` | Combined latent, measurement, and identifiability payload |
| `gate_overridden` | `GateOverrideContract?` | Present if the hard gate was overridden |
| `llm_trace` | `LLMTrace?` | Conversation trace |

## Artifacts Introduced

### Measurement Model

The measurement model defines how theoretical constructs are observed in data. It owns:

- the indicator list
- the construct-to-indicator mapping
- extraction mode
- aggregation semantics
- observation-window semantics
- the `model_clock` used by later extraction and discretization steps

### CausalSpec

`CausalSpec` is the combined causal-and-measurement payload. It packages:

- the Stage 1a `LatentModel`
- the measurement model introduced here
- the treatment-level identifiability result

Later stages should treat `CausalSpec` as the authoritative handoff object for "what causal question are we fitting?" and "how is it measured?"

### IdentifiabilityStatus

`IdentifiabilityStatus` is the causal-identification result attached to the `CausalSpec`. It records which treatment-to-outcome effects are identifiable under the latent and measurement assumptions and which are blocked.

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `Indicator` | `{name, construct_name, how_to_measure, measurement_dtype, aggregation, observation_window, ordinal_levels, source_columns, extraction_mode}` | `extraction_mode` is `"computed"` or `"semantic"` |
| `MeasurementModel` | `{indicators, model_clock}` | `model_clock` is the observation-window width used for extraction and discretization; see [../concepts/scope-and-timescales.md](../concepts/scope-and-timescales.md) |
| `IdentifiabilityStatus` | `{status, non_identifiable_treatments, marginalization_analysis}` | Per-treatment identifiability summary |
