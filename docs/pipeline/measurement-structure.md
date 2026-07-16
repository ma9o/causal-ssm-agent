# Measurement Structure and Identifiability

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | Yes | [`MeasurementStructure`](#measurementstructure), authored [`KnownInput`](#knowninput) declarations; derives [`CausalDesign`](#causaldesign) and [`IdentificationReport`](#identificationreport) when positive |

Operationalizes the [`LatentStructure`](latent-structure.md#latent-structure) against observed data by specifying indicators for each construct, then checks whether each treatment-to-outcome effect is causally identifiable[^pearl2009].

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | User | Original research question, used to justify measurement choices |
| `latent_structure` | [`latent_structure` transition](latent-structure.md) | `LatentStructure` with constructs and edges |
| `raw_dataframe` | [`raw_data` transition](ingestion.md) | Raw dataframe with column descriptions |

`latent_structure` transition provided theoretical structure without seeing any data. `measurement_structure` transition is the first point where the model meets the dataset.

## Process

`measurement_structure` transition runs one LLM conversation that bridges theory and data. The LLM sees the latent structure, the research question, and a schema summary of the ingested dataset. The conversation has two phases: an initial measurement-structure and known-input proposal checked by a validation tool, followed by a self-review pass using the same validator.

```mermaid
flowchart LR
    P[Propose] --> V1{Validator} -- errors --> P
    V1 -- VALID --> R[Review] --> V2{Validator} -- errors --> R
    V2 -- VALID --> M([MeasurementStructure + KnownInputs])
    M -->|derive| C([CausalDesign])
    C -->|derive when positive| I([IdentificationReport])
```

**Propose:** For each construct in the latent structure, the LLM proposes one or more indicators: observed variables that operationalize the construct in this dataset. Each indicator names the source columns it uses, how extraction will work, what kind of value it produces, and over what support window that value is defined. When a directly observed construct trajectory should condition the dynamics rather than remain a latent state, the proposal also identifies its source indicator as a known input.

**Validator:** The LLM submits its proposal via a `validate_measurement_structure` tool call. The tool checks schema and compiler constraints:

- *Outcome coverage:* every outcome construct has at least one indicator
- *No duplicate indicator definitions* across indicators
- *Valid construct references:* indicator references point to constructs in the latent structure
- *Dtype–aggregation compatibility:* `measurement_dtype` and `aggregation` are compatible
- *Computed-rule validity:* computed indicators have valid rule expressions
- *Known-input integrity:* declarations reference an existing construct and an indicator that measures that same construct
- *Estimation projection:* removing known inputs from the state vector still satisfies compiler-owned state coverage and loading-rank constraints

After the authored artifact validates, the machine derives the complete causal design and checks [causal identifiability](../reference/causal-design/identifiability.md) for each treatment-to-outcome pair.

**Review:** A follow-up prompt asks the LLM to review its validated measurement structure for coverage, operationalization clarity in `how_to_measure`, observation-window semantics, the [reflective measurement assumption](../reference/measurement-structure/assumptions.md#a1-reflective-measurement-structure), absence of cumulative or running metrics, and whether every known-input declaration is justified by direct observation and explicit missing-value semantics. If the review surfaces issues, the LLM revises and re-validates before the conversation ends.

### Example

For a study of developer workload and code quality, `measurement_structure` transition might map `Developer Workload` to indicators like "number of open PRs assigned" (computed, count) and "sprint velocity" (computed, mean), and map `Review Thoroughness` to "average review comment count per PR" (computed, mean). If an assigned on-call shift is represented as a construct with a directly recorded schedule indicator, it can be declared as a known input so its realized trajectory drives the retained latent states without becoming one itself.

## Outputs

| Output | Type | Description |
|---|---|---|
| `measurement_structure` | [`MeasurementStructure`](#measurementstructure) | Authored indicator mapping and model clock |
| `known_inputs` | `list[KnownInput]` | Authored declarations of observed trajectories compiled as transition inputs |
| `causal_design` | [`CausalDesign`](#causaldesign) | Machine-derived composition, identification status, and estimation projection |
| `identification_report` | [`IdentificationReport`](#identificationreport) | Machine-derived positive identification gate, present only when at least one treatment effect is explicitly identifiable |

### `MeasurementStructure`

| Field | Type | Description |
|---|---|---|
| `indicators` | `list[Indicator]` | Observed indicators attached to constructs |
| `model_clock` | `str` | Shared observation-window width used for extraction, discretization, and the default lag unit for construct-level temporal semantics |

Indicators are reflective[^bollen1989]: the construct causes the indicator value, not the reverse. The assumptions behind that commitment live in [measurement-structure/assumptions.md](../reference/measurement-structure/assumptions.md).

### `Indicator`

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Indicator name used everywhere downstream |
| `construct_name` | `str` | Name of the parent construct in the latent structure |
| `how_to_measure` | `str` | Human-readable measurement instructions grounded in the dataset |
| `measurement_dtype` | `str` | Semantic value type: `continuous`, `binary`, `count`, `ordinal`, or `categorical` |
| `aggregation` | `str` | Summary operator applied within each realized support window |
| `observation_window` | `str` | Window width such as `"1d"` or `"1w"` over which one indicator value is defined |
| `ordinal_levels` | `list[str]` \| `null` | Ordered labels when `measurement_dtype="ordinal"` |
| `source_columns` | `list[str]` | Raw columns needed to compute or interpret the indicator |
| `extraction_mode` | `str` | Whether extraction is deterministic (`computed`) or LLM-mediated (`semantic`) |

### `KnownInput`

| Field | Type | Description |
|---|---|---|
| `construct` | `str` | Construct removed from the latent state vector and treated as an observed transition driver |
| `source_indicator` | `str` | Indicator for the same construct that supplies the input trajectory |
| `scale` | `float` | Positive divisor applied to the source values before inference |
| `missing_policy` | `str` | Whether missing grid values become zero or carry the last observed value forward |

### `observation_window` and `model_clock`

Examples of indicator-level observation windows:

- "Average heart rate over the previous day"
- "Number of production incidents during the previous week"
- "Teacher feedback sentiment in the current grading period"

Different indicators may use different `observation_window` values as long as they are aligned back onto the shared `model_clock`.

### Indicator Level `aggregation`

| Operator | Support meaning | Typical anchor |
|---|---|---|
| `mean` | Average level over the window matters | `support_end` |
| `sum` | Cumulative amount over the window matters | `support_end` |
| `count` | Event frequency over the window matters | `support_end` |
| `last` | The most recent observed state in the window matters | `support_end` |
| `first` | The earliest observed state in the window matters | `support_start` |
| `std` | Within-window instability matters | `support_end` |

These are substantive commitments, not mere implementation details. A daily mean mood score and an end-of-day mood score encode different theories of what matters.

### Derived Observation Semantics

The `MeasurementStructure` does not store row timestamps itself, but it fully determines the row-level support semantics that [`measurements` transition](extraction.md) materializes into [`ObservationRecord`](extraction.md#observationrecord) fields. The derivation is deterministic: the indicator's `aggregation` operator selects the `support_kind` (point vs. interval) and the `anchor_policy` per the "Typical anchor" column above.

### `CausalDesign`

`CausalDesign` is the machine-derived combined input to downstream transitions:

| Field | Type | Description |
|---|---|---|
| `latent` | [`LatentStructure`](latent-structure.md#latent-structure) | The validated `latent_structure` transition construct-level graph |
| `measurement` | [`MeasurementStructure`](#measurementstructure) | The indicator mapping and model clock introduced here |
| `identifiability` | [`IdentifiabilityStatus`](#identifiabilitystatus) \| `null` | Treatment-level identifiability results from the `measurement_structure` transition checker |
| `estimation` | [`EstimationSpec`](#estimationspec) \| `null` | Executable state-space projection derived from the authored structures and known inputs |

### `EstimationSpec`

| Field | Type | Description |
|---|---|---|
| `state_order` | `list[str]` | Retained latent states in canonical compiler order |
| `edges` | `list[CausalEdge]` | Directed transition edges into retained states, including edges from known inputs |
| `induced_dependencies` | `list[InducedDependency]` | Covariance dependencies induced by marginalized latent root confounders |
| `known_inputs` | `list[KnownInput]` | Validated authored input declarations copied into the executable projection |

### `IdentifiabilityStatus`

| Field | Type | Description |
|---|---|---|
| `identifiable_treatments` | `dict[str, IdentifiedTreatmentStatus]` | Treatment names mapped to the identification method, estimand, marginalized confounders, and any instruments used |
| `non_identifiable_treatments` | `dict[str, NonIdentifiableTreatmentStatus]` | Treatment names mapped to the blocking confounders and optional notes |

### `IdentificationReport`

| Field | Type | Description |
|---|---|---|
| `outcome_name` | `str` | Outcome construct name |
| `estimable_treatments` | `list[str]` | Nonempty list of treatments with explicitly identified effects |
| `non_identifiable_treatments` | `dict[str, NonIdentifiableTreatmentStatus]` | Treatment names mapped to the blocking confounders and optional notes |

The identifiability assumptions, including temporal unrolling and the internal DAG-to-ADMG projection, live in [causal-design/identifiability.md](../reference/causal-design/identifiability.md).

[^pearl2009]: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Bibliography entry](../reference/bibliography.md)
[^bollen1989]: Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley. [Bibliography entry](../reference/bibliography.md)
