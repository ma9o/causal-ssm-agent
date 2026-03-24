# MeasurementModel: Indicators

`MeasurementModel` is the domain primitive that explains how constructs are observed in data. The authoritative schema lives in [Stage 1b](../../pipeline/01b-measurement-identifiability.md).

## Constructs vs Indicators

A construct is the theoretical variable in the causal graph. An indicator is the observed manifestation used to measure that construct in data.

Examples:

- In healthcare, `Medication Adherence` may be measured by pharmacy refill gaps and pill-count compliance.
- In software engineering, `Incident Load` may be measured by alert count and pages acknowledged.
- In education, `Student Engagement` may be measured by attendance, assignment completion, and classroom participation.

## Indicator Semantics

The schema for `Indicator` is defined in [Stage 1b](../../pipeline/01b-measurement-identifiability.md). Semantically, each indicator answers four questions:

| Question | Field family | Meaning |
|---|---|---|
| What is being measured? | `name`, `construct_name`, `how_to_measure` | The indicator's substantive meaning |
| What kind of value is produced? | `measurement_dtype`, `ordinal_levels` | The support and category semantics of the measurement |
| Where does the value come from? | `source_columns`, `extraction_mode` | Whether extraction is computed directly or requires semantic interpretation |
| Over what support is it defined? | `aggregation`, `observation_window` | How raw observations become one indicator value |

## Extraction Modes

`MeasurementModel` owns whether an indicator is extracted by direct computation or by semantic interpretation:

- `computed`: the value can be obtained by deterministic transformation or aggregation over raw columns.
- `semantic`: the value requires LLM interpretation over a support window before it becomes a numeric indicator.

Stage 2 executes those modes, but the choice belongs here in the measurement definition.

## Measurement Dtype

`measurement_dtype` is a semantic commitment about the support of the observed variable. It is not yet a full likelihood choice.

| Dtype | Meaning | Typical downstream consequence |
|---|---|---|
| `continuous` | Real-valued measurement | Gaussian-family default in Stage 4 |
| `binary` | Two-state measurement | Bernoulli-family default |
| `count` | Non-negative event count | Poisson-family default |
| `ordinal` | Ordered categories | Ordered logistic default |
| `categorical` | Unordered categories | Categorical-family default |

Stage 4 consumes these dtype semantics when constructing the `ModelSpec`.
