# Stage 3: Extraction Validation

| Modality | Interactive | Produces |
|---|---|---|
| Computed | No | [`IndicatorAudit`](#indicatoraudit) per indicator, dataset-level issues |

Audits the observations extracted by [Stage 2](02-indicator-extraction.md) against the indicator metadata declared in the [Stage 1b `CausalSpec`](01b-measurement-identifiability.md#causalspec), then computes an [empirical profile](#empiricalprofile) for each indicator. The audit result is the primary data-quality surface consumed by [Stage 4](04-model-specification-priors.md) when building decision cards for prior elicitation.

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage1b.result` | [Stage 1b](01b-measurement-identifiability.md) | [`CausalSpec`](01b-measurement-identifiability.md#causalspec)—[indicator](01b-measurement-identifiability.md#measurementmodel) and construct metadata, `model_clock` |
| `stage2.result` | [Stage 2](02-indicator-extraction.md) | Encoded long-format `ObservationRecord` table persisted from Stage 2 |

Stage 2 executed the extraction instructions; Stage 3 asks whether the resulting data are internally consistent, statistically usable, and plausible. No LLM is involved—every check is deterministic.

## Process

Stage 3 runs a fixed set of composable [validation rules](#validation-rules) over the persisted Stage 2 `ObservationRecord` table, reduces the findings into per-indicator statuses, computes empirical profiles from that same table, and packages everything into an [`IndicatorAudit`](#indicatoraudit) per indicator.

**Context assembly.** The stage parses the [`model_clock`](01b-measurement-identifiability.md#observation-windows-and-model-clock) from the [`CausalSpec`](01b-measurement-identifiability.md#causalspec) into hours, builds lookup tables for indicator metadata and construct metadata, and validates the single long-format table loaded from Stage 2. For each indicator, it pre-computes an `IndicatorContext`: the numeric `Float64` series after coercion and null removal, observation count, variance, declared `measurement_dtype`, whether the parent construct is time-invariant, and a parsed timestamp series attempted against nine format patterns with optional timezone stripping.

**Per-indicator rules.** Nine indicator-level rules run in sequence for each indicator. Each rule receives the indicator's data and context and returns zero or more [`ValidationIssue`](#validationissue)s with an attached `cell_key` linking each issue to the metric it concerns:

| Rule | Checks | Severity | Threshold |
|---|---|---|---|
| `missing` | Indicator declared in `CausalSpec` but absent from extracted data | warning | any absence |
| `no_numeric` | Rows exist but no values survived `Float64` coercion | error | zero numeric values |
| `timestamps` | Observation-time parseability | error if 100% unparseable; warning if >50% | fraction of `anchor_time` values that fail all nine timestamp formats |
| `sample_size` | Minimum observation count | warning | < 10 observations |
| `variance` | Zero-variance detection | error | variance = 0 (constant series) |
| `dtype_range` | Values conform to declared [`measurement_dtype`](01b-measurement-identifiability.md#measurement-dtype) | error for binary (values outside {0, 1}) and count (negative or fractional values); warning for continuous (outliers beyond 3× IQR) | see per-dtype logic below |
| `time_coverage` | Data span relative to model clock | warning | time span < 10 × `model_clock` hours; skipped for time-invariant constructs |
| `timestamp_gaps` | Largest consecutive gap | warning | max gap > 5 × `model_clock` hours; skipped for time-invariant constructs |
| `hallucination_signals` | Patterns suspicious of LLM fabrication: dominant duplicate values (non-binary, non-count) and perfect arithmetic sequences | warning | >50% duplicate concentration, or all sorted diffs identical with non-zero step (≥5 observations) |

**Dtype-range details.** The `dtype_range` rule applies different logic per declared type:

- *Binary*: any value not in {0, 1} is an error.
- *Count*: negative values or values more than 1e-6 from their nearest integer are errors (matching the tolerance used at SSM build time).
- *Continuous*: with ≥10 observations, values beyond Q1 − 3·IQR or Q3 + 3·IQR are reported as outlier warnings.

**Dataset-level rules.** One dataset-level rule runs after all indicators:

| Rule | Checks | Severity |
|---|---|---|
| `construct_correlations` | For constructs with ≥2 indicators, daily-aggregated Pearson correlation between every indicator pair; negative correlation violates the [reflective measurement assumption](../reference/measurement-model/assumptions.md#a1-reflective-measurement-model) | warning (when r < 0, with ≥10 aligned days) |

**Reduction.** A central reducer aggregates per-indicator findings into two structures: a flat issue list and a health-metrics map keyed by indicator name. For each metric key (`n_obs`, `variance`, `n_unparseable_timestamps`, `time_coverage_ratio`, `max_gap_ratio`, `dtype_violations`, `duplicate_pct`, `arithmetic_sequence_detected`), the worst severity among matching issues determines the cell status (`ok`, `warning`, or `error`). Rules own threshold logic; the reducer only aggregates.

**Empirical profiles.** After validation, the stage computes an [`EmpiricalProfile`](#empiricalprofile) for each indicator from the same encoded `ObservationRecord` table it validated. The profile captures central tendency, spread, quantiles, distributional shape indicators (zero fraction, non-negativity, unit-interval membership, integer-valuedness, variance-to-mean ratio), and the health metrics computed during validation (coverage ratio, gap ratio, dtype violations, duplicate percentage, arithmetic-sequence flag, unparseable timestamps).

**Audit assembly.** Each indicator's profile and validation findings are packaged into an [`IndicatorAudit`](#indicatoraudit). The audit map is keyed by indicator name.

**Outcome derivation.** The stage determines an outcome for the overall payload:

- `"fail"` — at least one issue has `severity: "error"` (the `is_valid` flag is `false`), with `fail_reason = "data_validation_failed"`.
- `"warn"` — no errors, but at least one issue has `severity: "warning"`.
- `"success"` — no issues at warning level or above.

When Stage 3 emits `"fail"`, the pipeline stops before Stage 4 because there is no validated Stage 2 `ObservationRecord` table on which to base quantitative fitting. `"warn"` and `"success"` continue downstream, and [Stage 4](04-model-specification-priors.md) still uses the audit to inform the LLM about data-quality constraints during prior elicitation.

## Outputs

| Output | Type | Description |
|---|---|---|
| `is_valid` | `bool` | `true` if no error-severity issues exist across all indicators and dataset checks |
| `indicators` | `dict[str, IndicatorAudit]` | Keyed by indicator name; each entry bundles the [empirical profile](#empiricalprofile) and [validation findings](#indicatorvalidation) |
| `dataset_issues` | `list[ValidationIssue]` | Issues not attributable to a single indicator (e.g., negative cross-indicator correlations) |

The payload also includes an `outcome` field (`"success"`, `"warn"`, or `"fail"`) used by the pipeline orchestrator for logging and the web UI for status display.

## Definitions

### IndicatorAudit

The per-indicator validation object emitted by Stage 3. It bundles two things: the data-facing [`EmpiricalProfile`](#empiricalprofile) and the rule-facing [`IndicatorValidation`](#indicatorvalidation). This is the authoritative definition of the data-quality surface for one indicator.

| Field | Type | Description |
|---|---|---|
| `profile` | [`EmpiricalProfile`](#empiricalprofile) ∣ `null` | Descriptive statistics from the encoded Stage 2 `ObservationRecord` collection; `null` if no numeric values survived coercion |
| `validation` | [`IndicatorValidation`](#indicatorvalidation) | Issues and per-check statuses |

### EmpiricalProfile

Descriptive statistics computed from one indicator's numeric `ObservationRecord` series. [Stage 4](04-model-specification-priors.md) reads these profiles to build distribution cards and construct-scale cards that ground the LLM's prior proposals.

| Field | Type | Description |
|---|---|---|
| `measurement_dtype` | `str` ∣ `null` | Declared dtype from the [`CausalSpec`](01b-measurement-identifiability.md#causalspec) (`"binary"`, `"count"`, `"continuous"`, or `"ordinal"`) |
| `n_obs` | `int` | Number of non-null numeric observations |
| `mean` | `float` ∣ `null` | Arithmetic mean |
| `std` | `float` ∣ `null` | Standard deviation |
| `min` | `float` ∣ `null` | Minimum value |
| `max` | `float` ∣ `null` | Maximum value |
| `q25` | `float` ∣ `null` | 25th percentile |
| `q50` | `float` ∣ `null` | Median |
| `q75` | `float` ∣ `null` | 75th percentile |
| `variance` | `float` ∣ `null` | Sample variance |
| `time_coverage_ratio` | `float` ∣ `null` | Observed time span divided by the required minimum (10 × `model_clock`); capped at 1.0. `null` for fewer than 2 timestamps or time-invariant constructs |
| `max_gap_ratio` | `float` ∣ `null` | Largest consecutive gap divided by the warning threshold (5 × `model_clock`). `null` for fewer than 3 timestamps or time-invariant constructs |
| `dtype_violations` | `int` ∣ `null` | Count of values that violate the declared `measurement_dtype` constraints |
| `duplicate_pct` | `float` ∣ `null` | Fraction of observations equal to the single most common value |
| `arithmetic_sequence_detected` | `bool` | `true` if all sorted consecutive differences are identical with non-zero step (≥5 observations) |
| `n_unparseable_timestamps` | `int` ∣ `null` | Count of `anchor_time` values that failed all timestamp format patterns |
| `zero_fraction` | `float` ∣ `null` | Fraction of observations that are exactly zero |
| `is_nonnegative` | `bool` ∣ `null` | `true` if minimum ≥ 0 |
| `is_unit_interval` | `bool` ∣ `null` | `true` if all values fall in [0, 1] |
| `looks_integer_valued` | `bool` ∣ `null` | `true` if every value is within 1e-8 of its nearest integer |
| `variance_to_mean_ratio` | `float` ∣ `null` | Variance / mean (index of dispersion); `null` when mean ≤ 0 |

### IndicatorValidation

Groups the validation findings for one indicator.

| Field | Type | Description |
|---|---|---|
| `issues` | `list[ValidationIssue]` | All issues attributed to this indicator across all rules |
| `checks` | `dict[str, status]` | Map from metric key to worst-case status (`"ok"`, `"warning"`, or `"error"`). Keys are drawn from: `n_obs`, `variance`, `n_unparseable_timestamps`, `time_coverage_ratio`, `max_gap_ratio`, `dtype_violations`, `duplicate_pct`, `arithmetic_sequence_detected` |

### ValidationIssue

A single finding emitted by a validation rule.

| Field | Type | Description |
|---|---|---|
| `indicator` | `str` ∣ `null` | Indicator name, or `null` for dataset-level issues |
| `issue_type` | `str` | Machine-readable category: `missing`, `no_numeric`, `unparseable_timestamps`, `low_n`, `no_variance`, `dtype_violation`, `insufficient_coverage`, `large_timestamp_gap`, `suspicious_pattern`, `low_construct_correlation`, or `no_data` |
| `severity` | `"error"` ∣ `"warning"` ∣ `"info"` | Determines cell status and overall `is_valid` flag |
| `message` | `str` | Human-readable description with concrete values |

### Validation Rules

The rule registry is the extension point for adding new checks. Each rule is a named function with a declared scope (`"indicator"` or `"dataset"`) that returns zero or more [`ValidationIssue`](#validationissue)s. Adding a new check requires appending one entry to `RULES`—no other code changes.

Example: for a study tracking developer productivity where Stage 2 extracted indicators "lines of code per day" (continuous), "number of PR reviews" (count), and "burnout self-report" (ordinal), Stage 3 might flag "lines of code per day" with a `suspicious_pattern` warning if >50% of extracted values are identical (suggesting the LLM hallucinated a constant), report `insufficient_coverage` on "burnout self-report" if the self-report survey data spans only 3 weeks against a `model_clock` of `"1d"`, and surface a `low_construct_correlation` warning if daily-aggregated "lines of code per day" and "number of PR reviews" correlate negatively despite both measuring the same construct.
