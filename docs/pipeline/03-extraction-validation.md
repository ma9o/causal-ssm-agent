# Stage 3: Extraction Validation

Audits extracted observations and computes empirical profiles per indicator. This page is the authoritative definition of `IndicatorAudit`. It is the extraction-quality assurance surface described in [../concepts/pipeline-dimensions.md](../concepts/pipeline-dimensions.md), applied to the Stage 2 observation artifacts.

## At a Glance

| Property | Value |
|---|---|
| Type | Computed |
| Interactive | No |
| Gate | No |
| Produces | Indicator audits and dataset-level issues |

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage1b.result` | Stage 1b | `CausalSpec` with indicator metadata and causal structure |
| `stage2.result` | Stage 2 | Raw and model-ready dataframes |

## Process

1. Run structural, data-quality, temporal, distributional, and alignment checks.
2. Compute empirical profiles per indicator: central tendency, spread, quantiles, coverage, gaps, duplicates, and related metrics.
3. Derive statuses at cell, indicator, and dataset level.

## Outputs

| Output | Type | Description |
|---|---|---|
| `is_valid` | `bool` | `True` if no errors are present |
| `indicators` | `dict[str, IndicatorAudit]` | Per-indicator profile plus findings |
| `dataset_issues` | `list[ValidationIssue]` | Dataset-level issues |

## Artifact Introduced

### IndicatorAudit

`IndicatorAudit` is the validation object for one indicator. It bundles:

- the empirical profile computed from extracted data
- the validation findings attached to that indicator

This is the authoritative definition of the data-quality surface emitted by Stage 3.

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `IndicatorAudit` | `{profile, validation}` | Empirical statistics bundled with findings |
| `validation` | `{issues: [...], checks: {check_name: status}}` | Each status is `ok`, `warning`, or `error` |
