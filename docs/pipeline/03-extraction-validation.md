# Stage 3: Extraction Validation

Audits extracted observations and computes empirical profiles per indicator.

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

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `IndicatorAudit` | `{profile, validation}` | Empirical statistics bundled with findings |
| `validation` | `{issues: [...], checks: {check_name: status}}` | Each status is `ok`, `warning`, or `error` |

## Related Docs

- [../concepts/artifact-glossary.md](../concepts/artifact-glossary.md)
- [../concepts/pipeline-dimensions.md](../concepts/pipeline-dimensions.md)
