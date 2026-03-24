# MeasurementModel Primitive

`MeasurementModel` is the domain primitive that explains how constructs are observed in data.

The authoritative schema lives in [Stage 1b](../../pipeline/01b-measurement-identifiability.md). This section explains the semantic contract that sits behind that schema.

## Owns

- the indicator list
- the construct-to-indicator mapping
- extraction mode
- measurement dtype
- aggregation semantics
- observation-window semantics
- the `model_clock` used by extraction and later discretization

## Does Not Own

- construct-to-construct causal edges
- treatment-level identifiability status
- fitted likelihood parameters or priors
