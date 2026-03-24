# CausalSpec Primitive

`CausalSpec` is the Stage 1b handoff object that bundles causal structure, measurement choices, and treatment-level identifiability status.

The authoritative schema lives in [Stage 1b](../../pipeline/01b-measurement-identifiability.md). This section explains what downstream stages are allowed to assume once that artifact exists.

## Packages

- the [`LatentModel`](../latent-model/index.md)
- the [`MeasurementModel`](../measurement-model/index.md)
- the treatment-level `IdentifiabilityStatus`

## Role in the Pipeline

`CausalSpec` is the point where the system can answer two questions together:

- what causal question is being fit?
- how is each construct operationalized in observed data?

## Reading Guide

- For causal-identification semantics, see [identifiability.md](identifiability.md).
- For downstream guarantees and consumers, see [handoff-contract.md](handoff-contract.md).
