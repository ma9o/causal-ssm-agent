# Modeling Assumption Map

This document is the cross-cutting map for assumptions A1-A9. Each assumption constrains what can be modeled and has implications for interpretation. Detailed semantics live with the primitive that primarily owns each assumption.

These assumptions cut across the full pipeline: they shape the `LatentModel`, `MeasurementModel`, `CausalSpec`, `ModelSpec`, and estimation/runtime behavior.

For the cross-cutting pipeline map, see [pipeline-dimensions.md](pipeline-dimensions.md). If you need to locate an artifact owner quickly, see [artifact-index.md](artifact-index.md).

## Summary

| Assumption | Primary owner primitive | Main consumers | Detail page |
|---|---|---|---|
| A1. Reflective measurement model | [MeasurementModel](../primitives/measurement-model/index.md) | Stages 1b, 4 | [measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md) |
| A3. Markov property for temporal dynamics | [LatentModel](../primitives/latent-model/index.md) | Stages 1a, 4, runtime | [latent-model/assumptions.md](../primitives/latent-model/assumptions.md) |
| A3a. Latent confounders have bounded temporal reach | [CausalSpec](../primitives/causal-spec/index.md) | Stage 1b | [causal-spec/identifiability.md](../primitives/causal-spec/identifiability.md) |
| A4. Acyclicity within time slice | [LatentModel](../primitives/latent-model/index.md) | Stages 1a, 1b | [latent-model/assumptions.md](../primitives/latent-model/assumptions.md) |
| A5. Time-invariant latents as subject-level static states | [LatentModel](../primitives/latent-model/index.md) | Stage 1a, runtime | [latent-model/assumptions.md](../primitives/latent-model/assumptions.md) |
| A6. Measurement error handling depends on indicator count | [MeasurementModel](../primitives/measurement-model/index.md) | Stages 1b, 4 | [measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md) |
| A7. Measurement model identification enables causal identification | [CausalSpec](../primitives/causal-spec/index.md) | Stage 1b | [causal-spec/identifiability.md](../primitives/causal-spec/identifiability.md) |
| A8. Indicator residuals are temporally independent | [MeasurementModel](../primitives/measurement-model/index.md) | Stages 1b, 4, runtime | [measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md) |
| A9. Single-indicator constructs absorb measurement error | [MeasurementModel](../primitives/measurement-model/index.md) | Stages 1b, 4 | [measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md) |

<!-- A2 is intentionally absent. It was removed during an early revision; numbering is kept stable to avoid breaking cross-references in code and other docs. -->

## A1. Reflective Measurement Model

Primary owner: [MeasurementModel](../primitives/measurement-model/index.md).

This assumption says indicators reflect constructs rather than form them. Details live in [../primitives/measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md).

## A3. Markov Property for Temporal Dynamics

Primary owner: [LatentModel](../primitives/latent-model/index.md).

This assumption constrains valid construct-level lag structure and is later consumed by Stage 4 and the runtime. Details live in [../primitives/latent-model/assumptions.md](../primitives/latent-model/assumptions.md).

## A3a. Latent Confounders Have Bounded Temporal Reach

Primary owner: [CausalSpec](../primitives/causal-spec/index.md).

This is the identification-facing temporal bound that makes Stage 1b's finite unrolling possible. Details live in [../primitives/causal-spec/identifiability.md](../primitives/causal-spec/identifiability.md).

## A4. Acyclicity Within Time Slice

Primary owner: [LatentModel](../primitives/latent-model/index.md).

This assumption keeps contemporaneous construct structure as a DAG. Details live in [../primitives/latent-model/assumptions.md](../primitives/latent-model/assumptions.md).

## A5. Time-Invariant Latents as Subject-Level Static States

Primary owner: [LatentModel](../primitives/latent-model/index.md).

This assumption constrains what time-invariant constructs can mean and how they connect to other constructs. Details live in [../primitives/latent-model/assumptions.md](../primitives/latent-model/assumptions.md).

## A6. Measurement Error Handling Depends on Indicator Count

Primary owner: [MeasurementModel](../primitives/measurement-model/index.md).

This assumption governs the semantic difference between single-indicator and multi-indicator constructs. Details live in [../primitives/measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md).

## A7. Measurement Model Identification Enables Causal Identification

Primary owner: [CausalSpec](../primitives/causal-spec/index.md).

This assumption is the bridge from identified measurement to identified causal effects. Details live in [../primitives/causal-spec/identifiability.md](../primitives/causal-spec/identifiability.md).

## A8. Indicator Residuals Are Temporally Independent

Primary owner: [MeasurementModel](../primitives/measurement-model/index.md).

This assumption keeps temporal dependence in construct dynamics rather than indicator residuals. Details live in [../primitives/measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md).

## A9. Single-Indicator Constructs Absorb Measurement Error

Primary owner: [MeasurementModel](../primitives/measurement-model/index.md).

This assumption defines the pragmatic treatment of single-indicator constructs. Details live in [../primitives/measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md).
