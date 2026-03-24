# Modeling Assumption Map

## Summary

| Assumption | Primary owner primitive | Main consumers | Detail page |
|---|---|---|---|
| A1. Reflective measurement model | [MeasurementModel](../primitives/measurement-model/index.md) | Stages 1b, 4 | [measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md) |
| A3. Markov property for temporal dynamics | [LatentModel](../primitives/latent-model/index.md) | Stages 1a, 4, runtime | [latent-model/assumptions.md](../primitives/latent-model/assumptions.md) |
| A3a. Latent confounders have bounded temporal reach | [CausalSpec](../primitives/causal-spec/index.md) | Stage 1b | [causal-spec/identifiability.md](../primitives/causal-spec/identifiability.md) |
| A4. Acyclicity within time slice | [LatentModel](../primitives/latent-model/index.md) | Stages 1a, 1b | [latent-model/assumptions.md](../primitives/latent-model/assumptions.md) |
| A4b. Endogenous time-varying directed effects are drift-mediated | [LatentModel](../primitives/latent-model/index.md) | Stages 1a, 4, runtime | [latent-model/assumptions.md](../primitives/latent-model/assumptions.md) |
| A5. Time-invariant latents as subject-level static states | [LatentModel](../primitives/latent-model/index.md) | Stage 1a, runtime | [latent-model/assumptions.md](../primitives/latent-model/assumptions.md) |
| A6. Measurement error handling depends on indicator count | [MeasurementModel](../primitives/measurement-model/index.md) | Stages 1b, 4 | [measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md) |
| A7. Measurement model identification enables causal identification | [CausalSpec](../primitives/causal-spec/index.md) | Stage 1b | [causal-spec/identifiability.md](../primitives/causal-spec/identifiability.md) |
| A8. Indicator residuals are temporally independent | [MeasurementModel](../primitives/measurement-model/index.md) | Stages 1b, 4, runtime | [measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md) |
| A9. Single-indicator constructs absorb measurement error | [MeasurementModel](../primitives/measurement-model/index.md) | Stages 1b, 4 | [measurement-model/assumptions.md](../primitives/measurement-model/assumptions.md) |

<!-- A2 is intentionally absent. It was removed during an early revision; numbering is kept stable to avoid breaking cross-references in code and other docs. -->
