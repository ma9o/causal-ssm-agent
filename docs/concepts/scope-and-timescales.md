# Scope and Timescale Map

This framework models dynamics of time-varying constructs with optional time-invariant covariates for causal effect estimation on single-individual or already-aggregated longitudinal data. The detailed time semantics live with the primitives that own them.

## Scope Boundary

In scope:

- time-varying constructs with optional time-invariant covariates
- explicit measurement definitions for every construct
- causal reasoning that can stop at structure when numeric identification is not justified

Out of scope:

- trajectory estimation for unmeasured constructs; every construct must have at least one indicator
- user-facing bidirected-edge representations instead of explicit latent confounder nodes

Latent state filtering is used internally for likelihood computation, but the framework's outputs are causal effect estimates rather than state-trajectory products.

## Where Temporal Semantics Live

Identifiability is checked by y0 in Stage 1b rather than enforced at the schema level. The table below points to the primitive-owned pages that define the temporal semantics used in that check.

| Question | Primary owner | Detail page |
|---|---|---|
| What is a construct, and which edges are legal between constructs? | [LatentModel](../primitives/latent-model/index.md) | [latent-model/constructs-and-edges.md](../primitives/latent-model/constructs-and-edges.md) |
| How do lag rules work at the construct level? | [LatentModel](../primitives/latent-model/index.md) | [latent-model/temporal-semantics.md](../primitives/latent-model/temporal-semantics.md) |
| How do indicators define support windows, aggregation, and `model_clock`? | [MeasurementModel](../primitives/measurement-model/index.md) | [measurement-model/windows-and-aggregation.md](../primitives/measurement-model/windows-and-aggregation.md) |
| How does temporal unrolling affect causal identification? | [CausalSpec](../primitives/causal-spec/index.md) | [causal-spec/identifiability.md](../primitives/causal-spec/identifiability.md) |
| How is elapsed `dt` used in continuous-to-discrete runtime transitions? | Runtime estimation | [../model-runtime/estimation.md](../model-runtime/estimation.md) |
