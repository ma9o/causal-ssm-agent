# CausalSpec: Handoff Contract

This page explains what downstream stages can rely on once `CausalSpec` has been produced.

## Downstream Consumers

| Consumer | What it takes from `CausalSpec` |
|---|---|
| Stage 2 extraction | indicator list, extraction mode, aggregation, observation-window semantics, `model_clock` |
| Stage 3 validation | indicator identity and construct mapping for audit grouping |
| Stage 4 model specification | latent structure, measurement dtype, loading structure, edge lags, identifiability status |
| Stage 6 intervention analysis | the original causal target and treatment eligibility context |

## Guarantees

Once a `CausalSpec` exists, downstream stages may assume:

- the causal graph and measurement definition refer to the same construct set
- indicator metadata is structurally valid for extraction and compilation
- treatment-level identifiability status has already been checked under the Stage 1b assumptions

## Non-Guarantees

`CausalSpec` does not yet choose:

- likelihood families beyond dtype-level measurement semantics
- parameter roles and constraints
- prior distributions
- whether the resulting functional specification is parametrically recoverable

Those belong to `ModelSpec` and the later assurance stages.
