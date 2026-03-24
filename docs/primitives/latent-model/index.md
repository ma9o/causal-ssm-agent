# LatentModel Primitive

`LatentModel` is the domain primitive that captures the theoretical causal structure over constructs before measurement choices are made.

The authoritative schema lives in [Stage 1a](../../pipeline/01a-latent-model.md). This section explains the semantic contract that sits behind that schema.

## Owns

- the construct set
- the directed edges between constructs
- the designated outcome
- explicit latent confounders as DAG nodes when unobserved common causes are part of the theory

## Does Not Own

- indicators or source columns
- observation windows or aggregation
- identifiability findings
- likelihoods, parameters, or priors

## Reading Guide

- For construct ontology and edge legality, see [constructs-and-edges.md](constructs-and-edges.md).
- For lag semantics and cross-timescale rules, see [temporal-semantics.md](temporal-semantics.md).
- For the assumptions that constrain valid latent structure, see [assumptions.md](assumptions.md).
