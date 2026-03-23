# Model Runtime Handoff Map

This is the shortest path for understanding how the downstream modeling stack hands artifacts from Stage 4 through Stage 6.

## Main Chain

```text
[CausalSpec](../pipeline/01b-measurement-identifiability.md#causalspec)
  -> [ModelSpec](../pipeline/04-model-specification-priors.md#modelspec) + priors
  -> CompiledSSMArtifact
  -> SSMModelBuilder
  -> [FittedArtifact](../pipeline/05b-inference-diagnostics.md#fittedartifact)
  -> Stage 6 intervention analysis
```

## What Each Handoff Means

| From | To | Meaning |
|---|---|---|
| [`CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec) | [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) + priors | Stage 4 turns the causal-and-measurement specification into a functional specification for fitting |
| [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) + priors | `CompiledSSMArtifact` | Pure compilation translates user-facing parameters and likelihoods into an executable serializable SSM bundle |
| `CompiledSSMArtifact` | `SSMModelBuilder` | Runtime reconstruction hydrates the compiled bundle into a live builder with observation metadata |
| `SSMModelBuilder` | [`FittedArtifact`](../pipeline/05b-inference-diagnostics.md#fittedartifact) | Stage 5b fits the model and persists the builder, posterior outputs, and diagnostics needed downstream |
| [`FittedArtifact`](../pipeline/05b-inference-diagnostics.md#fittedartifact) | Stage 6 | Intervention analysis uses posterior draws and runtime metadata to answer rung-2 and rung-3 questions |

## Reading Guide

- Functional specification details: [functional-specification.md](functional-specification.md)
- Pure compilation path: [compilation.md](compilation.md)
- Estimation and counterfactual computation: [estimation.md](estimation.md)
- Inference-method routing: [inference-routing.md](inference-routing.md)
- Stage-facing handoff points: [../pipeline/04-model-specification-priors.md](../pipeline/04-model-specification-priors.md), [../pipeline/05b-inference-diagnostics.md](../pipeline/05b-inference-diagnostics.md), [../pipeline/06-intervention-analysis.md](../pipeline/06-intervention-analysis.md)
