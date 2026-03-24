# ModelSpec Primitive

`ModelSpec` is the Stage 4 domain primitive that turns the causal-and-measurement handoff into a fitting-ready functional specification.

The authoritative schema lives in [Stage 4](../../pipeline/04-model-specification-priors.md). This section explains the semantic contract that sits behind that schema.

## Owns

- the parameter set to be estimated
- each parameter's role and constraint
- the likelihood choice per observed variable
- the user-facing prior proposals that accompany that parameterization

## Does Not Own

- the original construct graph
- indicator extraction windows
- compilation into executable runtime artifacts
- posterior outputs

## Reading Guide

- For the high-level translation from `CausalSpec` to `ModelSpec`, see [functional-specification.md](functional-specification.md).
- For detailed parameter, likelihood, and prior-object semantics, see [parameters-likelihoods-and-priors.md](parameters-likelihoods-and-priors.md).
- For the LLM-assisted elicitation workflow, see [prior-elicitation.md](prior-elicitation.md).
