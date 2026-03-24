# Stage 5b: Inference and Diagnostics

| Type | Interactive | Gate | Produces |
|---|---|---|---|
| estimation | No | No | fitted artifact plus PPC and sensitivity diagnostics |

Fits the model with the configured backend and runs post-fit diagnostics. Backend selection follows [../model-runtime/inference-routing.md](../model-runtime/inference-routing.md), and the fitted artifact produced here is the handoff into Stage 6 described in [../model-runtime/handoff-map.md](../model-runtime/handoff-map.md).

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | Stage 4 | Model spec and priors |
| `stage2.result` | Stage 2 | Model-ready data |
| `inference_method` | Pipeline config | Optional sampler override |

## Process

1. Fit the model with the configured or auto-routed backend.
2. Run power-scaling sensitivity analysis after fitting.
3. Run posterior predictive checks.
4. Persist the fitted artifact for Stage 6 and resume; see [../runtime/persistence-and-exposure.md](../runtime/persistence-and-exposure.md) for the persistence surfaces involved.

## Outputs

| Output | Type | Description |
|---|---|---|
| `power_scaling` | `list[PowerScalingResult]` | Per-parameter sensitivity diagnosis |
| `ppc` | `PPCResult` | Posterior predictive checks and warnings |
| `inference_metadata` | `{method, n_samples, duration_seconds}` | Web-facing summary metadata |
| `mcmc_diagnostics` | `MCMCDiagnostics?` | NUTS or NUTS-DA diagnostics |
| `svi_diagnostics` | `SVIDiagnostics?` | SVI diagnostics |
| `smc_diagnostics` | `SMCDiagnostics?` | SMC diagnostics |
| `loo_diagnostics` | `LOODiagnostics?` | Pareto-k diagnostics |
| `posterior_marginals` | `list[PosteriorMarginal]?` | Marginal distributions |
| `posterior_pairs` | `list[PosteriorPair]?` | Pairwise posterior views |

## Artifact Introduced

### FittedArtifact

`FittedArtifact` is the persisted fitted runtime object produced by Stage 5b. It owns:

- the inference result
- the runtime builder needed downstream
- timing and observation-support metadata
- post-fit diagnostics attached to the fitted run

This is the authoritative definition of the object consumed by Stage 6 and used during resume.
