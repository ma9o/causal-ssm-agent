# Stage 5a: SVI Preflight

Runs a cheap approximate fit as a sanity check before expensive inference.

## At a Glance

| Property | Value |
|---|---|
| Type | Computed |
| Interactive | No |
| Gate | No |
| Resume behavior | Always recomputed |

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | Stage 4 | Model spec and priors |
| `stage2.result` | Stage 2 | Model-ready data |

## Process

1. Run SVI with a fixed lightweight configuration.
2. Produce ELBO diagnostics and approximate posterior summaries.
3. Treat failure as best-effort only; the pipeline does not halt here.

## Outputs

| Output | Type | Description |
|---|---|---|
| `inference_metadata` | `{method, n_samples, duration_seconds}` | Web-facing summary metadata |
| `svi_diagnostics` | `SVIDiagnostics?` | ELBO curve and convergence metrics |
| `posterior_marginals` | `list[PosteriorMarginal]?` | Approximate marginals |
| `posterior_pairs` | `list[PosteriorPair]?` | Pairwise posterior views |

## Related Docs

- [../model-runtime/estimation.md](../model-runtime/estimation.md)
- [../runtime/execution-and-replay.md](../runtime/execution-and-replay.md)
