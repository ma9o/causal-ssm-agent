# Cached development inference

Use [`scripts/cached_fit.py`](../../apps/data-pipeline/scripts/cached_fit.py) to run
production inference on Modal while retaining expensive development artifacts. The
runner lives in `scripts/` because cache orchestration is not part of the production
pipeline. It calls the same runtime preparation, Pathfinder, IEKS initialization, exact
Euler–Maruyama particle sampler, diagnostics, and posterior-predictive code as production.

## Inputs

The runner accepts:

- a compiled `CompiledSSMArtifact` JSON file;
- the long-form model panel serialized either as Polars binary (`.bin`) or Parquet;
- an optional JSON object of sampler overrides.

Create a Polars binary panel with:

```python
from pathlib import Path

Path("panel.bin").write_bytes(data_for_model.serialize(format="binary"))
```

The compiled artifact must be its complete `model_dump(mode="json")`; a partial model
specification is not sufficient.

## Run

From `apps/data-pipeline`:

```bash
uv run modal run scripts/cached_fit.py \
  --compiled-ssm ../../scratchpad/fit-input/compiled_ssm.json \
  --panel ../../scratchpad/fit-input/panel.bin \
  --label nine-construct
```

To change fit budgets without invalidating Pathfinder, provide an override file:

```json
{
  "num_warmup": 1000,
  "num_samples": 500,
  "n_particles": 32
}
```

```bash
uv run modal run scripts/cached_fit.py \
  --compiled-ssm ../../scratchpad/fit-input/compiled_ssm.json \
  --panel ../../scratchpad/fit-input/panel.bin \
  --sampler-overrides ../../scratchpad/fit-input/sampler_overrides.json \
  --label nine-construct-short
```

## Persisted layers

| Layer | Modal volume | Identity |
| --- | --- | --- |
| JAX executables | `nof1-cached-fit-cache:/jax` | JAX compilation key, including computation and platform |
| Pathfinder and IEKS warmup | `nof1-cached-fit-cache:/pathfinder/<fingerprint>` | Target, data, source environment, and warmup policy |
| Fit results | `nof1-cached-fits:/runs/<timestamp>--<label>--<input-prefix>` | Immutable run directory |

The warmup entry contains Pathfinder’s selected mean and covariance Cholesky, exact
per-chain initial positions, the parameter proposal scaling matrix, and—when the
production `paid_mix` leaf is selected—the IEKS reference trajectories. It is committed
before particle MCMC starts, so it remains available if the later fit fails.

Each result directory contains the compiled artifact, original panel bytes, resolved
sampler configuration, cache provenance, posterior samples, retained latent paths,
serialized posterior, MCMC diagnostics, PPC output, and a compact summary. Inference
outputs are committed before PPC runs.

## Invalidation

The warmup cache is content-addressed and never silently reused across mismatched inputs.

| Change | Reuse Pathfinder? |
| --- | --- |
| Compiled model, priors, or structural topology | No |
| Panel values, support metadata, or serialization format | No |
| Inference source, runner source, dependency lock, or application config | No |
| Chain count, seed, IEKS settings, or Pathfinder settings | No |
| Warmup or retained-sample count | Yes |
| Particle count | Yes |

Changing a particle-kernel shape may still require a new JAX compilation even when the
Pathfinder artifact remains valid. JAX handles that cache independently.

Use `--no-reuse-warmup` to run the ordinary production warmup without reading or writing
the Pathfinder cache. Cache corruption, incomplete entries, and metadata mismatches are
errors rather than cache misses.

## Retrieve results

List completed runs:

```bash
uv run modal volume ls nof1-cached-fits /runs
```

Download one run:

```bash
uv run modal volume get \
  nof1-cached-fits \
  /runs/<run-name> \
  ../../scratchpad/cached-fits/<run-name>
```

Cached execution does not relax convergence or causal-identification requirements.
Posterior diagnostics and PPC must still pass before any numeric claim is reported.
