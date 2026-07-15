# Running Evaluation Benchmarks

The maintained evaluation surface is the nonlinear SSM benchmark suite under
`apps/data-pipeline/evaluation/benchmarks/`. The retired Inspect tasks were
removed because they imported transition modules that no longer exist and used
obsolete worker APIs.

From `apps/data-pipeline/`, the local CPU benchmark entrypoint is:

```bash
uv run python evaluation/benchmarks/benchmark.py
```

Use `--help` to inspect its current model, sampler, and output options. The
benchmark writes generated artifacts under `scratchpad/`.

GPU profiling and recovery entrypoints are
`evaluation/benchmarks/benchmark_gpu.py` and
`evaluation/benchmarks/benchmark_recovery_gpu.py`. They launch paid Modal GPU
work and must only be run when explicitly requested.
