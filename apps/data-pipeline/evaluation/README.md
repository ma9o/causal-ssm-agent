# evaluation/

Executable statistical benchmarks for the causal SSM runtime. This package is
a peer of `src/`: it consumes `nof1_causal_lab`, while production code never
imports it.

## Layout

- `benchmarks/` contains local CPU and explicitly launched Modal GPU entrypoints.
- `fixtures/synthetic_nonlinear.py` defines the shared nonlinear recovery fixture.
- `recovery/` extracts parameter-recovery diagnostics from fitted results.

Generated benchmark outputs belong in `scratchpad/` and are not committed.
GPU entrypoints spend money and must only be run when explicitly requested.
