# Persistence and Exposure

For execution and replay behavior, see [execution-and-replay.md](execution-and-replay.md). For stage-facing descriptions, see [../pipeline.md](../pipeline.md).

## Stage State Surfaces

Each executed stage can exist on three surfaces at once.

| Surface | Shape | Purpose |
|---|---|---|
| Internal result | Full Python dict returned by the stage runner | Carries private fields and heavyweight objects for downstream computation |
| Web payload | Contract-validated JSON subset | Feeds the web app and result APIs |
| Snapshot | `{result, web, gate}` pickle | Enables restore on resume |

The web payload is a projection of the full result, not the result itself.

## Heavy Artifacts

Some stages also persist sidecar artifacts outside the JSON payload.

| Stage | Artifact | Why it exists |
|---|---|---|
| 0 | Raw ingested dataframe parquet | Preserve the parsed source table |
| 2 | Raw observation rows parquet | Preserve canonical extracted observations |
| 2 | Model-ready parquet | Preserve encoded data used for fitting |
| 5b | Fitted artifact pickle | Preserve the fitted model and diagnostics runtime |

These artifacts are discovered by filename convention from the workspace run directory.

See [apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py).

## Public Contract Boundary

Each public stage payload is validated against an executable contract before it is written to disk.

That keeps docs, persisted JSON shape, frontend expectations, and generated TypeScript models aligned.

See [apps/data-pipeline/src/causal_ssm_agent/flows/stages/contracts.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/stages/contracts.py).

## Why This Matters

- internal helper fields such as parquet paths, compiled runtime objects, and private scratch values can exist without leaking into the web layer
- snapshots preserve full state for resume
- sidecar artifacts preserve heavyweight numerical state that JSON should not carry
