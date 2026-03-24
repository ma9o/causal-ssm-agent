# Pipeline Execution Semantics

This page owns the cross-cutting runtime behavior of the pipeline: how stage execution is ordered, how resume works, how the research question is materialized, and how internal results differ from web-facing payloads and heavyweight artifacts.

For the conceptual cross-stage map of artifacts, assumptions, temporal semantics, scope, execution modality, and assurance surfaces, see [pipeline-dimensions.md](pipeline-dimensions.md).

## 1. Control-Flow Semantics

Execution order is not hard-coded. It is derived from a dependency DAG declared in the stage registry, where each stage declares `stage_id`, `depends_on`, `contract`, `bind_inputs`, `runner`, optional gate behavior, and optional restore/persist/finalize behavior through a materializer. The runtime computes a topological order from `depends_on` and folds over that order.

| Property | Meaning | Current stages |
|---|---|---|
| `Interactive` | User can refine or follow up through the web surface | 1a, 1b, 4, 6 |
| `Override-eligible` | Pipeline can accept a user-supplied replacement payload for the stage | 1a, 1b, 4 |
| `Hard gate` | Failure can halt downstream execution unless explicitly overridden | 1b |
| `Warning-only gate` | Failure is reported but does not halt the pipeline | 4b |
| `Always recompute on resume` | Stage is intentionally not restored from checkpoint | 5a |
| `Terminal in-place persistence` | Interactive changes persist in the current stage rather than replaying downstream stages | 6 |

### Resume Semantics

Resume restores earlier dependencies and re-executes only the requested window:

1. Resolve `start_stage` and `end_stage`.
2. Restore earlier dependencies from snapshot or reconstructed artifacts.
3. Execute only stages inside the requested window.
4. Persist fresh web payloads and snapshots for stages that reran.

Important cases: most stages restore normally from persisted state; Stage 5a is intentionally never restored; Stages 0, 2, 4, 4b, and 5b use artifact-backed restore logic.

### Question and Context Resolution

The natural-language research question is materialized to `data/{workspace_id}/query.txt`. This lets fresh runs start from web-submitted text while resume runs can reload the same question without resubmission.

### Sources

- [../pipeline.md](../pipeline.md) for the stage-facing description
- [apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py) for the executable source of truth
- [apps/data-pipeline/src/causal_ssm_agent/flows/pipeline.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/pipeline.py) for replay and resume orchestration

## 2. Persistence and Exposure Boundary

Each stage has up to three distinct persistence surfaces:

| Surface | What it contains | Consumer |
|---|---|---|
| Internal stage result | Full runtime payload, including private fields and heavyweight objects | Downstream pipeline stages |
| Public web payload | JSON-serializable subset validated by stage contracts | Web app and API routes |
| Heavy artifact | Parquet or pickle sidecar files | Resume, exploration, or downstream numerical stages |

Examples:

- Stage 0 persists raw ingested data as parquet.
- Stage 2 persists both raw observation rows and model-ready encoded data as parquet.
- Stage 5b persists the fitted result as a pickle artifact.
- All stages persist validated JSON for the web layer.

This boundary matters because the web payload is not the same thing as the full runtime result. Internal fields prefixed with `_` are stripped from the public payload, while snapshots preserve the full state for resume.

See [../pipeline.md](../pipeline.md) for the public summary and [apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py) for the concrete persistence mechanics.
