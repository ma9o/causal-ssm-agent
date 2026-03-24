# Execution and Replay

For the stage-ordered reference, see [../pipeline.md](../pipeline.md). For persistence surfaces and web/internal boundaries, see [persistence-and-exposure.md](persistence-and-exposure.md).

## Execution Model

The pipeline is not driven by a hard-coded stage index. Execution order is derived from a dependency DAG declared in the stage registry.

Each stage declares:

- `stage_id`
- `depends_on`
- `contract`
- `bind_inputs`
- `runner`
- optional gate behavior
- optional restore/persist/finalize behavior through a materializer

The runtime computes a topological order from `depends_on` and folds over that order.

See [apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py).

## Resume Semantics

Resume restores earlier dependencies and re-executes only the requested window.

At a high level:

1. Resolve `start_stage` and `end_stage`.
2. Restore earlier dependencies from snapshot or reconstructed artifacts.
3. Execute only stages inside the requested window.
4. Persist fresh web payloads and snapshots for stages that reran.

Important cases:

- Most stages restore normally from persisted state.
- Stage 5a is intentionally never restored.
- Stages 0, 2, 4, 4b, and 5b use artifact-backed restore logic.

## Interactive Edits

The runtime distinguishes between three separate behaviors.

| Mechanism | Meaning | Stages |
|---|---|---|
| Interactive surface | User can refine or follow up in the UI | 1a, 1b, 4, 6 |
| Replay override | User-supplied payload replaces stage computation and downstream stages rerun from there | 1a, 1b, 4 |
| Terminal patch | User-supplied follow-up persists directly into the current stage result with no downstream replay | 6 |

Stage 6 is interactive but terminal. It never triggers downstream execution because there is no downstream stage.

## Gate Semantics

| Gate mode | Meaning | Current stages |
|---|---|---|
| No gate | Stage cannot block execution | 0, 1a, 2, 3, 4, 5a, 5b, 6 |
| Hard gate | Failure blocks downstream execution unless overridden | 1b |
| Warning-only gate | Failure is recorded but does not halt execution | 4b |

## Question and Context Resolution

The natural-language research question is materialized to `data/{workspace_id}/query.txt`.

This lets fresh runs start from web-submitted text while resume runs can reload the same question without resubmission.
