# Runtime Semantics

This document explains how the pipeline executes, persists state, resumes work, and handles interactive edits. It is the runtime companion to [pipeline_dimensions.md](pipeline_dimensions.md). For short definitions of the main pipeline objects, see [artifact_glossary.md](artifact_glossary.md).

For the stage-by-stage payload reference, see [pipeline_stages.md](../pipeline_stages.md).

## 1. Execution Model

The pipeline is not driven by a hard-coded stage index. Execution order is derived from a dependency DAG declared in the stage registry.

Each stage declares:

- `stage_id`
- `depends_on`
- `contract`
- `bind_inputs`
- `runner`
- optional gate behavior
- optional restore/persist/finalize behavior through a materializer

The runtime computes a topological order from `depends_on` and then folds over that order.

This matters because:

- stage order is executable metadata, not duplicated documentation
- resume windows can be expressed as `start_stage` and `end_stage`
- downstream stages only rely on declared upstream dependencies

See [apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/stage_registry.py).

## 2. Stage State Surfaces

Each executed stage can exist on three surfaces at once.

| Surface | Shape | Purpose |
|---|---|---|
| Internal result | Full Python dict returned by the stage runner | Carries private fields and heavyweight objects for downstream computation |
| Web payload | Contract-validated JSON subset | Feeds the web app and result APIs |
| Snapshot | `{result, web, gate}` pickle | Enables restore on resume |

The important distinction is that the public web payload is a projection of the full result, not the result itself.

`finalize_stage()` builds the web payload by:

1. selecting only fields present in the stage contract
2. merging any extra web-facing metadata
3. validating and persisting the web JSON
4. saving the full snapshot for resume

This is why internal helper fields such as parquet paths, compiled runtime objects, or underscored scratch fields can exist in the pipeline without leaking into the web layer.

## 3. Heavy Artifacts

Some stages also persist sidecar artifacts outside the JSON payload.

| Stage | Artifact | Why it exists |
|---|---|---|
| 0 | Raw ingested dataframe parquet | Preserve the parsed source table |
| 2 | Raw observation rows parquet | Preserve canonical extracted observations |
| 2 | Model-ready parquet | Preserve encoded data used for fitting |
| 5b | Fitted artifact pickle | Preserve the fitted model and diagnostics runtime |

These artifacts are discovered by filename convention from the workspace run directory.

See [apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/run_store.py).

## 4. Resume Semantics

Resume works by restoring prior stages and then re-executing only the requested window.

At a high level:

1. resolve the requested `start_stage` and `end_stage`
2. restore earlier dependencies from snapshot or public payload plus artifact reconstruction
3. execute only stages inside the requested window
4. persist new web payloads and snapshots for the stages that ran

The restore path prefers full snapshots. If a snapshot is missing, the runtime can reconstruct state from the public web payload plus sidecar artifacts where needed.

### Normal restore

Most stages restore from prior persisted state.

### Recompute-only stage

Stage 5a is intentionally never restored. It is treated as a cheap preflight that should be rerun on resume.

### Artifact-backed restore

Stages 0, 2, 4, 4b, and 5b rely on custom restore logic because their useful runtime state is richer than the public JSON payload.

## 5. Interactive Edits

The pipeline distinguishes between three related but different ideas:

| Mechanism | Meaning | Stages |
|---|---|---|
| Interactive surface | User can refine or follow up in the UI | 1a, 1b, 4, 6 |
| Replay override | User-supplied payload replaces stage computation and downstream stages rerun from there | 1a, 1b, 4 |
| Terminal patch | User-supplied follow-up persists directly into the current stage result with no downstream replay | 6 |

This is an important distinction:

- Stages 1a, 1b, and 4 participate in replay semantics.
- Stage 6 is interactive, but terminal. It does not drive downstream re-execution because there is no downstream stage.

## 6. Gate Semantics

Gates are evaluated after stage execution and before the pipeline proceeds.

There are three practical gate modes in the current system.

| Gate mode | Meaning | Current stages |
|---|---|---|
| No gate | Stage cannot block execution | 0, 1a, 2, 3, 4, 5a, 5b, 6 |
| Hard gate | Failure blocks downstream execution unless gates are explicitly overridden | 1b |
| Warning-only gate | Failure is recorded in the stage outcome but does not halt execution | 4b |

Stage 1b is the main hard gate because non-identifiable treatment effects should not silently propagate into later causal effect estimation.

## 7. Question and Context Resolution

The natural-language research question is materialized to `data/{workspace_id}/query.txt`.

This gives the runtime two behaviors:

- fresh runs can start from web-submitted query text
- resume runs can reload the same question without the caller resupplying it

Only stages that require the question force its presence.

## 8. Public Contract Boundary

Each public stage payload is validated against an executable contract before it is written to disk.

This gives the docs and runtime the same source of truth for:

- persisted JSON shape
- frontend API expectations
- generated TypeScript models and tool schemas

See [apps/data-pipeline/src/causal_ssm_agent/flows/stages/contracts.py](../../apps/data-pipeline/src/causal_ssm_agent/flows/stages/contracts.py).

## Reading Guide

Use this document when the question is about runtime behavior rather than domain logic:

- "Why did this stage rerun?" -> this document
- "Why can Stage 6 be interactive without replay?" -> this document
- "Why does the web payload not contain the full fitted runtime?" -> this document
- "Which stages are restored vs recomputed?" -> this document
- "What does a stage persist?" -> this document, then [pipeline_stages.md](../pipeline_stages.md)
