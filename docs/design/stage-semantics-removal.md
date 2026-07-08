# Removing Stage Semantics: Finishing the Asset-Centric Reframe

Status: **implemented** (2026-07-07). Companion to [action-hierarchy.md](action-hierarchy.md),
which established the artifact machine and deliberately deferred this cleanup.

## Goal

The artifact machine replaced the linear stage pipeline as the state model, but the stage
vocabulary (`stage-0` … `stage-6`, `StageId`, `STAGE_IDS`, `produced_by="stage-1b"`) still
survives on the wire, on disk, in the event protocol, and as the web app's internal progress
model. This document inventories every surviving location and specifies the removal plan, in
dependency order, so another agent can execute it.

The inventory below is historical: it records the old stage vocabulary that was removed.

The replacement vocabulary already exists and is already half-adopted: transitions are keyed by
the artifact they produce (`transition_id == produces` in
[`machine/graph.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py)), and
derivations already stamp artifact-named provenance (`produced_by="derive:causal_design"` in
[`machine/derivations.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/derivations.py)).
The work is finishing that rename outward.

**Ground rules** (from the repo conventions):

- No backwards compatibility, no dual-key fallbacks, no deprecation shims. Every change here is a
  clean break. Persisted fixtures are regenerated, not migrated.
- The artifact machine's semantics (guards, derivations, staleness, pins) do not change. This is
  a renaming/re-keying effort plus the deletion of one redundant frontend state model.

## Inventory: where stage semantics survive

Ordered by how deeply the semantics leak. Strata 1–3 are **load-bearing** (persisted data, wire
contracts, frontend state) and contradict the design doc; stratum 4 is naming debt that keeps
regenerating the old vocabulary.

### Stratum 1 — persisted and wire identifiers

| Location | Leak |
|---|---|
| [`machine/graph.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py) | `Transition.runner_id = "stage-N"` on every transition spec |
| [`machine/hierarchy.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/hierarchy.py) | `ContextSpec.runner_id`, `ToolQuerySpec(runner_id="stage-6", ...)`, validation against runner ids |
| [`episode_api.py`](../../apps/data-pipeline/src/nof1_causal_lab/episode_api.py) | `GET /api/machine` exposes `runner_id` per transition; docstrings instruct agents in stage terms ("before running stage-0") |
| [`machine/runners.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/runners.py) | `produced_by="stage-N"` stamped into every run-produced artifact version; `emit_stage_progress_event(workspace_id, spec.runner_id, ...)`; `ModelCompileError(..., stage_id="stage-4")` |
| [`machine/errors.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/errors.py) | `stage_id` field on machine errors |
| `data/DEMO/store/*/v*/meta.json` | Stage ids persisted **on disk**: `"produced_by":"stage-1b"` in the committed demo fixture |
| [`flows/runtime_events.py`](../../apps/data-pipeline/src/nof1_causal_lab/flows/runtime_events.py) | Event protocol: `STAGE_PROGRESS_EVENT_PREFIX = "nof1-causal-lab.pipeline-stage"`, `stage_id` in payloads, `emit_stage2_*` event emitters |
| [`apps/web/src/lib/stage-runtime.ts`](../../apps/web/src/lib/stage-runtime.ts) | Web mirror of the same event prefix and status types |
| [`apps/web/src/app/api/results/[workspaceId]/[stage]/route.ts`](../../apps/web/src/app/api/results/%5BworkspaceId%5D/%5Bstage%5D/route.ts) + [`endpoints.ts`](../../apps/web/src/lib/api/endpoints.ts) | Stage-keyed results route `/api/results/{ws}/{stage}` and `getStageResult` |
| [`apps/web/src/app/api/tools/dispatch/route.ts`](../../apps/web/src/app/api/tools/dispatch/route.ts), [`replay/route.ts`](../../apps/web/src/app/api/replay/route.ts) | `stageId` in request bodies for tool dispatch and artifact write-back |

### Stratum 2 — shared type vocabulary

| Location | Leak |
|---|---|
| [`packages/api-types/src/stages.ts`](../../packages/api-types/src/stages.ts) | `STAGE_IDS`, `StageId`, `STAGES: StageMeta[]` — including a `number: "1b"` field, the stage ordinal itself; re-exported by [`index.ts`](../../packages/api-types/src/index.ts) |
| [`packages/api-types/src/run.ts`](../../packages/api-types/src/run.ts) | `StageState[]`-shaped run types |
| [`flows/contracts_base.py`](../../apps/data-pipeline/src/nof1_causal_lab/flows/contracts_base.py) | `StageId` Literal |
| [`flows/stage_contracts.py`](../../apps/data-pipeline/src/nof1_causal_lab/flows/stage_contracts.py) | `STAGE_CONTRACTS: dict[StageId, ...]` aggregation |
| [`flows/stage_tools.py`](../../apps/data-pipeline/src/nof1_causal_lab/flows/stage_tools.py) | `STAGE_TOOLS`, `INTERACTIVE_STAGES`, `IS_INTERACTIVE_STAGE` per stage package |

### Stratum 3 — the web app still *thinks* in stages

| Location | Leak |
|---|---|
| [`pipeline-progress.ts`](../../apps/web/src/lib/hooks/pipeline-progress.ts) | Progress modeled as `Record<StageId, StageRunStatus>` with `getCurrentRunningStage` — a linear stage cursor, exactly the state model the artifact machine replaced |
| [`use-run-events.ts`](../../apps/web/src/lib/hooks/use-run-events.ts) | Builds `stageIdsByArtifact` from `GET /api/machine` purely to translate artifact moves *back* into stage updates |
| [`artifact-staleness.ts`](../../apps/web/src/lib/artifact-staleness.ts) | Groups stale artifacts "by producing stage" via `isStageId(artifact.produced_by)` |
| [`stage-result-loader.ts`](../../apps/web/src/lib/stage-result-loader.ts) | Stage-keyed loader map — the loaders already read artifact files internally; only the keys are stages |
| `apps/web/src/components/pipeline/stage-contents/stage-{0,1a,1b,2,3,4,5b,6}-content.tsx`, [`stage-section-router.tsx`](../../apps/web/src/components/pipeline/stage-section-router.tsx), `stage-section.tsx`, `active-stage-indicator.tsx`, `progress-bar.tsx`, `new-stages-notification.tsx` | Stage-keyed rendering and routing |
| `stage2-runtime.ts`, `stage4-admission-runtime.ts`, `stage4-data.ts`, `stage4-derived-data.ts`, `use-stage-data.ts`, `use-stage2-state.ts`, `use-stage4-admission.ts`, `stage0-data.ts`, `stage6-scenarios.ts` | Stage-named runtime/derived-data modules |

### Stratum 4 — code organization, config, tests, docs (labels that re-teach the old model)

| Location | Leak |
|---|---|
| `apps/data-pipeline/src/nof1_causal_lab/flows/stages/stage{0,1a,1b,2,3,4,4b,5b,6}/` | Package layout named by stage number |
| `Stage0Contract` … `Stage6Contract` and friends | Contract class names |
| `flows/llm_stage_runtime.py`, `flows/stage_tool_factory.py`, `flows/stage4_compile_cache.py` | Module names |
| [`utils/config.py`](../../apps/data-pipeline/src/nof1_causal_lab/utils/config.py) + [`config.yaml`](../../apps/data-pipeline/config.yaml) | Config keys `stage0_ingestion`, `stage1_structure_proposal`, `stage2_workers`, `stage4_prior_elicitation`, `stage6_commentary`; `stage1a_max_tool_turns` etc. |
| [`tool_server.py`](../../apps/data-pipeline/src/nof1_causal_lab/tool_server.py) | Stage-6 tool context dict keyed `"stage-1b"`, `"stage-4"`, `"stage-5b"`, `"stage-6"` |
| [`machine/temporal/`](../../apps/data-pipeline/src/nof1_causal_lab/machine/temporal) | `run_stage_activity` registered activity name, `_RUN_STAGE_TIMEOUT`, stage-phrased comments |
| [`flows/modal_runners.py`](../../apps/data-pipeline/src/nof1_causal_lab/flows/modal_runners.py) | `_run_stage_gpu` / `_run_stage_cpu` Modal function identities (payload is already artifact-named) |
| `tests/stages/stage{0,1b,2,3,4,5}/`, `tests/integration/stages/` | Test tree mirrors stage layout |
| [`scripts/validate_run.py`](../../apps/data-pipeline/scripts/validate_run.py), [`scripts/export_schemas.py`](../../apps/data-pipeline/scripts/export_schemas.py), [`scripts/export_agent_api.py`](../../apps/data-pipeline/scripts/export_agent_api.py) | Stage-keyed validation and codegen |
| [`evaluation/contracts.py`](../../apps/data-pipeline/evaluation/contracts.py) | `Stage` enum and `StageRunner` naming (values are already semantic: `identification`, `inference`) |
| `docs/pipeline/ingestion.md` … `analysis.md`, [`docs/pipeline.md`](../pipeline.md), [`docs/index.md`](../index.md) | Stage-numbered doc filenames and stage-by-stage framing |
| [`docs/guides/agent_quickstart.md`](../guides/agent_quickstart.md), [`docs/guides/agentic_integration_testing.md`](../guides/agentic_integration_testing.md) | Instructions phrased as "run stage-0", "long stages (stage-4, stage-5b)" |

## Target vocabulary

One rule: **the public identity of a run is the artifact it produces.** No parallel id space.

| Old | New |
|---|---|
| `runner_id: "stage-1b"` | *(deleted — `transition_id == produces` is already the identity)* |
| `produced_by: "stage-1b"` | `produced_by: "run:measurement_structure"` (symmetric with existing `derive:<artifact_id>`; writes stay `None` + provenance) |
| event `nof1-causal-lab.pipeline-stage.*` with `stage_id` | `nof1-causal-lab.transition.*` with `transition_id` (an `ArtifactId`) |
| `StageId` / `STAGE_IDS` / `STAGES` | `ArtifactId`-keyed `TRANSITIONS` metadata (`label`, `loadingHint`, `description`, `interactive`, `logScopePolicy`) — no `number` field; ordering, where a UI needs one, comes from a topological sort of the machine graph served by `GET /api/machine`, not from a hardcoded list |
| `ModelCompileError(stage_id=...)` | `ModelCompileError(transition_id=...)` |
| `/api/results/{ws}/{stage}` | `/api/artifacts/{ws}/{artifact_id}/view` (view payloads keyed by artifact) |
| `stageId` in dispatch/replay bodies | `contextId` (tool dispatch — contexts already have semantic ids: `ingestion`, `latent-structure`, `statistical-model-spec`, …) and `artifactId` (replay write-back — it is a `write(artifact)` move) |
| config `stage4_prior_elicitation` etc. | artifact/context-named keys: `ingestion`, `structure_proposal`, `extraction_workers`, `prior_elicitation`, `analysis_commentary` |
| `flows/stages/stageN/` | `flows/transitions/<artifact_or_context_name>/` (see step 6) |

The web UI's telemetry-only sub-events (extraction worker fan-out, model-spec admission) keep
their own prefixes but are re-keyed by context/artifact (`extraction.worker`, not
`stage2.worker`).

## Execution plan

Dependency-ordered. Steps 1–3 form one coherent breaking change and should land together (the
event protocol, provenance stamps, and machine surface are consumed jointly by the web app).
Steps 4–7 can land as follow-up commits on the same branch. Suggested worktree:
`feat-kill-stage-semantics`.

### Step 1 — delete `runner_id` from the machine surface

- Remove `runner_id` from `Transition` ([`machine/graph.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py)) and `ContextSpec` / `ToolQuerySpec` ([`machine/hierarchy.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/hierarchy.py)). `ToolQuerySpec` re-keys on `context_id`; hierarchy validation checks context ids against contexts, not runner ids.
- `GET /api/machine` ([`episode_api.py`](../../apps/data-pipeline/src/nof1_causal_lab/episode_api.py)) stops emitting `runner_id`. `transition_id` (== produced artifact) is the only id.
- `ModelCompileError.stage_id` → `transition_id` ([`machine/errors.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/errors.py)); update the raise site in [`machine/runners.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/runners.py) and the Temporal error mapping in [`machine/temporal/activities.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/temporal/activities.py).

### Step 2 — re-key provenance stamps

- In [`machine/runners.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/runners.py), every `store.write_version(..., produced_by="stage-N")` becomes `produced_by=f"run:{artifact_id}"`. Derivations already use `derive:<artifact_id>`; writes already use `None`.
- Regenerate the committed demo fixture `data/DEMO/store/` (re-run the demo episode or rewrite the `meta.json` stamps). **Do not** add read-side tolerance for old stamps.
- Update anything that parses `produced_by`: [`artifact-staleness.ts`](../../apps/web/src/lib/artifact-staleness.ts) (see step 4) and any validation in [`scripts/validate_run.py`](../../apps/data-pipeline/scripts/validate_run.py).

### Step 3 — re-key the event protocol by transition id

- [`flows/runtime_events.py`](../../apps/data-pipeline/src/nof1_causal_lab/flows/runtime_events.py): `STAGE_PROGRESS_EVENT_PREFIX` → `TRANSITION_EVENT_PREFIX = "nof1-causal-lab.transition"`; `emit_stage_progress_event(workspace_id, stage_id, status)` → `emit_transition_event(workspace_id, transition_id, status)`; `emit_stage2_plan_event` / `emit_stage2_worker_event` → `emit_extraction_plan_event` / `emit_extraction_worker_event` with `context_id: "measurement"` (or drop the id entirely — the prefix already scopes it).
- [`machine/runners.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/runners.py) emits with `artifact_id` directly — the `spec.runner_id` lookup disappears.
- Mirror in [`apps/web/src/lib/stage-runtime.ts`](../../apps/web/src/lib/stage-runtime.ts) → `transition-runtime.ts`, and update event replay/bootstrap paths ([`replay/route.ts`](../../apps/web/src/app/api/replay/route.ts), [`use-run-events.ts`](../../apps/web/src/lib/hooks/use-run-events.ts)).
- **Delete the translation layer**: `stageIdsByArtifact` in [`use-run-events.ts`](../../apps/web/src/lib/hooks/use-run-events.ts) exists only to map artifact moves back to stage ids; with artifact-keyed events it has no reason to exist.

### Step 4 — replace the shared type vocabulary and the frontend stage cursor

- [`packages/api-types/src/stages.ts`](../../packages/api-types/src/stages.ts) → `transitions.ts`: delete `STAGE_IDS`/`StageId`/`STAGES`; export `ArtifactId` (align with the machine's artifact ids) and `TRANSITION_META: Record<ArtifactId, TransitionMeta>` carrying `label`, `loadingHint`, `description`, `interactive`, `logScopePolicy`. **Delete the `number` field** — it is the stage ordinal. Update [`index.ts`](../../packages/api-types/src/index.ts) exports and [`run.ts`](../../packages/api-types/src/run.ts).
- [`pipeline-progress.ts`](../../apps/web/src/lib/hooks/pipeline-progress.ts): progress becomes `Record<ArtifactId, RunStatus>` driven by transition events and the journal; `getCurrentRunningStage` becomes "currently running transitions" (plural — the machine allows independent branches). Display order comes from the topological order served by `GET /api/machine`.
- [`artifact-staleness.ts`](../../apps/web/src/lib/artifact-staleness.ts): group by artifact id directly; `isStageId(produced_by)` checks disappear.
- [`stage-result-loader.ts`](../../apps/web/src/lib/stage-result-loader.ts): re-key the loader map by artifact id (the bodies already read artifact files).
- Components: `stage-contents/stage-N-content.tsx` → artifact-named view components (`raw-data-view.tsx`, `latent-structure-view.tsx`, `measurement-structure-view.tsx`, `measurements-view.tsx`, `validation-report-view.tsx`, `statistical-model-spec-view.tsx`, `posterior-view.tsx`, `baseline-report-view.tsx`); [`stage-section-router.tsx`](../../apps/web/src/components/pipeline/stage-section-router.tsx) routes on artifact id. Rename `stage2-runtime.ts` → `extraction-runtime.ts`, `stage4-*` → `model-spec-*`, `use-stage-data.ts` → `use-artifact-view.ts`, etc. Update stories and fixtures alongside.

### Step 5 — re-key the web API routes

- `/api/results/{ws}/{stage}` → `/api/artifacts/{ws}/{artifactId}/view`; [`endpoints.ts`](../../apps/web/src/lib/api/endpoints.ts) `getStageResult` → `getArtifactView`. Update the mock provider and route tests.
- [`tools/dispatch/route.ts`](../../apps/web/src/app/api/tools/dispatch/route.ts): body `stageId` → `contextId`, validated against the hierarchy's context ids exposed by `GET /api/machine`.
- [`replay/route.ts`](../../apps/web/src/app/api/replay/route.ts): body `stageId`/`stageData` → `artifactId`/`payload` — it is literally a `write(artifact)` move and should say so.
- Update [`tool_server.py`](../../apps/data-pipeline/src/nof1_causal_lab/tool_server.py): the analysis tool context dict keys `"stage-1b"`/`"stage-4"`/`"stage-5b"`/`"stage-6"` become artifact names (`causal_design`, `statistical_model_spec`, `posterior`, `baseline_report`).

### Step 6 — rename the interior (mechanical, can trail as separate commits)

- `flows/stages/stageN/` → `flows/transitions/<name>/` with artifact/context names: `ingestion` (stage0), `latent_structure` (stage1a), `measurement_structure` (stage1b), `extraction` (stage2), `validation` (stage3), `model_spec` (stage4, absorbing stage4b), `inference` (stage5b), `analysis` (stage6). Contract classes follow: `Stage2Contract` → `MeasurementsContract`, `Stage5bContract` → `PosteriorContract`, etc. `STAGE_CONTRACTS` → `ARTIFACT_CONTRACTS: dict[ArtifactId, ...]`; `StageId` Literal in [`contracts_base.py`](../../apps/data-pipeline/src/nof1_causal_lab/flows/contracts_base.py) is deleted in favor of the machine's `ArtifactId`.
- Module renames: `llm_stage_runtime.py` → `llm_transition_runtime.py`, `stage_tools.py` → `transition_tools.py` (`INTERACTIVE_STAGES` → `INTERACTIVE_CONTEXTS`), `stage_tool_factory.py`, `stage4_compile_cache.py` → `model_spec_compile_cache.py`.
- Config keys in [`config.yaml`](../../apps/data-pipeline/config.yaml) / [`utils/config.py`](../../apps/data-pipeline/src/nof1_causal_lab/utils/config.py): `stage0_ingestion` → `ingestion`, `stage1_structure_proposal` → `structure_proposal` (fields `stage1a_max_tool_turns`/`stage1b_max_tool_turns` → `latent_max_tool_turns`/`measurement_max_tool_turns`), `stage2_workers` → `extraction_workers`, `stage4_prior_elicitation` → `prior_elicitation`, `stage6_commentary` → `analysis_commentary`. Update `scripts/validate_config.py`, tests, and deployed config in the same change — no dual-key reading.
- Temporal: rename `run_stage_activity` → `run_transition_activity` and `_RUN_STAGE_TIMEOUT` → `_RUN_TRANSITION_TIMEOUT`. Registered activity names are contract-like with running workers — coordinate a worker redeploy; do not keep an alias.
- Modal: `_run_stage_gpu`/`_run_stage_cpu` → `_run_transition_gpu`/`_run_transition_cpu`; Modal function identity changes require redeploy, payloads are already artifact-named.
- Tests: `tests/stages/stageN/` → `tests/transitions/<name>/`; update `tests/integration/stages/`, `scripts/validate_run.py`, `scripts/export_schemas.py`, `scripts/export_agent_api.py`, and regenerate `packages/api-types/src/generated/*` via the codegen flow in [docs/guides/codegen.md](../guides/codegen.md).
- Evaluation: keep the semantic member values in [`evaluation/contracts.py`](../../apps/data-pipeline/evaluation/contracts.py); rename the `Stage` enum → `Target` (and `StageRunner` → `TargetRunner`), and scrub stage-numbered comments.
- Docs: rename `docs/pipeline/NN-*.md` to artifact-named files (`ingestion.md`, `latent-structure.md`, `measurement-structure.md`, `extraction.md`, `extraction-validation.md`, `statistical-model-spec.md`, `inference.md`, `analysis.md`), update [`docs/pipeline.md`](../pipeline.md), [`docs/index.md`](../index.md), [`agent_quickstart.md`](../guides/agent_quickstart.md), [`agentic_integration_testing.md`](../guides/agentic_integration_testing.md) to speak in transitions/artifacts ("run the `raw_data` transition", not "run stage-0"). Remove the now-obsolete Implementation Note from [action-hierarchy.md](action-hierarchy.md). Run `bun run docs:check`.

### Step 7 — guard against regression

Same pattern as the linearization guard test
([tests/models/ssm/test_linearization_init_only.py](../../apps/data-pipeline/tests/models/ssm/test_linearization_init_only.py)):
add a repo-wide vocabulary guard that fails CI if the stage vocabulary reappears.

- A test (e.g. `tests/infra/test_no_stage_vocabulary.py`) that scans `apps/data-pipeline/src/`,
  `apps/data-pipeline/scripts/`, `apps/web/src/`, and `packages/api-types/src/` for the patterns
  `\bstage-\d`, `\bStageId\b`, `\bSTAGE_IDS\b`, `\bStage\d`, `\bstage\d`, `stage_id`.
- No allowlist. If a hit is legitimate, the vocabulary was reintroduced and the test is doing its
  job.

This guard is what makes the removal permanent rather than a one-time sweep — stratum 4 naming is
harmless per se, but it is how the vocabulary re-teaches itself to every agent and contributor.

## Landing notes

Review follow-up closed these edge cases before merge:

- The committed `DEMO` fixture no longer contains numbered stage ids or `produced_by="stage-3"`;
  `validation_report` is stamped `derive:validation_report`.
- Modal function identities are `_run_transition_gpu` and `_run_transition_cpu`; redeploy changes
  the Modal function names intentionally.
- Pipeline display order is seeded from `GET /api/machine` `topological_artifact_order`; frontend
  progress tracks `runningTransitions` as a plural set instead of a singular cursor.
- Backend tests live under `tests/transitions/<name>/` and
  `tests/integration/transitions/`.
- Pipeline docs use artifact-named filenames and the agent guides speak in artifact-named
  transitions.

## Verification checklist (per landing step)

- `bun run --cwd apps/data-pipeline lint` and the relevant `uv run pytest tests/...` subsets
  (machine, infra, transitions touched). Do not run evals.
- `bun run docs:check` after any docs change.
- Web: typecheck/tests via the workspace scripts; verify against a running dev server per
  [docs/guides/agentic_integration_testing.md](../guides/agentic_integration_testing.md) — do not
  improvise the stack bring-up.
- Regenerated `data/DEMO/store` fixture loads end-to-end in the web UI (timeline, staleness
  grouping, artifact views).
- Grep-zero: `rg -n 'stage-\d|StageId|STAGE_IDS' apps packages` returns nothing outside the
  guard test itself.

## Explicit non-goals

- No changes to machine semantics: guards, derivation cascade, staleness, pins, and the return
  contract stay exactly as specified in [action-hierarchy.md](action-hierarchy.md).
- No migration/compatibility layer for old `meta.json` stamps, old event names, or old route
  shapes. Fixtures are regenerated; services are redeployed.
- No renumbering into a new ordinal scheme. There is no ordinal scheme; order is a topological
  property of the artifact graph.
