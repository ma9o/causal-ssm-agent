# Storage Lifecycle and Commit Boundaries

Each workspace has three storage tiers with different correctness and retention rules:

```text
data/{workspace_id}/
├── input/                         durable user inputs
├── store/{artifact_id}/v{N}/     durable immutable artifact versions
├── episode/
│   ├── journal/{seq}.json        durable transition commit log
│   └── traces/{seq}/{id}.json    durable promoted LLM traces
├── cache/                         regenerable computation reuse
└── scratch/
    ├── events/                    live UI telemetry
    └── runs/{run_id}/             run-scoped execution state and checkpoints
```

## Durable Ledger

Artifact activities write immutable versions before returning their transition effects. A version becomes current only when an `applied` record containing those effects is appended to the episode journal. Current state is therefore reconstructed by journal replay; there is no separately written latest-state manifest that can drift from the log.

Finalized LLM traces follow the same write-before-commit rule. The journal activity discovers every finalized `llm/{subroutine_id}/trace.json` owned by the sequence's scratch run, promotes them to `episode/traces/{seq}/`, then appends their subroutine IDs. Artifact payloads and workflow failures do not carry trace paths, so published artifacts never depend on disposable execution files.

A raised model-spec transition may contain a typed `(run_id, checkpoint_id)` resume selection. The checkpoint module alone resolves that selection into scratch storage. The collector preserves the selected run until a later model-spec transition supersedes it; artifact lineage and trace references remain closed within the durable tier.

## Scratch and Checkpoints

Everything produced while a transition executes lives below one `scratch/runs/{run_id}/` directory. Conversations, tool exchanges, staged contexts, extraction chunks, and model-spec checkpoints are one collection unit.

Checkpoints are not cache entries. They preserve paid semantic work needed to resume a raised model-spec transition, so run collection protects the selection referenced by the latest raised attempt. A later model-spec attempt releases or replaces that selection. Automatic collection runs under the episode move lock; offline collection is explicit and makes no filename-based liveness inference.

The UI event stream lives under `scratch/events/`. It reports intra-transition progress but never participates in state reconstruction. The journal is the durable account of move outcomes; telemetry can expire without changing episode state.

## Cache

Cache entries are safe to delete at any time. Admission evaluations are content-addressed by all semantic inputs and reject hash collisions. JAX compilation sidecars are guarded by their topology fingerprint and schema version. Cache collection applies both age and total-size bounds.

## Collection and Publication

The episode workflow collects completed run scratch after every journal append while its per-episode move lock is still held. Collection failure is logged but never changes the already-committed move outcome. `uv run nof1-sweep WORKSPACE_ID` expires telemetry and caches without guessing whether a run is active; offline maintenance may additionally pass `--collect-runs`.

Publishing copies only `input/`, `store/`, and `episode/`. It excludes `scratch/` and `cache/`, which keeps hosted read-only workspaces self-contained and prevents runtime conversations, checkpoints, or telemetry from becoming public accidentally.
