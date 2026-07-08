---
name: nof1-episode-api
description: "Drive or inspect the nof1-causal-lab episode state machine over HTTP with curl: run pipeline stages, write judgment artifacts (latent structure, causal design, priors), read episode state/timeline/artifacts, and invoke stage tools against the tool server. Use when navigating the episode machine as an external agent instead of the web viewer."
---

# nof1-causal-lab episode API — curl skill

> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI OpenAPI spec) by `apps/data-pipeline/scripts/export_agent_api.py`. Edit the route docstrings, not this file.

The episode machine is the single interface to an N-of-1 causal analysis. An
external agent drives it entirely over this HTTP API — the same surface the web
viewer uses. There is no SDK and no MCP server: `curl` is the interface.

## Orientation

Call `GET /api/machine` once. It returns the static artifact graph — every
transition with what it consumes, produces, and optionally co-produces — plus
each transition's creation class and the derivation graph:

- `deterministic` — pure compute, no credentials (e.g. identification).
- `batch_llm` — bulk LLM compute on the service's ambient key. You trigger it
  with a `run` move; you never supply a key.
- `judgment` — proposal work you can do yourself by writing the produced
  artifact directly. These transitions are flagged `writable`.

## The loop

1. `GET /api/machine` once, then `GET /api/episodes/{workspace_id}` for the live
   state: per-artifact freshness, the legal moves, and whether an auto-run is
   active.
2. Propose a move at `POST /api/episodes/{workspace_id}/moves` — either
   `{"move": {"kind": "run", "artifact_id": "latent_structure"}}` to run a transition, or
   `{"move": {"kind": "write", "artifact_id": "latent_structure", "provenance": "llm"}, "payload": {...}}`
   to author a judgment artifact directly.
3. Long transitions (`statistical_model_spec`, `posterior` — minutes to hours) can outlive a client
   timeout. Prefer `POST /api/episodes/{workspace_id}/auto` (a background driver
   that runs enabled transitions in dependency order) and poll the state.
4. Read what happened at `GET /api/episodes/{workspace_id}/timeline`: `applied`,
   `rejected` (illegal, state unchanged), or `raised` (typed transition error).

## Staleness

A `write` becomes a new provenance root and marks everything downstream stale
until re-run. Numeric tools (`simulate`, `get_model_info`) hard-flag
stale provenance chains in their warnings — never report numbers past those
flags.

## Data in, results out

Raw data enters by placing files under `data/{workspace_id}/input/` before
running the `raw_data` transition. Read artifact payloads at
`GET /api/episodes/{workspace_id}/artifacts/{artifact_id}`; binary files
(parquet, pickle) are served individually from `.../files/{filename}`.

## Read-only deployments

The hosted viewer's backend serves these same read endpoints against a published
store with no move plane. `GET /api/capabilities` reports `moves_enabled`; every
move returns 403 when it is `false`.

## Endpoints

### GET `/api/capabilities`

Whether this deployment serves the move plane.

`moves_enabled` is `false` on the hosted read-only viewer backend, where
every `POST` (moves, auto-run, start-episode) returns 403 and only the read
endpoints are live.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/capabilities"
```

### POST `/api/episodes`

Ensure the episode workflow exists; optionally seed the `question` root.

Idempotent: attaches to an existing episode or starts a fresh one. Passing
`question` writes the `question` root artifact with `human` provenance. Raw
data enters separately by placing files under `data/{workspace_id}/input/`
before running the `raw_data` transition. Returns the same shape as
`GET /api/episodes/{id}`.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"workspace_id": "string"}'
```

### GET `/api/episodes/{workspace_id}`

Current episode state: the single read to poll while navigating.

Returns per-artifact freshness (existence, staleness, version, provenance),
the `legal` moves available right now, and `auto_running` — whether the
background driver is active. Journal-backed, so it works even against a
published read-only store.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID"
```

### GET `/api/episodes/{workspace_id}/artifacts/{artifact_id}`

One artifact version: meta + inline JSON payloads.

Defaults to the episode's *current* version (the journal projection,
not merely the highest on disk). Binary payload files (parquet,
pickle) are listed by name, never inlined.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `version` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/artifacts/ARTIFACT_ID"
```

### GET `/api/episodes/{workspace_id}/artifacts/{artifact_id}/files/{filename}`

One declared payload file from an artifact version.

Defaults to the episode's current version. Unlike the JSON artifact
endpoint, this serves binary files as bytes and refuses undeclared
filenames so callers cannot browse arbitrary workspace paths.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `filename` (path, required)
- `version` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/artifacts/ARTIFACT_ID/files/FILENAME"
```

### POST `/api/episodes/{workspace_id}/auto`

Start the default navigation policy in the background.

Runs enabled stages in dependency order while their outputs are missing or
stale, stopping when quiescent or when a move fails. Returns immediately;
follow progress with `GET /api/episodes/{workspace_id}` (`auto_running`) and
the timeline. An LLM navigator replaces this policy by proposing `moves`
itself. 409 if a driver is already active for this workspace.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/auto" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{}'
```

### GET `/api/episodes/{workspace_id}/events`

Fine-grained telemetry (e.g. extraction worker fan-out, transition progress).

Pass the last-seen event id as `after` to page forward; omit it for the full
stream. This is finer-grained than the timeline, which records only whole
move outcomes.

**Parameters**

- `workspace_id` (path, required)
- `after` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/events"
```

### POST `/api/episodes/{workspace_id}/moves`

Propose one move; blocks until it is applied, rejected, or raises.

Two kinds:

- Run a transition: `{"move": {"kind": "run", "artifact_id": "latent_structure"}}`.
- Author a judgment artifact directly (skip the in-service stage):
  `{"move": {"kind": "write", "artifact_id": "latent_structure", "provenance":
  "llm"}, "payload": {...}}`. The payload is schema-validated against that
  artifact's contract, journaled, and provenance-stamped; the write becomes a
  new provenance root and marks everything downstream stale until re-run.

The synchronous outcome is the same record the timeline stores. Long transitions
(statistical model specification, posterior — minutes to hours) can outlive a client timeout; for
those prefer `POST /api/episodes/{workspace_id}/auto` plus polling.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/moves" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"move": {"kind": "run", "artifact_id": "question"}}'
```

### GET `/api/episodes/{workspace_id}/timeline`

The transition journal: every move attempt in order.

Each record is `applied` (state advanced), `rejected` (illegal move, state
unchanged), or `raised` (the transition ran but threw — the record carries the
typed error). Re-running after a `raised`/`rejected` is just proposing the
move again.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/timeline"
```

### GET `/api/machine`

The static artifact graph and action hierarchy — read once to orient.

Each transition entry declares what it `consumes`, `produces`, and
optionally co-produces (`produces_optional`), plus its **creation
class**: `deterministic` (pure compute, no credentials), `batch_llm` (bulk
LLM compute on the service's ambient key — you trigger it with a `run` move,
you never supply a key), or `judgment` (proposal work you can author yourself
by writing the produced artifact directly — these are flagged `writable`).

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/machine"
```

### GET `/api/tools/{context_id}`

List a context's validation/query tools — the same tools the in-service LLM loops use.

Each entry is `{name, description, parameters, result}` where `parameters`
and `result` are JSON Schemas. Fetch this first to learn a tool's argument
shape, then call `POST /api/tools/{context_id}/{tool_name}`. Examples:
ranking `simulate` / `get_model_info`, statistical-model-spec `submit_statistical_model_spec`.

**Parameters**

- `context_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/tools/CONTEXT_ID"
```

### POST `/api/tools/{context_id}/{tool_name}`

Execute a context tool against the workspace's current artifact-store versions.

Body is `{"workspace_id": "...", "input": {...}}` where `input` matches the
tool's `parameters` schema from `GET /api/tools/{context_id}`; 422 on a schema
violation. Numeric tools hard-flag stale provenance chains in their result
warnings — do not report numbers past those flags.

**Parameters**

- `context_id` (path, required)
- `tool_name` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/tools/CONTEXT_ID/TOOL_NAME" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"workspace_id": "string", "input": {}}'
```
