# Agentic Integration Testing

## Design Principles

| Concern | Mode | Why |
|---------|------|-----|
| File placement, episode moves, run registration | **Programmatic** (`curl`) | Reliable, fast, no UI fragility |
| Runtime errors, route discovery, build state | **Next.js devtools MCP** | Reads the running app directly and catches server/runtime errors before UI debugging |
| UI rendering verification, visual regression | **browser automation** (`browser_eval` / Playwright) | Only way to see rendered output |

## Prerequisites

```bash
bun run integration:start
```

Starts the stack under [process-compose](https://github.com/F1bonacc1/process-compose)
supervision (`brew install f1bonacc1/tap/process-compose`; config:
[`process-compose.yaml`](../../process-compose.yaml)): the Temporal dev
server on port `7233` (ephemeral state, binary auto-downloaded on first
use), the episode worker (task queue `nof1-episodes`), the tool server
with the episode facade on port `8100`, and the web app on port `3000`.
Startup order is health-gated (`depends_on` + readiness probes) and
crashed processes restart automatically. The script **stays in the
foreground** — wait until `curl -s http://localhost:8100/api/capabilities`
answers before proceeding (pass `-t=false` to disable the TUI when
redirecting output to a file).

To tear down the stack, kill the script (`Ctrl+C`, or `kill <pid>` if
backgrounded). All child processes are cleaned up automatically.

The process-compose API is pinned to port `8181` for targeted operations
against the running stack:

```bash
process-compose --port 8181 process list
process-compose --port 8181 process restart worker
```

The worker caches `apps/data-pipeline/config.yaml` at first read
(`lru_cache`), so after editing pipeline config, restart the `worker`
process — no need to bounce the whole stack. The Temporal dev server
persists its event history to `.local/agentic-integration-stack/temporal.db`
(the `--db-filename` on its command), so restarting the `temporal`
process — to serve the UI, pick up a change, or recover from a crash —
**resumes** in-flight episode workflows exactly where they left off
rather than orphaning them. To start genuinely fresh, delete that
`temporal.db` before boot (and wipe the workspace's `store/` + `episode/`
dirs). The Web UI is served at `http://localhost:8233`.

## Workspace Layout

```text
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Raw uploaded files for stage 0
│   ├── store/             # Versioned artifact store ({artifact}/v{N}/)
│   ├── episode/           # Transition journal, state read model, telemetry events
│   └── run/               # Internal sidecars only, such as stage-4 compile cache
└── DEMO/                  # Tracked mock fixture workspace (evals + manual sampling)
```

## Step-by-Step Flow

### 1. Create the workspace and start the run

```bash
WORKSPACE_ID="T3ST42"
QUESTION="How does screen time affect sleep?"

curl -s -X POST http://localhost:3000/api/upload \
  -F "workspaceId=$WORKSPACE_ID" \
  -F "file=@data/DEMO/input/dsar_bundle.zip"

curl -s -X POST http://localhost:3000/api/runs \
  -H 'Content-Type: application/json' \
  -d "{\"workspaceId\":\"$WORKSPACE_ID\",\"query\":\"$QUESTION\"}"
```

There is no auth: the facade is the source of truth for what is allowed, and
`curl -s http://localhost:8100/api/capabilities` reports whether the move
plane is available (`moves_enabled` is `false` on a read-only facade).

This writes the `question` artifact and starts the **auto-run driver**: a
default navigation policy that proposes `run(stage)` moves in dependency
order while stage outputs are missing or stale, stopping when the episode
is quiescent or a move fails.

### 2. Observe the episode

The episode facade (tool server, port `8100`) is the source of truth:

```bash
# Current state: artifact existence/staleness/versions + legal moves
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID | jq '.artifacts'

# The transition journal: every move attempt (applied / rejected / raised)
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID/timeline \
  | jq '.transitions[] | {seq, status, move, error_type}'

# Intra-stage telemetry (stage-2 worker fan-out, stage-4 agent graph)
curl -s "http://localhost:8100/api/episodes/$WORKSPACE_ID/events" | jq '.events[-3:]'
```

### 3. Verify via browser automation

Navigate to `http://localhost:3000/analysis/{WORKSPACE_ID}`, screenshot the progress bar, then poll and screenshot each stage section as it completes through the final "Complete" badge.

If the UI behaves unexpectedly, check Next.js devtools MCP errors before debugging the browser script.

## Resuming After a Stage Failure

A failed stage run is a `"raised"` transition in the journal — state is
unchanged, and the typed error plus diagnostics ride on the record:

```bash
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID/timeline \
  | jq '.transitions[] | select(.status=="raised") | {seq, move, error_type, error_message}'
```

Re-running is just proposing the move again (the machine validates
enabledness; there is no window arithmetic):

```bash
# Run one stage
curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/moves \
  -H 'Content-Type: application/json' \
  -d '{"move": {"kind": "run", "stage_id": "stage-3"}}'

# Or resume the default policy (runs everything enabled and stale/missing)
curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/auto \
  -H 'Content-Type: application/json' -d '{}'
```

The question does not need re-supplying: it is a versioned root artifact,
not a run parameter.

## Editing artifacts (replaces stage overrides)

A human/LLM edit is a `write` move: schema-validated, provenance-stamped,
and journaled. Editing `causal_spec` fans out a recomputed positive
`identification_report` when the spec has explicitly identified treatments;
downstream artifacts become **stale** (visible in `.artifacts`), and the
next auto-run recomputes exactly the stale suffix.

```bash
curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/moves \
  -H 'Content-Type: application/json' \
  -d '{
    "move": {"kind": "write", "artifact_id": "causal_spec", "provenance": "human"},
    "payload": {"causal_spec": {"latent": {...}, "measurement": {...}}}
  }'

curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/auto \
  -H 'Content-Type: application/json' -d '{}'
```

Through the web app, [`/api/replay`](../../apps/web/src/app/api/replay/route.ts)
performs the same write-then-auto sequence; an external agent proposes the
same moves over MCP (see the [agent quickstart](agent_quickstart.md)).

### Valid stage IDs

Dependencies are artifact-level (see
`apps/data-pipeline/src/nof1_causal_lab/machine/graph.py` for the
authoritative graph); the default topological order is:

```text
stage-0 → stage-1a → stage-1b → stage-2 → stage-3 → stage-4 → stage-5b → stage-6
```

Note the machine is not a tape: any stage whose consumed artifacts exist
can run, and `identification_report`/`model_data` are produced only when
nonempty — their absence structurally disables the fit chain.
