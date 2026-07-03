# Agentic Integration Testing

## Design Principles

| Concern | Mode | Why |
|---------|------|-----|
| File placement, episode moves, run registration | **Programmatic** (`curl`) | Reliable, fast, no UI fragility |
| Runtime errors, route discovery, build state | **Next.js devtools MCP** | Reads the running app directly and catches server/runtime errors before UI debugging |
| UI rendering verification, visual regression | **browser automation** (`browser_eval` / Playwright) | Only way to see rendered output |

## Prerequisites

All backend processes that might touch BYOK replay/refinement paths must source the
same root `.env` as the web app so `APP_SECRET` is available everywhere, not just
in Next.js.

```bash
bun run integration:start
```

Starts the Temporal dev server on port `7233` (ephemeral state, binary
auto-downloaded on first use), the episode worker (task queue
`nof1-episodes`), the tool server with the episode facade on port
`8100`, and the web app on port `3000`. The script owns all child
processes via process groups and **stays in the foreground** — wait for
the `Stack ready` banner before proceeding.

To tear down the stack, kill the script (`Ctrl+C`, or `kill <pid>` if
backgrounded). All child processes are cleaned up automatically.

## Workspace Layout

```text
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Raw uploaded files for stage 0
│   ├── store/             # Versioned artifact store ({artifact}/v{N}/)
│   ├── episode/           # Transition journal, state read model, telemetry events
│   └── run/               # Web-facing stage JSON projection
├── DEMO/                  # Tracked mock fixture workspace and stage 0-2 golden fixture
├── GOLDEN/                # Default tracked workspace for evals and manual sampling
└── SMALLGOLDEN/           # Smaller tracked workspace for quicker eval iteration
```

## Step-by-Step Flow

### 1. Create the workspace and start the run

```bash
COOKIE_JAR=$(mktemp)
WORKSPACE_ID="T3ST42"
QUESTION="How does screen time affect sleep?"
LAUNCH_ID="launch-1"

curl -s -c "$COOKIE_JAR" -X POST http://localhost:3000/api/upload \
  -F "workspaceId=$WORKSPACE_ID" \
  -F "file=@data/GOLDEN/input/MyActivity.json"

curl -sf -b "$COOKIE_JAR" http://localhost:3000/api/auth/status \
  | jq -e '.canRun == true'

curl -s -b "$COOKIE_JAR" -X POST http://localhost:3000/api/runs \
  -H 'Content-Type: application/json' \
  -d "{\"workspaceId\":\"$WORKSPACE_ID\",\"launchId\":\"$LAUNCH_ID\",\"query\":\"$QUESTION\"}"
```

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
and journaled. Editing `causal_spec` fans out recomputed
`identification_report`/`estimands` artifacts automatically; downstream
artifacts become **stale** (visible in `.artifacts`), and the next
auto-run recomputes exactly the stale suffix.

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

Through the web app, [`/api/refine/apply`](../../apps/web/src/app/api/refine/apply/route.ts)
and [`/api/replay`](../../apps/web/src/app/api/replay/route.ts) perform the
same write-then-auto sequence with payload merging.

### Valid stage IDs

Dependencies are artifact-level (see
`apps/data-pipeline/src/nof1_causal_lab/machine/graph.py` for the
authoritative graph); the default topological order is:

```text
stage-0 → stage-1a → stage-1b → stage-2 → stage-3 → stage-4 → stage-5b → stage-6
```

Note the machine is not a tape: any stage whose consumed artifacts exist
can run, and `estimands`/`model_data` are produced only when nonempty —
their absence structurally disables the fit chain.
