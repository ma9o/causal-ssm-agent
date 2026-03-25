# Agentic Integration Testing

## Design Principles

The methodology splits responsibilities between two modes:

| Concern | Mode | Why |
|---------|------|-----|
| File placement, pipeline trigger, run registration | **Programmatic** (`cp`, `curl`) | Reliable, fast, no UI fragility |
| Runtime errors, route discovery, build state | **Next.js devtools MCP** | Reads the running app directly and catches server/runtime errors before UI debugging |
| UI rendering verification, visual regression | **browser automation** (`browser_eval` / Playwright) | Only way to see rendered output |

The key insight: never make the browser do the heavy lifting. Use programmatic calls for setup, check the running app with Next.js devtools MCP, then hand off to browser automation only for the lightweight "enter a resume key, click Resume, screenshot" loop.

## Prerequisites

You need two long-running backend services plus the existing web frontend on port `3000`.

### 1. Reuse the existing Next.js dev server

Do not start another Next.js dev server from the same worktree. This repo normally already has one running on port `3000`. If you think it needs a restart, ask first.

Before doing browser work, use the Next.js devtools MCP against port `3000`:

- `nextjs_index(port=3000)` to discover the server
- `get_errors` to confirm there are no current runtime/build errors
- `get_routes` to confirm the route surface you expect

### 2. Start backend services

Start these processes in separate terminals (or background them). **Order matters** — Prefect must be up before the pipeline deployment registers.

| # | Process | Port | Start command | What it does |
|---|---------|------|---------------|--------------|
| 1 | Prefect server | 4200 | See below | Central API coordinator |
| 2 | Pipeline deployment | — | `cd apps/data-pipeline && uv run python -m causal_ssm_agent.flows.pipeline` | Calls `.serve()` to register the `causal-inference` deployment and poll for triggered runs |
| 3 | Next.js frontend | 3000 | Reuse the existing dev server | Web UI for session resume and stage visualization |

#### Prefect server (file-backed SQLite)

By default, Prefect writes its SQLite database to `~/.prefect/`. For integration
testing, use a dedicated file-backed SQLite database. This avoids the transaction
failures we hit during child-flow state transitions.

```bash
rm -f /tmp/causal-ssm-agent-prefect.db /tmp/causal-ssm-agent-prefect.db-shm /tmp/causal-ssm-agent-prefect.db-wal
cd apps/data-pipeline && PREFECT_SERVER_DATABASE_CONNECTION_URL="sqlite+aiosqlite:////tmp/causal-ssm-agent-prefect.db" uv run prefect server start
```

Delete the database files before every restart so each integration run starts from
a clean Prefect state.

### 3. Health-check the services

```bash
# Prefect server
curl -sf http://localhost:4200/api/health && echo "prefect ok"

# Pipeline deployment registered
curl -s -X POST http://localhost:4200/api/deployments/filter \
  -H 'Content-Type: application/json' \
  -d '{"deployments":{"name":{"any_":["causal-inference"]}}}' \
  | jq -r '.[0].id' && echo "deployment ok"

# Next.js frontend
curl -sf -o /dev/null http://localhost:3000 && echo "next.js ok"
```

All three must succeed before proceeding. For agentic runs, also confirm `get_errors` reports no current Next.js errors before moving to browser automation.

## Workspace Layout

```text
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Raw uploaded files for stage 0
│   ├── query.txt          # Materialized research question
│   ├── session.json       # Per-workspace run lineage metadata
│   └── run/               # Persisted stage JSON + artifacts
├── DEFAULT/               # Tracked mock fixture workspace
├── DOCTOLIB/              # Tracked mock fixture workspace
├── GOLDEN/                # Default tracked workspace for evals and manual sampling
├── MEDICAL_SEMANTICS/     # Tracked medical archive fixture for stage 0-2 golden tests
└── SMALLGOLDEN/           # Smaller tracked workspace for quicker eval iteration
```

## Step-by-Step Flow

### 1. Place data

Copy an input file into the workspace:

```bash
WORKSPACE_ID="T3ST42"
ACCESS_CODE="test"
mkdir -p data/$WORKSPACE_ID/input
cp data/GOLDEN/input/MyActivity.json data/$WORKSPACE_ID/input/
```

Stage 0 scans `data/{workspace_id}/input/` for non-hidden files and uses the most
recent one in that directory. If that file is a zip archive, it is extracted
before ingestion. Otherwise it is copied unchanged into the agent's working
directory for inspection. Plain-text inputs such as JSON, CSV, TSV, TXT,
Markdown, and log files work directly. Other file types can also be staged
here, but they only succeed if the ingestion agent can parse them.

### 2. Trigger pipeline via Prefect API

```bash
# Get deployment ID
DEPLOY_ID=$(curl -s -X POST http://localhost:4200/api/deployments/filter \
  -H 'Content-Type: application/json' \
  -d '{"deployments":{"name":{"any_":["causal-inference"]}}}' \
  | jq -r '.[0].id')

# Create flow run
FLOW_RUN_ID=$(curl -s -X POST "http://localhost:4200/api/deployments/$DEPLOY_ID/create_flow_run" \
  -H 'Content-Type: application/json' \
  -d "{\"tags\":[\"workspace:$WORKSPACE_ID\"],\"parameters\":{\"query\":\"How does screen time affect sleep?\",\"workspace_id\":\"$WORKSPACE_ID\",\"override_gates\":true}}" \
  | jq -r '.id')

echo "Flow Run ID: $FLOW_RUN_ID"
```

### 3. Verify workspace lookup

```bash
COOKIE_JAR=$(mktemp)

curl -s -c "$COOKIE_JAR" -X POST http://localhost:3000/api/workspaces/unlock \
  -H 'Content-Type: application/json' \
  -d "{\"workspaceId\":\"$WORKSPACE_ID\",\"accessCode\":\"$ACCESS_CODE\"}"

curl -s -b "$COOKIE_JAR" http://localhost:3000/api/analysis/$WORKSPACE_ID
# → {"workspaceId":"...","question":"...","rootFlowRunIds":["..."],"latestRootFlowRunId":"...","stages":{...}}
```

### 4. Resume via browser automation

Using browser automation:

```text
1. Navigate to http://localhost:3000
2. Type the resume key (`{WORKSPACE_ID}.{ACCESS_CODE}`) into the resume input
3. Click "Resume" button
4. Verify redirect to /analysis/{WORKSPACE_ID}
5. Screenshot the progress bar (should show the workspace ID badge)
```

### 5. Screenshot stages as they complete

Poll and screenshot as the pipeline progresses:

```text
1. Wait for stage-0 section to appear → screenshot
2. Wait for stage-1a section → screenshot
3. ... repeat through stage-6
4. Final screenshot when "Complete" badge appears
```

The screenshots serve as visual regression artifacts. If the UI behaves unexpectedly, check Next.js devtools MCP errors before assuming the browser script is wrong.

## Resuming After a Stage Failure

Resume behavior follows [execution-semantics.md#resume-semantics](../reference/execution-semantics.md#resume-semantics): earlier stages restore from persisted artifacts, then only the requested rerun window executes again.

### Identify the failed stage

Check the Prefect flow run to find which stage failed:

```bash
# Get the failed flow run's state
curl -s "http://localhost:4200/api/flow_runs/$FLOW_RUN_ID" | jq '{state: .state.type, name: .state.name}'

# List which stage artifacts already exist on disk
ls data/$WORKSPACE_ID/run/
```

The last successfully written `stage-*-state.pkl` tells you where execution stopped.
If `stage-2-state.pkl` exists but `stage-3-state.pkl` does not, `stage-3` failed.

### Rerun from the failed stage

Use the `start_stage` parameter to skip all earlier stages. You do **not** need to re-supply the `query` parameter; it was materialized to `data/{workspace_id}/query.txt` during the original run.

```bash
# Example: stage-3 failed, rerun from stage-3 onward
FLOW_RUN_ID=$(curl -s -X POST "http://localhost:4200/api/deployments/$DEPLOY_ID/create_flow_run" \
  -H 'Content-Type: application/json' \
  -d "{\"tags\":[\"workspace:$WORKSPACE_ID\"],\"parameters\":{\"workspace_id\":\"$WORKSPACE_ID\",\"override_gates\":true,\"start_stage\":\"stage-3\"}}" \
  | jq -r '.id')
```

You can also scope the rerun to a single stage by combining `start_stage` and
`end_stage`:

```bash
# Rerun only stage-4, then stop
FLOW_RUN_ID=$(curl -s -X POST "http://localhost:4200/api/deployments/$DEPLOY_ID/create_flow_run" \
  -H 'Content-Type: application/json' \
  -d "{\"tags\":[\"workspace:$WORKSPACE_ID\"],\"parameters\":{\"workspace_id\":\"$WORKSPACE_ID\",\"override_gates\":true,\"start_stage\":\"stage-4\",\"end_stage\":\"stage-4\"}}" \
  | jq -r '.id')
```

No extra web-side registration step is required after reruns. The analysis UI
discovers workspace history directly from Prefect root runs tagged with
`workspace:{workspace_id}`.

### Valid stage IDs

The full stage sequence is:

```text
stage-0 → stage-1a → stage-1b → stage-2 → stage-3 → stage-4 → stage-4b → stage-5a → stage-5b → stage-6
```
