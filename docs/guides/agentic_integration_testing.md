# Agentic Integration Testing

How to run full end-to-end integration tests of the pipeline and web UI using an AI agent (Claude Code) with the `browser_eval` tool.

## Design Principles

The methodology splits responsibilities between two modes:

| Concern | Mode | Why |
|---------|------|-----|
| File placement, pipeline trigger, run registration | **Programmatic** (`cp`, `curl`) | Reliable, fast, no UI fragility |
| UI rendering verification, visual regression | **browser_eval** (Playwright) | Only way to see rendered output |

The key insight: never make the browser do the heavy lifting. Use programmatic calls for setup, then hand off to `browser_eval` only for the lightweight "enter a user ID, click Resume, screenshot" loop.

## Prerequisites

You need three long-running services for integration testing.

### 1. Check for Next.js dev lock

The Next.js dev server acquires a lock at `apps/web/.next/dev/lock`. You cannot run two instances from the same `apps/web/` directory. Before starting the test server, check:

```bash
ls apps/web/.next/dev/lock 2>/dev/null && echo "LOCKED" || echo "OK"
```

If **LOCKED**: stop and ask the user to clear the lock or stop the existing process manually. Do NOT kill the process yourself.

### 2. Start services

Start these processes in separate terminals (or background them). **Order matters** — Prefect must be up before the pipeline deployment registers.

| # | Process | Port | Start command | What it does |
|---|---------|------|---------------|--------------|
| 1 | Prefect server | 4200 | See below | Central API coordinator |
| 2 | Pipeline deployment | — | `cd apps/data-pipeline && uv run python -m causal_ssm_agent.flows.pipeline` | Calls `.serve()` to register the `causal-inference` deployment and poll for triggered runs |
| 3 | Next.js frontend | 3000 | `cd apps/web && bun run dev -p 3000` | Web UI for session resume and stage visualization |

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

### 3. Health-check the HTTP services

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

All three must succeed before proceeding.

## Step-by-Step Flow

### 1. Place data

Copy an input file into the user workspace:

```bash
USER_ID="T3ST42"
mkdir -p data/$USER_ID/input
cp data/GOLDEN/input/MyActivity.json data/$USER_ID/input/
```

Stage 0 scans `data/{user_id}/input/` for non-hidden files and uses the most
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
  -d "{\"parameters\":{\"query\":\"How does screen time affect sleep?\",\"user_id\":\"$USER_ID\",\"override_gates\":true}}" \
  | jq -r '.id')

echo "Flow Run ID: $FLOW_RUN_ID"
```

### 3. Register run metadata

```bash
curl -s -X POST http://localhost:3000/api/sessions \
  -H 'Content-Type: application/json' \
  -d "{\"userId\":\"$USER_ID\",\"rootFlowRunId\":\"$FLOW_RUN_ID\",\"question\":\"How does screen time affect sleep?\"}"
# → {"ok":true}
```

### 4. Verify user lookup

```bash
curl -s http://localhost:3000/api/sessions/$USER_ID
# → {"rootFlowRunIds":["..."],"question":"...","createdAt":"..."}
```

### 5. Resume via browser_eval

Using the `browser_eval` tool:

```
1. Navigate to http://localhost:3000
2. Type the user ID into the resume input
3. Click "Resume" button
4. Verify redirect to /analysis/{USER_ID}
   If the session write failed but the run launched successfully, the URL may include ?rootFlowRunId=...
5. Screenshot the progress bar (should show the user ID badge)
```

### 6. Screenshot stages as they complete

Poll and screenshot as the pipeline progresses:

```
1. Wait for stage-0 section to appear → screenshot
2. Wait for stage-1a section → screenshot
3. ... repeat through stage-5
4. Final screenshot when "Complete" badge appears
```

The screenshots serve as visual regression artifacts — an agent can compare them against expected layouts.

## Resuming After a Stage Failure

**Do not restart the pipeline from scratch.** Every stage persists its output to
`data/{user_id}/run/` (both `{stage_id}-state.pkl` snapshots and `{stage_id}.json`
web payloads). When a stage fails, the earlier stages' artifacts are already on disk
and can be reused.

### Identify the failed stage

Check the Prefect flow run to find which stage failed:

```bash
# Get the failed flow run's state
curl -s "http://localhost:4200/api/flow_runs/$FLOW_RUN_ID" | jq '{state: .state.type, name: .state.name}'

# List which stage artifacts already exist on disk
ls data/$USER_ID/run/
```

The last successfully written `stage-*-state.pkl` tells you where execution stopped.
If `stage-2-state.pkl` exists but `stage-3-state.pkl` does not, stage 3 failed.

### Rerun from the failed stage

Use the `start_stage` parameter to skip all earlier stages — they are restored from
their on-disk snapshots automatically. You do **not** need to re-supply the `query`
parameter; it was materialized to `data/{user_id}/query.txt` during the original run.

```bash
# Example: stage-3 failed, rerun from stage-3 onward
FLOW_RUN_ID=$(curl -s -X POST "http://localhost:4200/api/deployments/$DEPLOY_ID/create_flow_run" \
  -H 'Content-Type: application/json' \
  -d "{\"parameters\":{\"user_id\":\"$USER_ID\",\"override_gates\":true,\"start_stage\":\"stage-3\"}}" \
  | jq -r '.id')
```

You can also scope the rerun to a single stage by combining `start_stage` and
`end_stage`:

```bash
# Rerun only stage-4, then stop
FLOW_RUN_ID=$(curl -s -X POST "http://localhost:4200/api/deployments/$DEPLOY_ID/create_flow_run" \
  -H 'Content-Type: application/json' \
  -d "{\"parameters\":{\"user_id\":\"$USER_ID\",\"override_gates\":true,\"start_stage\":\"stage-4\",\"end_stage\":\"stage-4\"}}" \
  | jq -r '.id')
```

### Re-register the new flow run

After triggering a resume run, update the session so the web UI tracks the new
flow run ID:

```bash
curl -s -X POST http://localhost:3000/api/sessions \
  -H 'Content-Type: application/json' \
  -d "{\"userId\":\"$USER_ID\",\"rootFlowRunId\":\"$FLOW_RUN_ID\",\"question\":\"How does screen time affect sleep?\"}"
```

### Valid stage IDs

The full stage sequence is:

```
stage-0 → stage-1a → stage-1b → stage-2 → stage-3 → stage-4 → stage-4b → stage-5a → stage-5b → stage-6
```

## Why User IDs Enable This

The user ID is the linchpin:

1. **Names the workspace** — `data/{user_id}/input/`, `data/{user_id}/query.txt`, and `data/{user_id}/run/`
2. **Links to the active Prefect run** — `sessions.json` maps `user_id → flowRunId`
3. **Serves as a resume token** — type it into the landing page to recover `/analysis/{user_id}`
4. **Is fully stateless on the client** — no localStorage, no cookies, no sessionStorage

Anonymous users still get a short generated user ID, while authenticated users reuse their durable OpenRouter user ID. Both resume through the same path.

## What browser_eval Provides

The browser automation integration supports:

- **Navigation** — `goto(url)`
- **Screenshots** — viewport capture, returned as base64
- **Click / Type / Fill** — form interaction
- **File uploads** — `setInputFiles()` on file inputs
- **JS execution** — run arbitrary scripts in page context
- **Console messages** — capture `console.log` output

This means an agent can verify not just that the API returns correct JSON, but that charts render, DAGs display correctly, and the UI transitions through stages as expected.
