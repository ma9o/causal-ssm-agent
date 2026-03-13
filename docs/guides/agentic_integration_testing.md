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
| 3 | Next.js frontend | 3001 | `cd apps/web && bun run dev -p 3001` | Web UI for session resume and stage visualization |

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
curl -sf -o /dev/null http://localhost:3001 && echo "next.js ok"
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
curl -s -X POST http://localhost:3001/api/sessions \
  -H 'Content-Type: application/json' \
  -d "{\"userId\":\"$USER_ID\",\"flowRunId\":\"$FLOW_RUN_ID\",\"question\":\"How does screen time affect sleep?\"}"
# → {"ok":true}
```

### 4. Verify user lookup

```bash
curl -s http://localhost:3001/api/sessions/$USER_ID
# → {"flowRunId":"...","question":"...","createdAt":"..."}
```

### 5. Resume via browser_eval

Using the `browser_eval` tool:

```
1. Navigate to http://localhost:3001
2. Type the user ID into the resume input
3. Click "Resume" button
4. Verify redirect to /analysis/{USER_ID}
   If that user has a tracked live run, the URL may also include ?flowRunId=...
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
