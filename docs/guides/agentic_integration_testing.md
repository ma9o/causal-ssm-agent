# Agentic Integration Testing

How to run full end-to-end integration tests of the pipeline + web UI using an AI agent (Claude Code) with `next-devtools-mcp`'s `browser_eval` tool.

## Design Principles

The methodology splits responsibilities between two modes:

| Concern | Mode | Why |
|---------|------|-----|
| File placement, pipeline trigger, session registration | **Programmatic** (`cp`, `curl`) | Reliable, fast, no UI fragility |
| UI rendering verification, visual regression | **browser_eval** (Playwright) | Only way to see rendered output |

The key insight: never make the browser do the heavy lifting. Use programmatic calls for setup, then hand off to `browser_eval` only for the lightweight "type 6 characters, click Resume, screenshot" loop.

## Prerequisites

You need three **dedicated** services for integration testing. Do NOT reuse the existing dev server on port 3000 — these are isolated test instances.

### 1. Check for Next.js dev lock

The Next.js dev server acquires a lock at `apps/web/.next/dev/lock`. You cannot run two instances from the same `apps/web/` directory. Before starting the test server, check:

```bash
ls apps/web/.next/dev/lock 2>/dev/null && echo "LOCKED" || echo "OK"
```

If **LOCKED**: the user already has a dev server running from this worktree. Ask them to switch that terminal to this branch and restart on port 3001, or stop it manually. Do NOT kill the process yourself.

### 2. Start services (in order)

Start these three services in separate terminals (or background them). **Order matters** — Prefect must be up before the pipeline deployment registers.

| # | Service | Port | Start command | What it does |
|---|---------|------|---------------|--------------|
| 1 | Prefect server | 4200 | `cd apps/data-pipeline && uv run prefect server start` | Central API coordinator |
| 2 | Pipeline deployment | — | `cd apps/data-pipeline && uv run python -m causal_ssm_agent.flows.pipeline` | Calls `.serve()` to register the `causal-inference` deployment and poll for triggered runs |
| 3 | Next.js frontend | 3001 | `cd apps/web && bun run dev -p 3001` | Web UI for session resume and stage visualization |

### 3. Health-check all three

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

The `.mcp.json` at the worktree root must configure `next-devtools-mcp` for `browser_eval` access.

## Step-by-Step Flow

### 1. Place data

Copy test data into a session-code-named directory:

```bash
CODE="T3ST42"
mkdir -p apps/data-pipeline/data/raw/$CODE
cp apps/data-pipeline/data/raw/test_user/MyActivity.json \
   apps/data-pipeline/data/raw/$CODE/
```

### 2. Trigger pipeline via Prefect API

```bash
# Get deployment ID
DEPLOY_ID=$(curl -s -X POST http://localhost:4200/api/deployments/filter \
  -H 'Content-Type: application/json' \
  -d '{"deployments":{"name":{"any_":["causal-inference"]}}}' \
  | jq -r '.[0].id')

# Create flow run
RUN_ID=$(curl -s -X POST "http://localhost:4200/api/deployments/$DEPLOY_ID/create_flow_run" \
  -H 'Content-Type: application/json' \
  -d "{\"parameters\":{\"query\":\"How does screen time affect sleep?\",\"user_id\":\"$CODE\",\"override_gates\":true}}" \
  | jq -r '.id')

echo "Run ID: $RUN_ID"
```

### 3. Register session

```bash
curl -s -X POST http://localhost:3001/api/sessions \
  -H 'Content-Type: application/json' \
  -d "{\"code\":\"$CODE\",\"runId\":\"$RUN_ID\",\"question\":\"How does screen time affect sleep?\"}"
# → {"ok":true}
```

### 4. Verify session lookup

```bash
curl -s http://localhost:3001/api/sessions/$CODE
# → {"runId":"...","question":"...","createdAt":"..."}

# Case-insensitive
curl -s http://localhost:3001/api/sessions/$(echo $CODE | tr '[:upper:]' '[:lower:]')
# → same result
```

### 5. Resume via browser_eval

Using `next-devtools-mcp`'s `browser_eval` tool:

```
1. Navigate to http://localhost:3001
2. Type session code into the resume input (monospace field, maxLength=6)
3. Click "Resume" button
4. Verify redirect to /analysis/{runId}?code={CODE}
5. Screenshot the progress bar (should show session code badge)
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

## Why Session Codes Enable This

The 6-character session code is the linchpin:

1. **Names the data directory** — `data/raw/{code}/` (replaces throwaway `user-{timestamp}`)
2. **Links to the Prefect run** — `sessions.json` maps `code → runId`
3. **Serves as a resume token** — type it into the landing page to recover the analysis URL
4. **Is fully stateless on the client** — no localStorage, no cookies, no sessionStorage

An agent holds the code in a shell variable. A human writes it on a napkin. Both can resume.

## What browser_eval Provides

The `next-devtools-mcp` Playwright integration supports:

- **Navigation** — `goto(url)`
- **Screenshots** — viewport capture, returned as base64
- **Click / Type / Fill** — form interaction
- **File uploads** — `setInputFiles()` on file inputs
- **JS execution** — run arbitrary scripts in page context
- **Console messages** — capture `console.log` output

This means an agent can verify not just that the API returns correct JSON, but that charts render, DAGs display correctly, and the UI transitions through stages as expected.
