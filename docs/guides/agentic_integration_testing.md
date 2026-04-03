# Agentic Integration Testing

## Design Principles

| Concern | Mode | Why |
|---------|------|-----|
| File placement, pipeline trigger, run registration | **Programmatic** (`curl`) | Reliable, fast, no UI fragility |
| Runtime errors, route discovery, build state | **Next.js devtools MCP** | Reads the running app directly and catches server/runtime errors before UI debugging |
| UI rendering verification, visual regression | **browser automation** (`browser_eval` / Playwright) | Only way to see rendered output |

## Prerequisites

All backend processes that might touch BYOK replay/refinement paths must source the
same root `.env` as the web app so `APP_SECRET` is available everywhere, not just
in Next.js.

```bash
bun run integration:start
```

Starts Prefect on port `4200`, the pipeline deployment poller, the tool
server on port `8100`, and the web app on port `3000`. The script owns
all child processes via process groups and **stays in the foreground** —
wait for the `Stack ready` banner before proceeding.

To tear down the stack, kill the script (`Ctrl+C`, or `kill <pid>` if
backgrounded). All child processes are cleaned up automatically.

## Workspace Layout

```text
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Raw uploaded files for stage 0
│   ├── query.txt          # Materialized research question
│   └── run/               # Persisted stage JSON + artifacts
├── DEFAULT/               # Tracked mock fixture workspace
├── DOCTOLIB/              # Tracked mock fixture workspace
├── GOLDEN/                # Default tracked workspace for evals and manual sampling
├── MEDICAL_SEMANTICS/     # Tracked medical archive fixture for stage 0-2 golden tests
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

FLOW_RUN_ID=$(curl -s -b "$COOKIE_JAR" -X POST http://localhost:3000/api/runs \
  -H 'Content-Type: application/json' \
  -d "{\"workspaceId\":\"$WORKSPACE_ID\",\"launchId\":\"$LAUNCH_ID\",\"query\":\"$QUESTION\"}" \
  | jq -r '.rootFlowRunId')

curl -s -b "$COOKIE_JAR" http://localhost:3000/api/analysis/$WORKSPACE_ID
# → {"workspaceId":"...","question":"...","rootFlowRunIds":["..."],"latestRootFlowRunId":"...","stages":{...}}
```

### 2. Verify via browser automation

Navigate to `http://localhost:3000/analysis/{WORKSPACE_ID}`, screenshot the progress bar, then poll and screenshot each stage section as it completes through the final "Complete" badge.

If the UI behaves unexpectedly, check Next.js devtools MCP errors before debugging the browser script.

## Resuming After a Stage Failure

### Identify the failed stage

```bash
# Get the failed flow run's state
curl -s "http://localhost:4200/api/flow_runs/$FLOW_RUN_ID" | jq '{state: .state.type, name: .state.name}'

# List which stage artifacts already exist on disk
ls data/$WORKSPACE_ID/run/
```

The last successfully written `stage-*-state.pkl` tells you where execution stopped.

### Rerun from the failed stage

Use the `start_stage` parameter to skip all earlier stages. You do **not** need to re-supply the `query` parameter; it was materialized to `data/{workspace_id}/query.txt` during the original run.

```bash
DEPLOY_ID=$(curl -s -X POST http://localhost:4200/api/deployments/filter \
  -H 'Content-Type: application/json' \
  -d '{"deployments":{"name":{"any_":["causal-inference"]}}}' \
  | jq -r '.[0].id')

# Example: stage-3 failed, rerun from stage-3 onward
FLOW_RUN_ID=$(curl -s -X POST "http://localhost:4200/api/deployments/$DEPLOY_ID/create_flow_run" \
  -H 'Content-Type: application/json' \
  -d "{\"tags\":[\"workspace:$WORKSPACE_ID\"],\"parameters\":{\"workspace_id\":\"$WORKSPACE_ID\",\"start_stage\":\"stage-3\"}}" \
  | jq -r '.id')
```

You can also scope the rerun to a single stage by combining `start_stage` and
`end_stage`:

```bash
# Rerun only stage-4, then stop
FLOW_RUN_ID=$(curl -s -X POST "http://localhost:4200/api/deployments/$DEPLOY_ID/create_flow_run" \
  -H 'Content-Type: application/json' \
  -d "{\"tags\":[\"workspace:$WORKSPACE_ID\"],\"parameters\":{\"workspace_id\":\"$WORKSPACE_ID\",\"start_stage\":\"stage-4\",\"end_stage\":\"stage-4\"}}" \
  | jq -r '.id')
```

No extra web-side registration step is required after reruns. The analysis UI
discovers workspace history directly from Prefect root runs tagged with
`workspace:{workspace_id}`.

## Apply a stage refinement via API

Use [`/api/refine/apply`](../../apps/web/src/app/api/refine/apply/route.ts) when you have a refinement patch and optional chat messages:

- for non-terminal stages, merge the patch into the current stage payload and trigger [`/api/replay`](../../apps/web/src/app/api/replay/route.ts)
- for terminal `stage-6`, persist the patch in place through the tool server

```bash
curl -s -b "$COOKIE_JAR" -X POST http://localhost:3000/api/refine/apply \
  -H 'Content-Type: application/json' \
  -d '{
    "workspaceId": "'"$WORKSPACE_ID"'",
    "stageId": "stage-1b",
    "rootFlowRunId": "'"$FLOW_RUN_ID"'",
    "stagePatch": {
      "causal_spec": {
        "measurement": {
          "model_clock": "1d",
          "indicators": []
        }
      }
    },
    "messages": []
  }'
```

The response matches the UI contract:

```json
{
  "ok": true,
  "updatedFields": ["causal_spec"],
  "resumeFrom": "stage-2",
  "rootFlowRunId": "..."
}
```

Use [`/api/replay`](../../apps/web/src/app/api/replay/route.ts) directly when you already have the fully materialized stage payload and do not need the server to merge refinement messages into `llm_trace` first:

```bash
curl -s -b "$COOKIE_JAR" -X POST http://localhost:3000/api/replay \
  -H 'Content-Type: application/json' \
  -d '{
    "workspaceId": "'"$WORKSPACE_ID"'",
    "stageId": "stage-1b",
    "rootFlowRunId": "'"$FLOW_RUN_ID"'",
    "stageData": {
      "causal_spec": {
        "measurement": {
          "model_clock": "1d",
          "indicators": []
        }
      }
    }
  }'
```

This route launches a new root flow run with `start_stage` set to the edited stage and a `stage_overrides` entry for that stage, so only downstream stages rerun.

### Valid stage IDs

The full stage sequence is:

```text
stage-0 → stage-1a → stage-1b → stage-2 → stage-3 → stage-4 → stage-4b → stage-5a → stage-5b → stage-6
```
