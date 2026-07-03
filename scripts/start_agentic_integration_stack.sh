#!/usr/bin/env bash

set -euo pipefail
set -m  # job control: each background job gets its own process group

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
STACK_DIR="$REPO_ROOT/.local/agentic-integration-stack"
LOG_DIR="$STACK_DIR/logs"

TEMPORAL_PORT=7233
TOOL_PORT=8100
WEB_PORT=3000

START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-90}"
POLL_INTERVAL_SECONDS=1

usage() {
  cat <<'EOF'
Usage: bash scripts/start_agentic_integration_stack.sh

Starts the local integration-test stack described in docs/guides/agentic_integration_testing.md:
  1. Temporal dev server on port 7233 (ephemeral state)
  2. Episode worker (Temporal task queue nof1-episodes)
  3. Tool server on port 8100 (tools + episode facade)
  4. Next.js frontend on port 3000

Logs are written under .local/agentic-integration-stack/logs/.
The script stays in the foreground and cleans up all child processes on exit.
EOF
}

log() {
  printf '[integration-stack] %s\n' "$*"
}

die() {
  printf '[integration-stack] ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

is_listening() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

require_port_free() {
  local port="$1"
  local label="$2"
  if is_listening "$port"; then
    die "$label is already listening on port $port. Stop the existing process before starting a fresh integration stack."
  fi
}

require_process_absent() {
  local pattern="$1"
  local label="$2"
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    die "$label is already running. Stop the existing process before starting a fresh integration stack."
  fi
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local elapsed=0

  until curl -sf "$url" >/dev/null 2>&1; do
    if (( elapsed >= START_TIMEOUT_SECONDS )); then
      die "Timed out waiting for $label at $url"
    fi
    sleep "$POLL_INTERVAL_SECONDS"
    ((elapsed += POLL_INTERVAL_SECONDS))
  done
}

wait_for_port() {
  local port="$1"
  local label="$2"
  local elapsed=0

  until is_listening "$port"; do
    if (( elapsed >= START_TIMEOUT_SECONDS )); then
      die "Timed out waiting for $label on port $port"
    fi
    sleep "$POLL_INTERVAL_SECONDS"
    ((elapsed += POLL_INTERVAL_SECONDS))
  done
}

CHILD_PIDS=()

cleanup() {
  trap - EXIT INT TERM
  for pid in "${CHILD_PIDS[@]}"; do
    kill -- -"$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_process() {
  local name="$1"
  local workdir="$2"
  local logfile="$3"
  local command="$4"
  local launch_script

  mkdir -p "$LOG_DIR"
  log "Starting $name"
  printf -v launch_script \
    'set -euo pipefail
cd %q
set -a
source %q
set +a
%s
' \
    "$workdir" \
    "$ENV_FILE" \
    "$command"
  bash -lc "$launch_script" >"$logfile" 2>&1 </dev/null &
  CHILD_PIDS+=($!)
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

[[ $# -eq 0 ]] || die "Unexpected arguments. Run with --help for usage."

need_cmd bash
need_cmd bun
need_cmd curl
need_cmd jq
need_cmd lsof
need_cmd pgrep
need_cmd uv

[[ -f "$ENV_FILE" ]] || die "Missing $ENV_FILE"

mkdir -p "$LOG_DIR"

temporal_log="$LOG_DIR/temporal.log"
worker_log="$LOG_DIR/worker.log"
tool_log="$LOG_DIR/tool-server.log"
web_log="$LOG_DIR/web.log"

require_port_free "$TEMPORAL_PORT" "Temporal dev server"
require_port_free "$TOOL_PORT" "Tool server"
require_port_free "$WEB_PORT" "Next.js dev server"
require_process_absent "nof1_causal_lab\\.machine\\.temporal\\.worker" "Episode worker process"

start_process \
  "temporal" \
  "$REPO_ROOT/apps/data-pipeline" \
  "$temporal_log" \
  "uv run python scripts/temporal_dev_server.py --port $TEMPORAL_PORT"
wait_for_port "$TEMPORAL_PORT" "Temporal dev server"

start_process \
  "worker" \
  "$REPO_ROOT/apps/data-pipeline" \
  "$worker_log" \
  "TEMPORAL_ADDRESS='localhost:$TEMPORAL_PORT' uv run python -m nof1_causal_lab.machine.temporal.worker"

start_process \
  "tool-server" \
  "$REPO_ROOT/apps/data-pipeline" \
  "$tool_log" \
  "TEMPORAL_ADDRESS='localhost:$TEMPORAL_PORT' uv run uvicorn nof1_causal_lab.tool_server:app --port $TOOL_PORT"
wait_for_http "http://localhost:${TOOL_PORT}/api/tools/docs" "tool server"

start_process \
  "web" \
  "$REPO_ROOT/apps/web" \
  "$web_log" \
  "bun run dev --port $WEB_PORT"
wait_for_http "http://localhost:${WEB_PORT}" "Next.js frontend"

capabilities="$(curl -sf "http://localhost:${WEB_PORT}/api/capabilities")"
moves_enabled="$(printf '%s' "$capabilities" | jq -r '.moves_enabled')"

if [[ "$moves_enabled" != "true" ]]; then
  die "/api/capabilities returned moves_enabled=$moves_enabled (is the facade read-only?)"
fi

cat <<EOF
[integration-stack] Stack ready (kill this process to tear down)
[integration-stack]   Temporal: localhost:${TEMPORAL_PORT} (ephemeral dev server)
[integration-stack]   Episode worker: task queue nof1-episodes
[integration-stack]   Tool server + episode facade: http://localhost:${TOOL_PORT}
[integration-stack]   Web app: http://localhost:${WEB_PORT}
[integration-stack]   Capabilities: moves_enabled=$moves_enabled
[integration-stack]   Logs: $LOG_DIR
EOF

wait || true
