#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
PC_CONFIG="$REPO_ROOT/process-compose.yaml"
LOG_DIR="$REPO_ROOT/.local/agentic-integration-stack/logs"

PC_PORT=8181
TEMPORAL_PORT=7233
TOOL_PORT=8100
WEB_PORT=3000

usage() {
  cat <<'EOF'
Usage: bash scripts/start_agentic_integration_stack.sh [process-compose up flags]

Starts the local integration-test stack described in
docs/guides/agentic_integration_testing.md under process-compose supervision
(config: process-compose.yaml):
  1. Temporal dev server on port 7233 (ephemeral state)
  2. Episode worker (Temporal task queue nof1-episodes)
  3. Tool server on port 8100 (tools + episode facade)
  4. Next.js frontend on port 3000

Startup order and health gating are declared in process-compose.yaml; failed
processes restart automatically (availability.restart: on_failure).

The process-compose API listens on port 8181 for targeted operations:
  process-compose --port 8181 process restart worker
  process-compose --port 8181 process list

Logs are written under .local/agentic-integration-stack/logs/.
The script stays in the foreground; kill it (Ctrl+C) to tear down the stack.
Pass -t=false to disable the TUI (e.g. when redirecting output to a file).
EOF
}

die() {
  printf '[integration-stack] ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

is_listening() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

require_port_free() {
  local port="$1"
  local label="$2"
  if is_listening "$port"; then
    die "$label is already listening on port $port. Stop the existing process before starting a fresh integration stack."
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

need_cmd bun
need_cmd curl
need_cmd jq
need_cmd lsof
need_cmd pgrep
need_cmd uv
need_cmd process-compose

[[ -f "$ENV_FILE" ]] || die "Missing $ENV_FILE"

require_port_free "$TEMPORAL_PORT" "Temporal dev server"
require_port_free "$TOOL_PORT" "Tool server"
require_port_free "$WEB_PORT" "Next.js dev server"
require_port_free "$PC_PORT" "process-compose API"
if pgrep -f "nof1_causal_lab\.machine\.temporal\.worker" >/dev/null 2>&1; then
  die "Episode worker process is already running. Stop the existing process before starting a fresh integration stack."
fi

mkdir -p "$LOG_DIR"

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

reap_orphaned_codex() {
  # process-compose stops the worker but does not recurse into the headless
  # `codex exec --json` subprocesses it spawned for the agentic stages: those
  # reparent to launchd and keep spending tokens after the stack is down. Reap
  # them here. Scoped to the headless `codex exec --json` invocation, so a
  # developer's own interactive `codex …` sessions are never touched.
  pkill -f 'codex exec --json' 2>/dev/null || true
}
trap reap_orphaned_codex EXIT

cd "$REPO_ROOT"
# Run process-compose as a child (not exec) so the trap survives. Ctrl+C or
# `kill <this pid>` forwards to it; `process-compose down` makes it exit on its
# own. Either way we wait for it to fully drain, then the EXIT trap reaps.
process-compose up -f "$PC_CONFIG" --port "$PC_PORT" "$@" &
pc_pid=$!
trap 'kill -TERM "$pc_pid" 2>/dev/null || true' INT TERM
while kill -0 "$pc_pid" 2>/dev/null; do wait "$pc_pid" 2>/dev/null || true; done
