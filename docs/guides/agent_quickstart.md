# Agent Quickstart

The service is headless: an external LLM agent (Claude Code, claude.ai, or
anything MCP-speaking) navigates the episode machine, and the web viewer
renders what the journal records. This guide is the agent-side setup.

## Start the service

```bash
bun run integration:start
```

This brings up the Temporal dev server (port `7233`), the episode worker, the
tool server with the episode facade (port `8100`), and the web viewer (port
`3000`). See [agentic_integration_testing.md](agentic_integration_testing.md)
for details.

LLM-backed stages read the ambient `OPENROUTER_API_KEY` from the service's
environment — credentials are infra config, never per-move parameters.

## Register the MCP gateway

```bash
claude mcp add nof1 -- uv run --project apps/data-pipeline nof1-mcp
```

The gateway is a thin stdio adapter over the HTTP facade
(`TOOL_SERVER_URL`, default `http://localhost:8100`) — the agent drives the
exact surface the viewer and curl users see.

### Tools

- `describe_machine` — the artifact graph: what each stage consumes/produces,
  plus its execution class (`deterministic` / `batch_llm` / `judgment`).
- `get_episode` / `get_timeline` / `get_events` — state + freshness report +
  legal moves; the transition journal (including rejections and typed stage
  errors); intra-stage telemetry.
- `read_artifact` — an artifact version's meta (provenance, `derived_from`
  pins) and JSON payloads.
- `start_episode`, `run_stage`, `write_artifact`, `start_auto_run` — the
  moves. Writes are schema-validated against the artifact contracts and
  journaled with `llm` provenance.
- `list_stage_tools` / `invoke_stage_tool` — per-stage validation and query
  tools (e.g. stage-6 `simulate`), executed against pinned store versions
  with stale-provenance hard-flags.

### Navigation loop

1. `describe_machine` once, then `get_episode(workspace_id)`.
2. Propose moves: `run_stage` for compute, `write_artifact` for judgment
   (constructs, causal spec, priors, narrative — you can author these
   directly instead of running the in-service stage).
3. Long stages (stage-4, stage-5b) can outlive client tool timeouts: prefer
   `start_auto_run` + `get_episode` polling.
4. Read what happened from `get_timeline` — a `raised` transition carries the
   typed error; state is unchanged, and re-running is just proposing again.

Raw data enters by placing files under `data/{workspace_id}/input/` before
running stage-0.

## curl instead of MCP

Every tool maps 1:1 onto the facade; see the
[integration testing guide](agentic_integration_testing.md) for the curl
forms of the same flows.

## Publishing a workspace

The hosted viewer is read-only and serves exactly what was deliberately
published to the hosted store:

```bash
uv run --project apps/data-pipeline nof1-publish WORKSPACE_ID \
  --exclude raw_data --exclude input
```

Publishing is an idempotent file copy (the store is append-only), so
re-running it during a live episode gives the hosted viewer a live tail.
Raw N-of-1 data is personal data — exclude it unless the workspace is
synthetic.
