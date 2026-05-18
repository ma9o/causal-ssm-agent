# Web Frontend

Next.js app for session management and pipeline stage visualization. Part of the [nof1-causal-lab](../../README.md) monorepo.

```bash
bun run dev    # Start dev server on port 3000 if one is not already running
bun run build  # Production build
bun run test   # Vitest unit tests
```

In this repo's shared dev workflow, the web dev server is often already running on port `3000`. Reuse that server when possible; do not start a second copy from the same worktree. For runtime diagnostics, prefer the Next.js devtools MCP (`get_errors`, `get_routes`, `get_project_metadata`) over ad hoc browser-only checks.

Server-side routes default to local pipeline services:

- `PREFECT_API_URL=http://localhost:4200/api`
- `TOOL_SERVER_URL=http://localhost:8100`

Browser-side Prefect websockets can be overridden with:

- `NEXT_PUBLIC_PREFECT_EVENTS_URL`
- `NEXT_PUBLIC_PREFECT_LOGS_URL`

Live Prefect log streaming also requires the server-side Prefect settings:

- `PREFECT_SERVER_LOGS_STREAM_OUT_ENABLED=true`
- `PREFECT_SERVER_LOGS_STREAM_PUBLISHING_ENABLED=true`

The web app bootstraps logs once via REST and then expects live log delivery to come from Prefect's `logs/out` WebSocket. There is no polling fallback.

OpenRouter web access uses these server-side env vars:

- `OPENROUTER_API_KEY` for anonymous local web execution and direct pipeline execution
- `APP_SECRET` to derive the encrypted OpenRouter session cookie secret and BYOK handoff secret (minimum 32 characters)
- `OPENROUTER_CREDITS_API_KEY` for optional trial credit inspection
- `BYOK_SECRET_STORE_URL` for the OpenRouter ref store; defaults to `file:.local/byok-secret-store.db` locally and can point at `libsql://...` in deployed environments
- `BYOK_SECRET_STORE_AUTH_TOKEN` for remote libSQL/Turso deployments

The web app reads those keys from the runtime environment first, then falls back to the monorepo root `.env` for local development. `APP_SECRET` is required anywhere the web app needs to manage OpenRouter sessions or mint BYOK handoff refs. In local dev and CI, the web app and pipeline share the same file-backed libSQL database for single-use OpenRouter handoff refs. In deployed environments, both services can point that same store at Turso with the same URL and token.

See the root README for full project context and [`docs/guides/`](../../docs/guides/) for usage guides.
