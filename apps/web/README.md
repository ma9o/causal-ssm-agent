# Web Frontend

Next.js app for session management and pipeline stage visualization. Part of the [causal-ssm-agent](../../README.md) monorepo.

```bash
bun run dev    # Start dev server on port 3000
bun run build  # Production build
bun run test   # Vitest unit tests
```

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

If `OPENROUTER_API_KEY` is not present in the web runtime env, the app falls back to the monorepo root `.env`.

See the root README for full project context and [`docs/guides/`](../../docs/guides/) for usage guides.
