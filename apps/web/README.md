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

Browser-side Prefect events can be overridden with `NEXT_PUBLIC_PREFECT_EVENTS_URL`.
If `OPENROUTER_API_KEY` is not present in the web runtime env, the app falls back to the monorepo root `.env`.

See the root README for full project context and [`docs/guides/`](../../docs/guides/) for usage guides.
