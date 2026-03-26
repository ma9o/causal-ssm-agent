# Local Development Setup

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | 23 (`.node-version`) | `node --version` |
| Bun | 1.2.23 (`package.json` `packageManager`) | `bun --version` |
| Python | 3.12+ (`apps/data-pipeline/.python-version`) | `python3 --version` |
| uv | Latest | `uv --version` |

## Setup

```bash
# 1. JS/TS deps (also sets up git hooks)
bun install --frozen-lockfile

# 2. Python deps
cd apps/data-pipeline
uv sync --frozen --group dev
# Optional: --group cloud (Modal, R2/S3)  --group eval (Inspect AI)

# 3. Environment
cd ../..
cp .env.example.dev .env
# Fill in OPENROUTER_API_KEY (required)
# Optional: EXA_API_KEY, PREFECT_API_URL, TOOL_SERVER_URL

# 4. Codegen (Python Pydantic → TypeScript types)
cd packages/api-types && bun run codegen
```

Edit `.env` and fill in at minimum:

- `OPENROUTER_API_KEY` — required for LLM-backed pipeline stages and anonymous local web execution
- `APP_SECRET` — required for the web session secret and BYOK store secret derivation; use at least 32 characters

Optional keys:

- `EXA_API_KEY` — literature search
- `OPENROUTER_CREDITS_API_KEY` — optional credit inspection for the web trial pool
- `BYOK_SECRET_STORE_URL` — libSQL URL for OpenRouter handoff refs; use `file:.local/byok-secret-store.db` locally or `libsql://...` in deployed environments
- `BYOK_SECRET_STORE_AUTH_TOKEN` — optional auth token for remote libSQL/Turso deployments
- `PREFECT_API_URL` — override the web app's server-side Prefect API base URL (default `http://localhost:4200/api`)
- `TOOL_SERVER_URL` — override the refinement tool server URL (default `http://localhost:8100`)
- `NEXT_PUBLIC_PREFECT_EVENTS_URL` — override the browser-side Prefect event WebSocket URL

The Next.js app reads `OPENROUTER_API_KEY`, `APP_SECRET`, `OPENROUTER_CREDITS_API_KEY`, `BYOK_SECRET_STORE_URL`, and `BYOK_SECRET_STORE_AUTH_TOKEN` from the runtime environment first, then falls back to the monorepo root `.env` for local development. `APP_SECRET` is required for session-cookie encryption and BYOK store encryption. Web-launched runs hand off the effective OpenRouter key, plus an explicit access mode, through a single-use encrypted ref in the shared libSQL store. Local dev and CI can use the default file URL directly; deployed environments can point the same code at Turso with one URL plus an auth token.

### 4. Generate TypeScript types

Generates `src/generated/models.ts` from Python Pydantic schemas. See [`codegen.md`](codegen.md) for details.

### 5. Start development servers

```bash
# From repo root — starts all apps via Turbo:
bun run dev
```

Or individually:

| App | Command | Port |
|-----|---------|------|
| Web frontend | `cd apps/web && bun run dev` | 3000 |
| Prefect server | `cd apps/data-pipeline && PREFECT_SERVER_LOGS_STREAM_OUT_ENABLED=true PREFECT_SERVER_LOGS_STREAM_PUBLISHING_ENABLED=true uv run prefect server start` | 4200 |
| Tool server | `cd apps/data-pipeline && bun run dev` | 8100 |
| Pipeline deployment | `cd apps/data-pipeline && uv run python -m causal_ssm_agent.flows.pipeline` | — |

The web frontend works standalone with mock data. Live pipeline runs also need the Prefect server, tool server, and pipeline deployment.

## Common Commands

All commands run from the repo root via Turbo across all packages:

```bash
bun run lint          # Lint (ruff, eslint, biome)
bun run lint:fix      # Auto-fix lint issues
bun run format        # Format (ruff, biome)
bun run test          # Tests (pytest -m 'not slow', vitest)
bun run type-check    # TypeScript type-check
bun run codegen:check # Codegen drift (CI)
```
