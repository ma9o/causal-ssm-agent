# Local Development Setup

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | 23 (see `.node-version`) | `node --version` |
| Bun | 1.2.23 (see `package.json` `packageManager`) | `bun --version` |
| Python | 3.12+ (see `apps/data-pipeline/.python-version`) | `python3 --version` |
| uv | Latest | `uv --version` |

## Steps

### 1. Install monorepo dependencies

```bash
bun install --frozen-lockfile
```

This installs all JavaScript/TypeScript packages across the monorepo (Next.js, TanStack, XYFlow, etc.) and sets up git hooks via the `prepare` script.

### 2. Install Python dependencies

```bash
cd apps/data-pipeline
uv sync --frozen --group dev
```

Optional dependency groups:
- `--group cloud` — Modal + remote storage integrations (Stage 0 sandbox, production offload, GPU compute, R2/S3)
- `--group eval` — Inspect AI (LLM evaluation framework)

### 3. Configure environment variables

```bash
# From repo root:
cp .env.example.dev .env
```

Edit `.env` and fill in at minimum:
- `OPENROUTER_API_KEY` — required for LLM-backed pipeline stages

Optional keys:
- `EXA_API_KEY` — literature search
- `PREFECT_API_URL` — override the web app's server-side Prefect API base URL (default `http://localhost:4200/api`)
- `TOOL_SERVER_URL` — override the refinement tool server URL (default `http://localhost:8100`)
- `NEXT_PUBLIC_PREFECT_EVENTS_URL` — override the browser-side Prefect event WebSocket URL

The Next.js app falls back to the monorepo root `.env` for `OPENROUTER_API_KEY`, so a single root `.env` is enough for local development.

### 4. Generate TypeScript types

```bash
cd packages/api-types
bun run codegen
```

Generates `src/generated/models.ts` from Python Pydantic schemas. See [`docs/guides/codegen.md`](codegen.md) for details.

### 5. Start development servers

```bash
# From repo root — starts all apps via Turbo:
bun run dev
```

Or individually:

| App | Command | Port |
|-----|---------|------|
| Web frontend | `cd apps/web && bun run dev` | 3000 |
| Prefect server | `cd apps/data-pipeline && uv run prefect server start` | 4200 |
| Tool server | `cd apps/data-pipeline && bun run dev` | 8100 |
| Pipeline deployment | `cd apps/data-pipeline && uv run python -m causal_ssm_agent.flows.pipeline` | — |

The web frontend at `http://localhost:3000` works standalone with mock data. Live runs also need the Prefect server, the tool server, and the pipeline deployment.

## Common Commands

```bash
# Lint & format (Python)
cd apps/data-pipeline
uv run ruff check src/ tests/         # Lint
uv run ruff check --fix src/ tests/   # Auto-fix
uv run ruff format src/ tests/        # Format

# Tests (Python — excludes slow tests by default)
uv run pytest tests/

# Tests (TypeScript)
cd apps/web && bun run test

# Type-check (TypeScript)
bun run type-check

# Codegen drift check (CI runs this)
cd packages/api-types && bun run codegen:check
```

## Project Layout

See the root [README.md](../../README.md) for the full monorepo structure and [`docs/index.md`](../index.md) for the documentation map.
