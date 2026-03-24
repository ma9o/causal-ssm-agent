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

See [`codegen.md`](codegen.md) for type-generation details.

## Running

`bun run dev` from the repo root starts everything via Turbo. Or individually:

| App | Command | Port |
|-----|---------|------|
| Web frontend | `cd apps/web && bun run dev` | 3000 |
| Prefect server | `cd apps/data-pipeline && uv run prefect server start` | 4200 |
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
