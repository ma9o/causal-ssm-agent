# Local Development Setup

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | 23 (`.node-version`) | `node --version` |
| Bun | 1.3.14 (`package.json` `packageManager`) | `bun --version` |
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
# Optional: EXA_API_KEY, TOOL_SERVER_URL

# 4. Code generation
bun run codegen
bun run docs:codegen
```

Edit `.env` and fill in at minimum:

- `OPENROUTER_API_KEY` — the ambient credential for LLM-backed pipeline stages (the only key mechanism; there is no per-user handoff)

Optional keys:

- `EXA_API_KEY` — literature search
- `TOOL_SERVER_URL` — override the episode facade / tool server URL (default `http://localhost:8100`)
- `TEMPORAL_ADDRESS` — override the Temporal dev server address (default `localhost:7233`)
- `EPISODE_FACADE_READ_ONLY=1` — serve reads only (what the hosted viewer's facade sets)

### 4. Generate API Artifacts and Docs

Generates committed API artifacts and documentation from Python sources. See the [code generation guide](codegen.md) for details.

### 5. Start development servers

```bash
# From repo root — starts all apps via Turbo:
bun run dev
```

Or individually:

| App | Command | Port |
|-----|---------|------|
| Web viewer | `cd apps/web && bun run dev` | 3000 |
| Temporal dev server | `cd apps/data-pipeline && uv run python scripts/temporal_dev_server.py` | 7233 |
| Episode worker | `cd apps/data-pipeline && uv run python -m nof1_causal_lab.machine.temporal.worker` | — |
| Tool server / episode facade | `cd apps/data-pipeline && bun run dev` | 8100 |

The web viewer works standalone with mock data. Live episodes also need the Temporal dev server, the episode worker, and the tool server — `bun run integration:start` brings up the whole stack (see the [agent quickstart](agent_quickstart.md) and the [integration testing guide](agentic_integration_testing.md)).

## Common Commands

All commands run from the repo root via Turbo across all packages:

```bash
bun run lint          # Lint (ruff, eslint, biome)
bun run lint:fix      # Auto-fix lint issues
bun run format        # Format (ruff, biome)
bun run format:check  # Check formatting without writing
bun run test          # Tests (pytest -m 'not slow', vitest)
bun run type-check    # TypeScript type-check
bun run codegen:check # Generated API artifact drift
bun run docs:check    # Generated documentation drift and markdown
```
