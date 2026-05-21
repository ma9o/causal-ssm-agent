THINK VERY HARD

- Project description: nof1-causal-lab is for observational longitudinal causal questions, especially intensive longitudinal data (ILD) and idiographic / N-of-1 settings where measurements are irregular, messy, and semantically heterogeneous. The LLM proposes constructs, indicators, causal structure, and priors. It combines explicit causal-identification checks with continuous-time latent state-space (CT-SEM-style) estimation, and only produces numeric causal claims when those checks support them. When they don't, the system stops at structural reasoning.

- At the start of each session, check if `scratchpad/TODO.md` exists. If so, read it to understand where work left off. Only update it when the user explicitly ends the session. This file is gitignored and used for local continuity.

- Interpret `cp` as an alias for "commit and push". Every time you commit make sure to split commits atomically, avoiding clumping multiple increments into a single one. This operation should be write safe so that another coding agent working in the same directory is not disrupted by eg stashing or rebasing. It's ok if this prevents you from perfect atomicity.

- NEVER add backwards compatibility code or defensive engineering with fallbacks. 

- NEVER add backwards compatibility code or defensive engineering with fallbacks. 

- NEVER add backwards compatibility code or defensive engineering with fallbacks. 

- For integration testing, starting services, health-checking the stack, or triggering pipeline runs manually, always read and follow [docs/guides/agentic_integration_testing.md](docs/guides/agentic_integration_testing.md). Do not improvise steps from memory.

# Notebooks

- Use `nbformat` and `jupyter-client` to programmatically read and write notebooks using a persistent kernel for your session, so that state is preserved across edits and you can run cells in-place without losing outputs. 

# Docs

- NEVER clump together links or references. ALWAYS either juxtapose references to the sentence or clause they support or use hyperlinks on the terms themselves.
- When you edit `README.md` or files under `docs/`, run `bun run docs:check` before handoff to verify the markdown docs.

- in `docs/pipeline`, each stage doc is the authoritative definition of its outputs and artifacts. Downstream stages should link back to these definitions rather than re-describing them.

- in `docs/pipeline`, the Outputs sections should always be a table for the relevant fields. When describing the dataclasses that make up those outputs, just include the table of fields and descriptions without the extra prose. Do not describe fields and dataclasses used for internal plumbing like `outcome` or `llm_trace` or higher level constructs like `IndicatorAudit` that bundle multiple pieces together. Focus on the core artifacts that are inputs and outputs of pipeline stages, and link to the stage where they are defined for details.

# Web app

- NEVER put domain logic like statistical computations into the frontend code. 

- A dev server is likely already running on port 3000. Do not start a new one if so. If you need to restart the server, ask me first.
- To check for errors, use the next-devtools MCP 
- We are strictily in the `bun` ecosystem, not `npm` or `pnpm` or `yarn`

# Data Pipeline

- A B200 costs 6$/h on Modal so when you run GPU benchmarks be conscious of the costs.

- NEVER run evals (`inspect eval`, `uv run inspect eval`, etc.) unless explicitly asked. Evals cost money. Only run `uv run pytest tests/` for testing.

- Before committing, run `bun run --cwd apps/data-pipeline lint` — this runs ruff check, ruff format check, config validation, and vulture in one pass. For autofix during development, use `uv run ruff check --fix src/ tests/` and `uv run ruff format src/ tests/`. Avoid running tests marked as `slow` unless your change directly impacts them; in general run the subset of tests that makes sense for the change.

- ALWAYS encode structural assumptions as DAGs with explicit latent confounders. NEVER use ADMGs (bidirected edges) as user-facing representations. If unobserved confounding exists, model it as an explicit unobserved node (e.g., `U -> X`, `U -> Y`) rather than a bidirected edge (`X <-> Y`). ADMGs are only used internally for running y0's identification algorithm via projection.


## polars
Docs: https://docs.pola.rs/api/python/stable/reference/index.html

## uv
Docs: https://docs.astral.sh/uv/

## Prefect
Docs: https://docs.prefect.io/v3/get-started

## inspect (AISI)
Docs: https://inspect.aisi.org.uk/

## NumPyro
Docs: https://num.pyro.ai/en/stable/

## JAX
Docs: https://docs.jax.dev/en/latest/

## cuthbert
Docs: https://github.com/cuthbert-ai/cuthbert
- Differentiable Kalman filter via `gaussian.moments` (use `associative=False`)
- Differentiable particle filter via `smc.particle_filter`
- Both called through `cuthbert.filtering.filter()`

## NetworkX
Docs: https://networkx.org/documentation/stable/
- Use `DiGraph` for causal DAGs (directed edges)
- Create from edge list: `nx.DiGraph([(cause, effect), ...])`
- Check for cycles: `nx.is_directed_acyclic_graph(G)`
- Add node attributes: `G.add_node(name, dtype='continuous', ...)`

## ArViz
Docs: https://python.arviz.org/en/stable/index.html

## Exa
Docs: https://exa.ai/docs/sdks/python-sdk-specification


## y0 (Causal Identification)

Docs: https://y0.readthedocs.io/
Theory: [docs/modeling/assumptions.md](docs/modeling/assumptions.md) (A3a for temporal unrolling)

### Design Principle

- **User-facing**: DAGs with explicit latent confounders (never ADMGs)
- **Internally**: Unroll to 2 timesteps, project to ADMG for ID algorithm
- See A3a in assumptions.md for why this works
