ALWAYS THINK VERY HARD

# Web app

- NEVER put domain logic (statistical computations, field derivations, pass/fail decisions) into the frontend code. If the frontend needs a field, compute it at the source (the pipeline domain code that owns that data). Serialization boundaries should only assemble and pass through — never transform or derive.

- A dev server is ALREADY running on port 3000. Do NOT start a new one.
- To check for errors, use the next-devtools MCP 
- If you need to restart the server, ask me first.
- We are strictily in the `bun` ecosystem, not `npm` or `pnpm` or `yarn`

# Data Pipeline

- Avoid running tests marked as `slow` unless you changed something that directly impacts them. In general always run the subset of tests that make sense for the changes.

- A B200 costs 6$/h on Modal so when you run GPU benchmarks be conscious of the costs.

- At the start of each session, check if `scratchpad/TODO.md` exists. If so, read it to understand where work left off. Only update it when the user explicitly ends the session. This file is gitignored and used for local continuity.

- Interpret `cp` as an alias for "commit and push".

- Every time you commit make sure to split commits atomically, avoiding clumping multiple increments into a single one.

- Every time you make a change to the file structure make sure to report it under Structure in the README

- NEVER edit files under `data/` during refactoring. These are evaluation/example data files, not source code.

- NEVER run evals (`inspect eval`, `uv run inspect eval`, etc.) unless explicitly asked. Evals cost money. Only run `uv run pytest tests/` for testing.

- NEVER add backwards compatibility code. This project is not deployed anywhere yet. When refactoring, completely replace old patterns with new ones - do not support both old and new formats simultaneously.

- ALWAYS run `uv run ruff check src/ tests/` before committing to catch linting errors. Use `uv run ruff check --fix src/ tests/` to auto-fix issues. For formatting, run `uv run ruff format src/ tests/`.

- ALWAYS encode structural assumptions as DAGs with explicit latent confounders. NEVER use ADMGs (bidirected edges) as user-facing representations. If unobserved confounding exists, model it as an explicit unobserved node (e.g., `U -> X`, `U -> Y`) rather than a bidirected edge (`X <-> Y`). ADMGs are only used internally for running y0's identification algorithm via projection.



## Terminology: Causal Modeling

We avoid "structural" due to SEM/SCM terminology collision:
- In SEM: "structural" loosely means "latent-to-latent relationships" (vs measurement)
- In SCM/Pearl: "structural" means the functional equations X_i = f_i(Pa_i, U_i)

**Use these terms instead:**

| Concept | Term | Domain |
|---------|------|--------|
| Latent-to-latent DAG (what LLM proposes) | **Latent model** | SEM distinction |
| Latent-to-observed mapping | **Measurement model** | SEM distinction |
| DAG encoding parent-child relationships | **Topological structure** | SCM distinction (y0) |
| Mathematical form of causal mechanisms | **Functional specification** | SCM distinction (NumPyro) |

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

------

## y0 (Causal Identification)

Docs: https://y0.readthedocs.io/
Theory: [docs/modeling/assumptions.md](docs/modeling/assumptions.md) (A3a for temporal unrolling)

## Design Principle

- **User-facing**: DAGs with explicit latent confounders (never ADMGs)
- **Internally**: Unroll to 2 timesteps, project to ADMG for ID algorithm
- See A3a in assumptions.md for why this works

## Identification Pattern

```python
from causal_ssm_agent.utils.identifiability import check_identifiability

result = check_identifiability(latent_model, measurement_model)
# See src/causal_ssm_agent/utils/identifiability.py for implementation
```
