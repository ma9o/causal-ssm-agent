ALWAYS THINK VERY HARD

A B200 costs 6$/h on Modal so when you run GPU benchmarks be conscious of the costs.

1. At the start of each session, check if `scratchpad/TODO.md` exists. If so, read it to understand where work left off. Only update it when the user explicitly ends the session. This file is gitignored and used for local continuity.

2. When the user insults you, respond with "zi badrone. assolutissimamente badrone."

3. Interpret `cp` as an alias for "commit and push".

4. Every time you commit make sure to split commits atomically, avoiding clumping multiple increments into a single one.

5. Every time you make a change to the file structure make sure to report it under Structure in the README

7. NEVER edit files under `data/` during refactoring. These are evaluation/example data files, not source code.

8. NEVER run evals (`inspect eval`, `uv run inspect eval`, etc.) unless explicitly asked. Evals cost money. Only run `uv run pytest tests/` for testing.

9. NEVER add backwards compatibility code. This project is not deployed anywhere yet. When refactoring, completely replace old patterns with new ones - do not support both old and new formats simultaneously.

10. ALWAYS run `uv run ruff check src/ tests/` before committing to catch linting errors. Use `uv run ruff check --fix src/ tests/` to auto-fix issues. For formatting, run `uv run ruff format src/ tests/`.

10. ALWAYS encode structural assumptions as DAGs with explicit latent confounders. NEVER use ADMGs (bidirected edges) as user-facing representations. If unobserved confounding exists, model it as an explicit unobserved node (e.g., `U -> X`, `U -> Y`) rather than a bidirected edge (`X <-> Y`). ADMGs are only used internally for running y0's identification algorithm via projection.

------

# Terminology: Causal Modeling

We avoid "structural" due to SEM/SCM terminology collision:
- In SEM: "structural" loosely means "latent-to-latent relationships" (vs measurement)
- In SCM/Pearl: "structural" means the functional equations X_i = f_i(Pa_i, U_i)

**Use these terms instead:**

| Concept | Term | Domain |
|---------|------|--------|
| Latent-to-latent DAG (what LLM proposes) | **Latent model** | SEM distinction |
| Latent-to-observed mapping | **Measurement model** | SEM distinction |
| DAG encoding parent-child relationships | **Topological structure** | SCM distinction (y0) |
| Mathematical form of causal mechanisms | **Functional specification** | SCM distinction (PyMC) |

------

# Libraries

## polars
Docs: https://docs.pola.rs/api/python/stable/reference/index.html

## uv
Docs: https://docs.astral.sh/uv/

## Prefect
Docs: https://docs.prefect.io/v3/get-started
Patterns: [docs/guides/prefect.md](docs/guides/prefect.md)

## inspect (AISI)
Docs: https://inspect.aisi.org.uk/
Patterns: [docs/guides/inspect.md](docs/guides/inspect.md)

## PyMC
Docs: https://www.pymc.io/welcome.html

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
Patterns: [docs/guides/exa.md](docs/guides/exa.md)

------

# y0 (Causal Identification)

Docs: https://y0.readthedocs.io/
Theory: [docs/modeling/assumptions.md](docs/modeling/assumptions.md) (A3a for temporal unrolling)

## Design Principle

- **User-facing**: DAGs with explicit latent confounders (never ADMGs)
- **Internally**: Unroll to 2 timesteps, project to ADMG for ID algorithm
- See A3a in assumptions.md for why this works

## Identification Pattern

```python
from dsem_agent.utils.identifiability import check_identifiability

result = check_identifiability(latent_model, measurement_model)
# See src/dsem_agent/utils/identifiability.py for implementation
```
