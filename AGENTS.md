- Project description: nof1-causal-lab is for observational longitudinal causal questions, especially intensive longitudinal data (ILD) and idiographic / N-of-1 settings where measurements are irregular, messy, and semantically heterogeneous. The LLM proposes constructs, indicators, causal structure, and priors. It combines explicit causal-identification checks with continuous-time latent state-space estimation, and only produces numeric causal claims when those checks support them.

- When the user references TODOs, it means the top level `scratchpad/TODO.md`. This file is gitignored and used for local continuity.

- Interpret `cp` as an alias for "commit and push". Every time you commit make sure to split commits atomically, avoiding clumping multiple increments into a single one. This operation should be write safe so that another coding agent working in the same directory is not disrupted by eg stashing or rebasing. It's ok if this prevents you from perfect atomicity.

- NEVER add backwards compatibility code or defensive engineering with fallbacks. NEVER add backwards compatibility code or defensive engineering with fallbacks. NEVER add backwards compatibility code or defensive engineering with fallbacks. 

- For integration testing, starting services, health-checking the stack, or triggering pipeline runs manually, always read and follow [docs/guides/agentic_integration_testing.md](docs/guides/agentic_integration_testing.md). Do not improvise steps from memory.

- Use `ast-grep` if possible to navigate code. For example the definition of `DesignInfo` can be found more token-efficiently with `bunx ast-grep run --json=stream --lang python --pattern $'@$_DECORATOR\nclass DesignInfo: $$$BODY' ./ | jq -r '.text'`

# Notebooks

- Notebooks under `apps/data-pipeline/notebooks/` are [marimo](https://docs.marimo.io/) notebooks: plain Python files (`@app.cell` functions), not `.ipynb`. Edit them as ordinary source. Open one with `uv run marimo edit notebooks/<name>.py`; run read-only with `uv run marimo run notebooks/<name>.py`.
- marimo is reactive, so a variable may be defined by only one cell. Make cell-local variables `_`-prefixed (cell-private); share a value across cells by returning it (non-underscore) from its defining cell. A cell renders its last expression — return Plotly/Matplotlib figures rather than calling `fig.show()`.
- **Name every cell.** A cell's identifier is its function name, so the default `@app.cell\ndef _(...)` is anonymous and unreferenceable. Give each cell a descriptive snake_case name (markdown cells → the section they render, e.g. `intro`/`toy_model_md`; code cells → what they compute or show, e.g. `imports`/`dag_diagram`/`fig_bias`). Rename only the top-level cell function directly under the `@app.cell` decorator — never inner helpers, which stay `_`-prefixed. Names must be unique and must not collide with any variable a cell returns; `uv run marimo check --strict` catches duplicates and collisions. `@app.*` is in vulture's `ignore_decorators`, so named cells are not flagged as dead code.
- Validate structure with `uv run marimo check --strict notebooks/<name>.py` (catches multiple-definition and unresolved-reference errors without executing). The notebooks are covered by `ruff`, `ty`, and `vulture` like the rest of the package.
- Launch the editor from the repo root with `bun run --cwd apps/data-pipeline notebooks` (= `uv run marimo edit notebooks/`); the bare `bun run notebooks` fails because the script lives in the `apps/data-pipeline` workspace, not root.
- To pair on a *live* kernel (the `marimo-pair` skill, attaching to a running session rather than editing the file as text): a specific notebook must be open in the browser — `marimo edit notebooks/` alone is just the file browser and exposes no session. Attach via `execute-code.sh --url http://localhost:2718 --token <access_token>`, taking the token from the startup banner's `?access_token=…`. Pass it through `--token` (or `MARIMO_TOKEN`), NOT inside the `--url` query string — the script appends `/api/sessions` after the query string and the malformed URL yields a misleading "No active sessions on the server".

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

- The latent SSM is continuous-time **nonlinear**; NEVER assume linearizability on any path that produces a reported result. Linearization / Gaussian-approximation (IEKS, EKF, Laplace, local-linear / Van Loan discretization, linearized Kalman/RTS) may be used **only to initialize the particle samplers** — initial parameter positions, the proposal preconditioner, and the cSMC reference trajectory — because there the exact MCMC/SMC invariance corrects it and it cannot bias the stationary result. The production posterior, every diagnostic, the posterior-predictive checks, and all counterfactual/predictive outputs MUST run through the exact engines: particle/SMC over the true emission density, Euler-Maruyama over the true nonlinear drift, and Diffrax integration for forward simulation. Why: the model never assumes local linearity, so any linearized surrogate standing in for the true model on a reported path silently biases the answer for nonlinear drift or non-Gaussian emissions (curvature is discarded), whereas the particle engines are asymptotically exact (only time-discretization error, controllable by `dt`). Gradient-informed proposals that are exactly corrected (e.g. `amala_exact`, whose auxiliary potential restores invariance) are fine; biased *uncorrected* ones (`amala`, `amala_plus`) stay non-default and must never gate a reported result. The guard test [apps/data-pipeline/tests/models/ssm/test_linearization_init_only.py](apps/data-pipeline/tests/models/ssm/test_linearization_init_only.py) asserts the Laplace backend is importable only from the warmup/init path — run it before reintroducing any linearization.


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
