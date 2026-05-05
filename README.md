# causal-ssm-agent

[![CI](https://github.com/ma9o/causal-ssm-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ma9o/causal-ssm-agent/actions/workflows/ci.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?logo=python&logoColor=white)
![Next.js 16](https://img.shields.io/badge/Next.js-16-000?logo=next.js)
![NumPyro + JAX](https://img.shields.io/badge/NumPyro-JAX-9b59b6)

**causal-ssm-agent** is an opinionated LLM harness for end-to-end Bayesian causal inference on N-of-1 time series data.

The ultimate goal of the project is to facilitate epistemically optimal decision-making at the individual level, uisng high-leverage digital trace datasets (medical records, chatbot conversation logs, browsing history, etc.) while transparently incorporating existing scientific knowledge - where available - in the form of prior distributions and modeling assumptions. 

In practice, the user will pose a question in natural language given a dataset of their choosing. First, the system will lay out the causal DAG implied by the question and a measurment model for the DAG that is compatible with the given dataset. If the causal effect in question is structurally identifiable, the DAG is translated into a continous-time state-space model and estimated with MCMC. Finally, an LLM will run simulations on the fitted model to estimate the causal effects of interventions and counterfactual scenarios that answer the original question.

```mermaid
flowchart LR
  Q(["Question"])
  DS(["Dataset"])
  L["Causal DAG"]
  M["Measurement\nmodel proposal"]
  DSC["Data extraction &\nvalidation"]
  ID{"Identified?"}
  MS["Model\nspecification"]
  P["Priors &\nlikelihoods"]
  EST["Estimation"]
  SIM["Simulation"]
  R(["Causal effect\nestimate"])

  Q --> L 
  DS --> M 
  L --> M --> DSC --> ID
  ID -- yes --> MS
  subgraph Bayesian modeling state machine 
  MS --> P --> EST
  end
  EST --> SIM --> R
  ID -- no --> L
```

<!-- TODO demo video -->

## Features and Goals

- **Methodological rigor without friction** - An user should be simply able to provide a dataset and a question, and the software should provide the most rigorous possible answer without pushing any methodological decision onto the user.
- **Interpretability and interactivity** - At any stage, users can inspect and intervene on the LLM outputs in the UI, either by interactively challenging the LLM in conversation or directly overriding its decisions.
- **Support for large datasets, irregular timestamps and semantic heterogeneity** - via multivariate continuous-time Ornstein-Uhlenbeck dynamics with non-Guassian indicator-specific likelihoods (Poisson, Bernoulli, Beta, etc.). 
- **Robust LLM-based numerical modeling and prior elicitation** - by embedding the LLM decision process in a state machine that minimizes the LLM's decision surface at each step, and gates progression on numerical checks (e.g. prior/posterior predictive, SDE stability, scale adequacy, etc.)
- **Fast and accurate MCMC estimation in `jax`** - Exact inference in minutes using O(log T) associative Kalman filtering on GPU ([Corenflos et al. 2025](https://arxiv.org/abs/2303.00301)). Efficient caching ensures that we never waste time waiting for compilation.
- **Compatibile with `codex` and `claude-code`** - Leverage your existing subscription for the interactive stages of the pipeline.

## Modeling

Causal identification on the SSM is achieved by temporal unrolling the DAG as per [Jahn et al. (2025)](https://proceedings.mlr.press/v275/jahn25a.html) then running the [ID algorithm](https://doi.org/10.1016/j.artint.2008.12.006) on the unrolled segment for each treatment-outcome pair.

The latent dynamics are modeled as a multivariate Ornstein-Uhlenbeck process:

$$d\boldsymbol{\eta}(t) = \bigl(\mathbf{A}\,\boldsymbol{\eta}(t) + \mathbf{c}\bigr)\,dt + \mathbf{G}\,d\mathbf{W}(t)$$

with indicator-specific likelihoods (see the supported [distribution families](docs/reference/model-spec/likelihoods.md#distribution-families) and [link functions](docs/reference/model-spec/likelihoods.md#link-functions)):

$$y_i(t) \mid \boldsymbol{\eta}(t) \sim F_i\!\left(g_i^{-1}\left((\boldsymbol{\Lambda}\boldsymbol{\eta}(t)+\boldsymbol{\mu})_i\right); \theta_i\right)$$

See [causal-spec](docs/reference/causal-spec/identifiability.md), [latent-model](docs/reference/latent-model/assumptions.md) and [measurement-model](docs/reference/measurement-model/assumptions.md) for the structural assumptions baked into the modeling framework.

## Quick Start

```bash
bun install --frozen-lockfile
cd apps/data-pipeline && uv sync --frozen --group dev && cd ../..

# Environment — set OPENROUTER_API_KEY at minimum
cp .env.example.dev .env

# Start all dev servers - NextJS, Prefect and LLM tools server
bun run dev
```

See the [dev setup guide](docs/guides/dev_setup.md) for full details including environment variables and optional dependencies.

## Documentation

| Section | Description |
|---------|-------------|
| **[Index](docs/index.md)** | Full documentation entrypoint with route-by-question |
| **[Pipeline](docs/pipeline.md)** | Stage-by-stage walkthrough and artifact ownership |
| **[Reference](docs/reference/)** | Assumptions, compilation, estimation, inference routing |
| **[Guides](docs/guides/)** | Dev setup, codegen, integration testing, evals |

## Project Structure

```text
causal-ssm-agent/                    # Turborepo monorepo
├── apps/
│   ├── data-pipeline/               # Python — Prefect pipeline + NumPyro models
│   │   ├── src/causal_ssm_agent/
│   │   │   ├── orchestrator/        # LLM agents: construct, measurement, prior proposals
│   │   │   ├── workers/             # Parallel indicator extraction + prior research
│   │   │   ├── models/              # NumPyro SSM compilation + likelihoods
│   │   │   ├── flows/               # Prefect stages (0–6) + orchestration
│   │   │   └── utils/               # Identifiability, config, LLM runtime
│   │   ├── evals/                   # Inspect AI evaluation suites
│   │   └── tests/                   # pytest suite
│   └── web/                         # Next.js — interactive frontend
│       └── src/
│           ├── app/                  # App router + API routes
│           ├── components/           # DAG editor, charts, pipeline views
│           └── lib/                  # Clients, hooks, types
├── packages/
│   └── api-types/                   # Generated TypeScript types from Pydantic
├── data/                            # Workspace data (local inputs, runs, artifacts)
└── docs/                            # Documentation
```

<!-- cloc:start -->

## Lines of Code

| Language | Files | Blank | Comment | Code | p50&nbsp;/&nbsp;p90&nbsp;/&nbsp;max | Top files |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Python | 345 | 17,381 | 10,499 | 104,798 | 157&nbsp;/&nbsp;698&nbsp;/&nbsp;9,076 | [test_stage4.py](<apps/data-pipeline/tests/test_stage4.py>), [test_inference_strategies.py](<apps/data-pipeline/tests/test_inference_strategies.py>), [test_pipeline.py](<apps/data-pipeline/tests/test_pipeline.py>) |
| TypeScript | 322 | 3,953 | 1,178 | 35,571 | 71&nbsp;/&nbsp;253&nbsp;/&nbsp;1,076 | [generate-markdown.ts](<apps/web/src/lib/utils/generate-markdown.ts>), [generate-markdown.test.ts](<apps/web/src/lib/utils/generate-markdown.test.ts>), [_shared.test.ts](<apps/web/src/app/api/analysis/_shared.test.ts>) |
| JSON | 52 | 0 | 0 | 32,408 | 169&nbsp;/&nbsp;1,165&nbsp;/&nbsp;6,320 | [stage-4.json](<data/DOCTOLIB/run/stage-4.json>), [stage-5b.json](<data/DOCTOLIB/run/stage-5b.json>), [contracts.json](<packages/api-types/schemas/contracts.json>) |
| Markdown | 36 | 1,400 | 9 | 4,266 | 59&nbsp;/&nbsp;171&nbsp;/&nbsp;1,024 | [report.md](<data/DOCTOLIB/report.md>), [report.md](<data/DEFAULT/report.md>), [llm-driven-specification.md](<docs/reference/model-spec/llm-driven-specification.md>) |
| Jupyter Notebook | 9 | 0 | 68,025 | 3,452 | 277&nbsp;/&nbsp;901&nbsp;/&nbsp;901 | [stage4_manual_golden_repair.ipynb](<apps/data-pipeline/notebooks/stage4_manual_golden_repair.ipynb>), [pathological_geometries_gallery.ipynb](<apps/data-pipeline/notebooks/pathological_geometries_gallery.ipynb>), [pathfinder_gallery.ipynb](<apps/data-pipeline/notebooks/pathfinder_gallery.ipynb>) |
| CSV | 2 | 0 | 0 | 642 | 321&nbsp;/&nbsp;321&nbsp;/&nbsp;321 | [expected-stage2-model-data.csv](<data/MEDICAL_SEMANTICS/expected-stage2-model-data.csv>), [expected-stage2-raw-data.csv](<data/MEDICAL_SEMANTICS/expected-stage2-raw-data.csv>) |
| YAML | 10 | 53 | 82 | 337 | 1&nbsp;/&nbsp;80&nbsp;/&nbsp;122 | [ci.yml](<.github/workflows/ci.yml>), [deploy.yml](<.github/workflows/deploy.yml>), [config.yaml](<apps/data-pipeline/config.yaml>) |
| JavaScript | 4 | 45 | 10 | 231 | 17&nbsp;/&nbsp;165&nbsp;/&nbsp;165 | [update_readme_cloc.js](<scripts/update_readme_cloc.js>), [copy-perspective-assets.mjs](<apps/web/scripts/copy-perspective-assets.mjs>), [eslint.config.mjs](<apps/web/eslint.config.mjs>) |
| Bourne Shell | 1 | 37 | 0 | 180 | 180&nbsp;/&nbsp;180&nbsp;/&nbsp;180 | [start_agentic_integration_stack.sh](<scripts/start_agentic_integration_stack.sh>) |
| TOML | 1 | 10 | 0 | 117 | 117&nbsp;/&nbsp;117&nbsp;/&nbsp;117 | [pyproject.toml](<apps/data-pipeline/pyproject.toml>) |
| CSS | 1 | 6 | 1 | 100 | 100&nbsp;/&nbsp;100&nbsp;/&nbsp;100 | [globals.css](<apps/web/src/app/globals.css>) |
| Bourne Again Shell | 2 | 7 | 5 | 22 | 2&nbsp;/&nbsp;20&nbsp;/&nbsp;20 | [sync-private-data](<.githooks/sync-private-data>), [post-merge](<.githooks/post-merge>) |
| SVG | 5 | 0 | 0 | 5 | 1&nbsp;/&nbsp;1&nbsp;/&nbsp;1 | [file.svg](<apps/web/public/file.svg>), [globe.svg](<apps/web/public/globe.svg>), [next.svg](<apps/web/public/next.svg>) |
| Text | 3 | 0 | 0 | 3 | 1&nbsp;/&nbsp;1&nbsp;/&nbsp;1 | [query.txt](<data/DEFAULT/query.txt>), [query.txt](<data/DOCTOLIB/query.txt>), [query.txt](<data/MEDICAL_SEMANTICS/query.txt>) |
| **Total** | **793** | **22,892** | **79,809** | **182,132** | **97&nbsp;/&nbsp;510&nbsp;/&nbsp;9,076** | **[test_stage4.py](<apps/data-pipeline/tests/test_stage4.py>), [stage-4.json](<data/DOCTOLIB/run/stage-4.json>), [stage-5b.json](<data/DOCTOLIB/run/stage-5b.json>)** |

<!-- cloc:end -->
