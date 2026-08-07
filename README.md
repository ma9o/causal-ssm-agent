# nof1-causal-lab

[![CI](https://github.com/ma9o/nof1-causal-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ma9o/nof1-causal-lab/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?logo=python&logoColor=white)
![Next.js 16](https://img.shields.io/badge/Next.js-16-000?logo=next.js)
![NumPyro + JAX](https://img.shields.io/badge/NumPyro-JAX-9b59b6)
![Prefect](https://img.shields.io/badge/Prefect-3-070e10?logo=prefect&logoColor=white)

**nof1-causal-lab** is an opinionated LLM harness for end-to-end Bayesian causal inference on N-of-1 time series data.

The ultimate goal of the project is to facilitate epistemically optimal decision-making at the individual level, using dense digital trace datasets (medical records, chatbot conversation logs, browsing history, etc.) while transparently incorporating existing scientific knowledge, where available, in the form of prior distributions and modeling assumptions.

The user will pose a question in natural language given a dataset of their choosing. First, the system will lay out the causal DAG implied by the question and a measurement structure for the DAG that is compatible with the given dataset. If the causal effect in question is structurally identifiable, the DAG is translated into a continuous-time state-space model and estimated with MCMC. Finally, an LLM will run simulations on the fitted model to estimate the causal effects of interventions and counterfactual scenarios that answer the original question.

```mermaid
flowchart LR
  Q(["Question"])
  DS(["Dataset"])
  L["Causal DAG"]
  M["Measurement\nstructure"]
  ID{"Identified?"}
  MS["Statistical model specification\n& estimation"]
  SIM["Simulation"]
  R(["Causal effect\nestimate"])

  Q --> L 
  DS --> M 
  L --> M --> ETL --> ID
  ID -- yes --> MS
  subgraph Bayesian construct-admission state machine
  MS --> SIM
  end
  SIM --> R
  ID -- no --> L
```

In practice, the framework is designed for longitudinal consumer datasets that are easily accessible via data subject access requests (DSARs), from more domain specific ones like Apple Health, Oura, 23andMe, Strava (cardiometabolic health, chronic conditions, performance & adaptation), Anki, Duolingo, YouTube (education & deliberate practice) to more cross cutting ones like Google Takeout, WhatsApp, ChatGPT/Claude logs (mental health, cognition & attention, habit & behavior change) - and, most interestingly, their intersections!

## Features and Goals

- **Methodological rigor without friction** - An user should be simply able to provide a dataset and a question, and the software should provide the most rigorous possible answer without pushing any methodological decision onto the user.
- **Interpretability and interactivity** - At any stage, users can inspect and intervene on the LLM outputs in the UI, either by interactively challenging the LLM in conversation or directly overriding its decisions.
- **Support for large datasets, irregular timestamps and semantic heterogeneity** - via a continuous-discrete nonlinear state-space model — continuous-time latent dynamics observed at discrete, irregular timestamps — with non-Gaussian indicator-specific likelihoods (Poisson, Bernoulli, Beta, etc.).
- **Robust LLM-based numerical modeling and prior elicitation** - by admitting one construct at a time, checkpointing every accepted contribution, and gating progression on exact prior-predictive reachability checks.
- **Fast and accurate parameter and state estimation in `jax`** - Exact inference in minutes using [parallel-in-time particle smoothing](https://arxiv.org/pdf/2401.14868) on GPU. Efficient caching ensures that we never waste time waiting for compilation.
- **Compatible with `codex` and `claude-code`** - Leverage your existing subscription for the interactive stages of the pipeline.

## Demo

[https://project-n98yx.vercel.app/analysis/DEMO](https://project-n98yx.vercel.app/analysis/DEMO)

| <img src="docs/assets/stage1b.png" width="400" alt="stage2"><br>Causal design and measurement structure | <img src="docs/assets/stage2.gif" width="400" alt="stage2"><br>Parallel data extraction |
|:--:|:--:|
| <img src="docs/assets/stage4-loading.gif" width="400" alt="stage2"><br>**Construct-admission state machine** | <img src="docs/assets/stage4-done.gif" width="400" alt="stage4"><br>**Statistical model specification** <tr></tr> |
| <img src="docs/assets/stage5.gif" width="400" alt="stage5"><br>**Inference diagnostics** | <img src="docs/assets/stage6.gif" width="400" alt="stage2"><br>**Counterfactual simulation** |

## Modeling

Causal identification on the SSM is achieved by temporal unrolling the DAG as per [Jahn et al. (2025)](https://proceedings.mlr.press/v275/jahn25a.html) then running the [ID algorithm](https://doi.org/10.1016/j.artint.2008.12.006) on the unrolled segment for each treatment-outcome pair.

The system is a continuous-discrete nonlinear state-space model: the latent constructs evolve in continuous time as a stochastic differential equation, observed at discrete and possibly irregular times. The drift is a composite vector field — per-construct decay and intercepts plus linear, saturating (Hill), and bilinear edges — so the dynamics are nonlinear in general:

<!-- docs-latex:start eyJkaXNwbGF5Ijp0cnVlLCJsYXRleCI6ImRcXGJvbGRzeW1ib2x7XFxldGF9KHQpID0gXFxtYXRoYmZ7Zn1cXGJpZ2woXFxib2xkc3ltYm9se1xcZXRhfSh0KSwgdDsgXFxib2xkc3ltYm9se1xcdGhldGF9XFxiaWdyKVxcLGR0ICsgXFxtYXRoYmZ7R31cXCxkXFxtYXRoYmZ7V30odCkifQ -->
<p align="center">
  <img src="docs/assets/generated/latex/display-a10ec81048d5b964206d.svg" alt="LaTeX: d\boldsymbol{\eta}(t) = \mathbf{f}\bigl(\boldsymbol{\eta}(t), t; \boldsymbol{\theta}\bigr)\,dt + \mathbf{G}\,d\mathbf..." width="400">
</p>
<!-- docs-latex:end -->

Observations follow indicator-specific likelihoods (see the supported [distribution families](docs/reference/statistical-model-spec/likelihoods.md#distribution-families) and [link functions](docs/reference/statistical-model-spec/likelihoods.md#link-functions)):

<!-- docs-latex:start eyJkaXNwbGF5Ijp0cnVlLCJsYXRleCI6InlfaSh0KSBcXG1pZCBcXGJvbGRzeW1ib2x7XFxldGF9KHQpIFxcc2ltIEZfaVxcIVxcbGVmdChnX2leey0xfVxcbGVmdCgoXFxib2xkc3ltYm9se1xcTGFtYmRhfVxcYm9sZHN5bWJvbHtcXGV0YX0odCkrXFxib2xkc3ltYm9se1xcbXV9KV9pXFxyaWdodCk7IFxcdGhldGFfaVxccmlnaHQpIn0 -->
<p align="center">
  <img src="docs/assets/generated/latex/display-30c67cbe14177fd65ee8.svg" alt="LaTeX: y_i(t) \mid \boldsymbol{\eta}(t) \sim F_i\!\left(g_i^{-1}\left((\boldsymbol{\Lambda}\boldsymbol{\eta}(t)+\boldsymbol{..." width="461">
</p>
<!-- docs-latex:end -->

See [causal-design](docs/reference/causal-design/identifiability.md), [latent-structure](docs/reference/latent-structure/assumptions.md) and [measurement-structure](docs/reference/measurement-structure/assumptions.md) for the structural assumptions baked into the modeling framework.

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
| **[Guides](docs/guides/)** | Dev setup, code generation, integration testing, evaluations |

## Project Structure

```text
nof1-causal-lab/                    # Turborepo monorepo
├── apps/
│   ├── data-pipeline/               # Python — Prefect pipeline + NumPyro models
│   │   ├── src/nof1_causal_lab/
│   │   │   ├── flows/               # Prefect stages (0–6) + orchestration
│   │   │   ├── orchestrator/        # LLM agents: construct, measurement, prior proposals
│   │   │   ├── workers/             # Parallel indicator extraction + prior research
│   │   │   ├── models/              # NumPyro SSM compilation + predictive checks
│   │   │   ├── distributions.py     # Indicator-specific likelihoods (Poisson, Bernoulli, Beta, …)
│   │   │   ├── tool_server.py       # LLM tools server 
│   │   │   └── utils/               # Identifiability, config, LLM runtime
│   │   ├── notebooks/               # Exploratory analysis
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
