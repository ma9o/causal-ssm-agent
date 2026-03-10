# causal-ssm-agent

This project explores an end-to-end, LLM-orchestrated framework for causal inference over long-context, multi-source data (e.g. large document collections or aggregated web search). An "orchestrator" LLM proposes candidate variables, time granularities, and a causal DAG; "worker" LLMs then populate those dimensions at scale, after which we use y0 for identifiability checks (via Pearl's ID algorithm), and NumPyro for full Bayesian state-space model estimation with LLM-elicited priors. The goal is to build a system that not only estimates causal effects and counterfactuals from messy, high-dimensional evidence, but also knows when to trust those numeric estimates and when to fall back to purely structural, qualitative reasoning.

**Key Innovation: Continuous-Time Modeling**

Unlike traditional discrete-time approaches that require upfront aggregation, this framework uses continuous-time state-space modeling which:
- Handles irregularly-spaced observations natively via Kalman/particle filtering
- Avoids information loss from pre-aggregation
- Models dynamics via stochastic differential equations
- Supports hierarchical (multi-subject) panel data
- Computes counterfactual effects via do-operator on CT steady states

## Key Feature: Natural Language Causal Queries

Users don't need to be data scientists or understand causal inference terminology. They can ask questions in plain language:

- *"Why do I feel tired on Mondays?"*
- *"Does talking to my therapist actually help?"*
- *"What's making my code reviews take so long?"*

The orchestrator LLM translates these informal queries into formal causal structures - identifying relevant variables, potential confounders, and constructing a proper DAG. This democratizes causal inference, making it accessible to anyone with data and curiosity.

## Tech stack

- polars dataframes
- uv
- Prefect for pipeline orchestration
- AISI's Inspect agent framework
- NetworkX for causal DAG representation
- y0 for identifiability checks (Pearl's ID algorithm)
- JAX/NumPyro for Bayesian SSM estimation
- cuthbert for differentiable Kalman filtering and particle filtering
- Multiple inference backends: SVI, NUTS, NUTS-DA, Hess-MC², PGAS, Tempered SMC, Laplace-EM, Structured VI, DPF

## Documentation

See [`docs/index.md`](docs/index.md) for the full documentation structure.

- **[Modeling](docs/modeling/)** - Theoretical foundations: scope, assumptions, estimation
- **[Guides](docs/guides/)** - Practical usage: data workflow, running evals, codegen

## Structure

```
causal-ssm-agent/                  # Turborepo monorepo
├── apps/
│   ├── data-pipeline/             # Python – Prefect pipeline + NumPyro models
│   │   ├── src/causal_ssm_agent/
│   │   │   ├── orchestrator/      # LLM model specification (latent + measurement)
│   │   │   ├── workers/           # Indicator extraction + prior research LLMs
│   │   │   ├── models/            # NumPyro SSM, compiler, likelihoods, prior/posterior predictive
│   │   │   ├── flows/             # Prefect pipeline stages (0 → 5)
│   │   │   └── utils/             # Shared utilities (config, llm runtime, LiteLLM client, data, identifiability)
│   │   ├── benchmarks/            # Inference method benchmarks (parameter recovery)
│   │   ├── data/                  # Raw, processed, queries, eval data
│   │   ├── evals/                 # Inspect AI evals
│   │   ├── notebooks/             # Showcase notebooks
│   │   ├── tests/                 # pytest tests
│   │   │   ├── test_pipeline.py   # Flow orchestration and replay contract coverage
│   │   │   └── test_stage2_extract.py  # Stage 2 worker collection and progress logging coverage
│   │   └── tools/                 # CLI tools + UIs
│   └── web/                       # Next.js frontend
│       └── src/
│           ├── app/               # Next.js app router pages
│           ├── components/        # React components (stages, charts, DAG, pipeline)
│           └── lib/               # API clients, hooks, types, utilities
│               └── hooks/         # Prefect progress state, event streaming, stage telemetry
├── packages/
│   ├── api-types/                 # Generated TypeScript types (from pipeline schemas)
│   └── typescript-config/         # Shared TS config
├── docs/                          # Project documentation
│   ├── modeling/                  # Theoretical foundations
│   ├── reference/                 # Technical specifications
│   ├── guides/                    # Practical usage guides
│   └── literature.md              # Reference papers (links + summaries)
└── scratchpad/                    # Temporary work files (gitignored)
```
