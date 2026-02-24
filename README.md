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

- **[Modeling](docs/modeling/)** - Theoretical foundations: scope, assumptions, theory, estimation
- **[Reference](docs/reference/)** - Technical specifications: schemas, pipeline stages
- **[Guides](docs/guides/)** - Practical usage: quickstart, data workflow, running evals
- **[Mplus Parity](docs/reference/mplus-parity.md)** - Feature comparison with Asparouhov et al. (2017)

## Structure

```
causal-ssm-agent/                  # Turborepo monorepo
├── apps/
│   ├── data-pipeline/             # Python – Prefect pipeline + NumPyro models
│   │   ├── src/causal_ssm_agent/
│   │   │   ├── orchestrator/      # LLM model specification (latent + measurement)
│   │   │   ├── workers/           # Indicator extraction + prior research LLMs
│   │   │   ├── models/            # NumPyro SSM, likelihoods, prior/posterior predictive
│   │   │   ├── flows/             # Prefect pipeline stages (1a → 5)
│   │   │   └── utils/             # Shared utilities (config, llm, data, identifiability)
│   │   ├── benchmarks/            # Inference method benchmarks (parameter recovery)
│   │   ├── data/                  # Raw, processed, queries, eval data
│   │   ├── docs/                  # Modeling theory, reference specs, guides
│   │   ├── evals/                 # Inspect AI evals
│   │   ├── notebooks/             # Showcase notebooks
│   │   ├── tests/                 # pytest tests
│   │   └── tools/                 # CLI tools + UIs
│   └── web/                       # Next.js frontend
│       └── src/
│           ├── app/               # Next.js app router pages
│           ├── components/        # React components
│           └── lib/               # Client-side utilities
├── packages/
│   ├── api-types/                 # Generated TypeScript types (from pipeline schemas)
│   └── typescript-config/         # Shared TS config
├── docs/                          # Top-level codegen guide
└── scratchpad/                    # Temporary work files (gitignored)
```
