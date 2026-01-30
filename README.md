# dsem-agent

This project explores an end-to-end, LLM-orchestrated framework for causal inference over long-context, multi-source data (e.g. large document collections or aggregated web search). An "orchestrator" LLM proposes candidate variables, time granularities, and a causal DAG; "worker" LLMs then populate those dimensions at scale, after which we use y0 for identifiability checks (via Pearl's ID algorithm), and PyMC for full Bayesian GLM estimation with LLM-elicited priors. The goal is to build a system that not only estimates causal effects and counterfactuals from messy, high-dimensional evidence, but also knows when to trust those numeric estimates and when to fall back to purely structural, qualitative reasoning.

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
- DSPy for prompt optimization
- NetworkX for causal DAG representation
- y0 for identifiability checks (Pearl's ID algorithm)
- PyMC for Bayesian GLM estimation
- ArViz for posterior diagnostics

## Documentation

See [`docs/index.md`](docs/index.md) for the full documentation structure.

- **[Modeling](docs/modeling/)** - Theoretical foundations: scope, DSEM overview, assumptions, theory
- **[Reference](docs/reference/)** - Technical specifications: schemas, pipeline stages
- **[Guides](docs/guides/)** - Practical usage: quickstart, data workflow, running evals

## Structure

```
dsem-agent/
├── data/
│   ├── raw/           # Raw input data (gitignored)
│   ├── processed/     # Converted text chunks (gitignored)
│   ├── queries/       # Test queries for pipeline
│   └── eval/          # Example DAGs for evals
├── docs/
│   ├── modeling/      # Theoretical foundations (scope, DSEM, assumptions, theory)
│   ├── reference/     # Technical specs (schemas, pipeline)
│   └── guides/        # Practical usage (quickstart, data, evals)
├── evals/             # Inspect AI evals (eval{N}_{name}.py) + scripts/
│   └── deprecated/    # Deprecated evals
├── src/dsem_agent/
│   ├── orchestrator/  # Two-stage model specification (latent + measurement)
│   │   ├── agents.py  # Stage 1a: latent model, Stage 1b: measurement model
│   │   ├── prompts.py # LLM prompts for both stages
│   │   └── schemas.py # Construct, Indicator, LatentModel, MeasurementModel, DSEMModel
│   ├── workers/       # Indicator extraction LLMs
│   ├── causal/        # y0 identifiability, sensitivity analysis
│   ├── models/        # PyMC GLM specification
│   ├── flows/         # Prefect pipeline + stages/
│   │   └── stages/    # stage1a_latent, stage1b_measurement, stage2_workers, ...
│   └── utils/         # Shared utilities
├── tests/             # pytest tests (test_{name}.py)
└── tools/             # CLI tools + UIs (dag_cli.py, dag_explorer.py, log readers)
```
