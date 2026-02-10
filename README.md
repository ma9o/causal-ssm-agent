# dsem-agent

This project explores an end-to-end, LLM-orchestrated framework for causal inference over long-context, multi-source data (e.g. large document collections or aggregated web search). An "orchestrator" LLM proposes candidate variables, time granularities, and a causal DAG; "worker" LLMs then populate those dimensions at scale, after which we use y0 for identifiability checks (via Pearl's ID algorithm), and NumPyro for full Bayesian Continuous-Time SEM (CT-SEM) estimation with LLM-elicited priors. The goal is to build a system that not only estimates causal effects and counterfactuals from messy, high-dimensional evidence, but also knows when to trust those numeric estimates and when to fall back to purely structural, qualitative reasoning.

**Key Innovation: Continuous-Time Modeling**

Unlike traditional discrete-time approaches that require upfront aggregation, this framework uses Continuous-Time Structural Equation Modeling (CT-SEM) which:
- Handles irregularly-spaced observations natively via Kalman filtering
- Avoids information loss from pre-aggregation
- Models dynamics via stochastic differential equations
- Supports hierarchical (multi-subject) panel data

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
- JAX/NumPyro for Bayesian CT-SEM estimation
- dynamax for Kalman likelihood computation
- cuthbert for differentiable particle filtering
- Multiple inference backends: SVI, NUTS, Hess-MC², PGAS, Tempered SMC

## Documentation

See [`docs/index.md`](docs/index.md) for the full documentation structure.

- **[Modeling](docs/modeling/)** - Theoretical foundations: scope, DSEM overview, assumptions, theory
- **[Reference](docs/reference/)** - Technical specifications: schemas, pipeline stages
- **[Guides](docs/guides/)** - Practical usage: quickstart, data workflow, running evals
- **[DSEM Parity](docs/reference/dsem-parity.md)** - Feature comparison with Asparouhov et al. (2017)

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
│   ├── reference/     # Technical specs (schemas, pipeline, dsem-parity)
│   ├── guides/        # Practical usage (quickstart, data, evals)
│   └── papers/        # Reference papers (Asparouhov 2017, etc.)
├── evals/             # Inspect AI evals (eval{N}_{name}.py) + scripts/
│   └── deprecated/    # Deprecated evals
├── src/dsem_agent/
│   ├── orchestrator/  # Two-stage model specification (latent + measurement)
│   │   ├── agents.py  # Stage 1a: latent model, Stage 1b: measurement model
│   │   ├── prompts/   # LLM prompts for all stages
│   │   │   ├── latent_model.py      # Stage 1a prompts
│   │   │   ├── measurement_model.py # Stage 1b prompts
│   │   │   └── model_proposal.py    # Stage 4 model specification prompts
│   │   ├── schemas.py        # Construct, Indicator, LatentModel, MeasurementModel, DSEMModel
│   │   ├── schemas_model.py  # ModelSpec, ParameterSpec for Stage 4
│   │   └── stage4_orchestrator.py   # Stage 4 orchestrator logic
│   ├── workers/       # Indicator extraction + prior research LLMs
│   │   ├── agents.py         # Stage 2 worker agents
│   │   ├── schemas.py        # Worker output schemas
│   │   ├── schemas_prior.py  # PriorProposal, PriorValidationResult
│   │   ├── prior_research.py # Stage 4 worker prior research
│   │   └── prompts/          # Worker prompts
│   ├── causal/        # y0 identifiability, sensitivity analysis
│   ├── models/        # NumPyro state-space model specification
│   │   ├── ssm/                # State-space model implementation
│   │   │   ├── model.py        # SSMModel, SSMSpec, SSMPriors
│   │   │   ├── inference.py    # fit() dispatcher + InferenceResult
│   │   │   ├── hessmc2.py      # Hess-MC² (SMC with CoV L-kernels)
│   │   │   ├── pgas.py         # PGAS (Gibbs CSMC + MALA parameters)
│   │   │   ├── tempered_smc.py # Tempered SMC + preconditioned HMC/MALA
│   │   │   ├── mcmc_utils.py   # Shared MCMC utilities (HMC step, mass matrix)
│   │   │   ├── utils.py        # Shared site discovery and matrix assembly
│   │   │   ├── discretization.py # CT→DT conversion (incl. batched vmap)
│   │   │   └── core.py         # Utility functions
│   │   ├── likelihoods/        # State-space likelihood backends
│   │   │   ├── base.py         # Protocol, CTParams, MeasurementParams, InitialStateParams
│   │   │   ├── kalman.py       # Kalman filter via dynamax lgssm_filter
│   │   │   ├── particle.py     # Bootstrap PF via cuthbert (auto-upgrades to RBPF)
│   │   │   └── rao_blackwell.py # Rao-Blackwell PF (Kalman + quadrature)
│   │   ├── ssm_builder.py      # SSMModelBuilder for pipeline integration
│   │   └── prior_predictive.py # Prior predictive validation
│   ├── flows/         # Prefect pipeline + stages/
│   │   └── stages/    # stage1a..4_model, stage4b_parametric_id, stage5_inference
│   └── utils/         # Shared utilities (config, llm, data, parametric_id, etc.)
├── benchmarks/        # Inference method benchmarks (parameter recovery)
│   ├── problems/      # Standardized test problems (ground truths)
│   ├── metrics.py     # Recovery metrics (RMSE, coverage, reporting)
│   ├── modal_infra.py # Shared Modal GPU setup
│   ├── run.py         # Unified CLI (--method pgas/tempered_smc/all)
│   └── results.md     # Empirical results across methods
├── notebooks/         # PyMC showcase notebooks (tracked)
├── scratchpad/        # Temporary work files (gitignored contents)
├── tests/             # pytest tests (test_{name}.py)
└── tools/             # CLI tools + UIs (dag_cli.py, dag_explorer.py, log readers)
```
