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
│   │   │   │   └── windows.py     # Support-window planning and chunking for stage 2 extraction
│   │   │   ├── models/            # NumPyro SSM, pure compilation pipeline, likelihoods, prior/posterior predictive
│   │   │   │   ├── likelihoods/   # Observation families, support-aware emissions, and trajectory observation operators
│   │   │   │   ├── ssm_compilation.py             # Public pure compilation entrypoint (translate -> compile priors -> bind)
│   │   │   │   ├── ssm_spec_translation.py        # ModelSpec -> SSMSpec translation and causal mask construction
│   │   │   │   ├── ssm_prior_indexing.py          # Semantic parameter name -> SSMPriors slot mapping
│   │   │   │   ├── ssm_prior_compilation.py       # Prior transforms, lag checks, and sample-site bindings
│   │   │   │   ├── ssm_observation_metadata.py    # Observation-family metadata hydration and support validation
│   │   │   │   ├── ssm/inference_structure.py     # Shared likelihood-path + auto-routing + first-pass RB planning
│   │   │   │   └── ssm/prior_predictive_runtime.py  # Compile-stable prior predictive runtime from serialized semantics
│   │   │   ├── flows/             # Prefect pipeline stages (0 → 6) + replay/resume orchestration
│   │   │   │   └── stages/llm_stage_task.py       # Shared Prefect task factory for LLM-backed stages
│   │   │   └── utils/             # Shared utilities (config, llm runtime, LiteLLM client, data, identifiability)
│   │   │       ├── observation_semantics.py  # Canonical support-kind / summary-operator / anchor-policy semantics
│   │   │       └── medical_semantics_fixture.py  # Shared MEDICAL_SEMANTICS fixture loading + ordered Stage 2 comparison reports
│   │   ├── benchmarks/            # Inference method benchmarks (parameter recovery)
│   │   ├── evals/                 # Inspect AI evals for active Stage 1a/1b/2 surfaces + filesystem-discovered questions
│   │   │   └── multi_model/eval_medical_semantics_orchestrator.py  # Judge-ranked orchestrator eval on the fixed MEDICAL_SEMANTICS fixture
│   │   ├── notebooks/             # Showcase notebooks
│   │   ├── tests/                 # pytest tests
│   │   │   ├── test_pipeline.py   # Flow orchestration and replay contract coverage
│   │   │   ├── test_medical_semantics_fixture.py  # Fixture contract for MEDICAL_SEMANTICS stage 2 observation rows
│   │   │   ├── test_stage4b_parametric_id.py  # Stage 4b first-pass Rao-Blackwellization payload gating
│   │   │   └── test_stage2_extract.py  # Stage 2 worker collection and progress logging coverage
│   │   └── tools/                 # Narrow retained utilities (eval log reader, GPU smoke, LaTeX renderer)
│   └── web/                       # Next.js frontend
│       ├── scripts/               # Install-time asset helpers (for example Perspective WASM/CSS copying)
│       └── src/
│           ├── app/               # Next.js app router pages + userId-keyed analysis/API routes (+ colocated route tests)
│           │   ├── api/analysis/  # Server-side analysis manifest, stage log-source resolution, and resumed-run hydration
│           │   ├── api/results/[userId]/[stage]/dataframe/  # Stage parquet download endpoint for full-data exploration
│           │   └── explore/[userId]/[stage]/  # Standalone Perspective-powered full-data explorer
│           ├── components/        # React components (stages, charts, DAG, pipeline)
│           │   ├── pipeline/stage-contents/stage-4b-content.test.ts  # Verifies the Stage 4b surface renders T-rule payloads
│           │   └── stages/parametric-id/  # Stage 4b surfaces such as inference-structure-card.tsx and t-rule-card.tsx
│           └── lib/               # API clients, hooks, user-id helpers, types, utilities
│               ├── api/analysis.ts  # Shared typed contracts + client helpers for sessions, replay, and analysis manifests
│               ├── hooks/         # Prefect progress state, shared websocket transport, and stage log telemetry
│               │   ├── use-prefect-socket.ts  # Shared Prefect WebSocket auth/subscription hook used by events and logs
│               │   └── use-stage-logs.test.ts  # Covers Prefect log backlog/bootstrap routing and stream filter contracts
│               └── root-flow-runs.ts  # Shared root Prefect run lineage helpers used by sessions + manifest hydration
├── packages/
│   ├── api-types/                 # Generated TypeScript types + exported schema snapshots
│   └── typescript-config/         # Shared TS config
├── data/                          # Root data workspace shared by web + pipeline
│   ├── <USER_ID>/                 # User workspace: input/, query.txt, run/
│   ├── DEFAULT/                   # Tracked mock fixture user workspace
│   ├── DOCTOLIB/                  # Tracked mock fixture user workspace
│   ├── MEDICAL_SEMANTICS/         # Tracked stage 0-2 medical archive fixture workspace
│   │   ├── expected-stage2-raw-data.csv  # Expected full Stage 2 raw observation rows for fixture regression
│   │   ├── expected-stage2-model-data.csv  # Expected full Stage 2 model-ready rows for fixture regression
│   │   └── README.md              # Human-readable Stage 2 artifact shape and observation semantics contract
│   ├── GOLDEN/                    # Golden dataset submodule
│   ├── processed/                 # Preprocessed chunk files for eval/manual tools (gitignored)
│   ├── sessions.seed.json         # Tracked fixture run metadata keyed by user ID
│   └── sessions.json              # Runtime run metadata keyed by user ID (gitignored)
├── docs/                          # Project documentation (see docs/index.md)
│   ├── modeling/                  # Theoretical foundations + SSM compilation pipeline
│   ├── guides/                    # Practical usage: dev setup, data workflow, evals, codegen, integration testing
│   └── literature.md              # Consolidated bibliography
└── scratchpad/                    # Temporary work files (gitignored)
```

Web routes and persisted workspaces are keyed by `userId` / `user_id`:
- `apps/web/src/app/analysis/[userId]/` and `apps/web/src/app/api/results/[userId]/[stage]/`
- `apps/web/src/lib/user-id.ts` generates anonymous user IDs before auth
