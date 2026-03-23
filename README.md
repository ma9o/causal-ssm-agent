# causal-ssm-agent

This project explores an end-to-end, LLM-orchestrated framework for causal inference over long-context, multi-source data (e.g. large document collections or aggregated web search). An "orchestrator" LLM proposes candidate variables, time granularities, and a causal DAG; "worker" LLMs then populate those dimensions at scale, after which we use y0 for identifiability checks (via Pearl's ID algorithm), and NumPyro for Bayesian state-space model estimation with LLM-elicited priors. The goal is to build a system that not only estimates causal effects and counterfactuals from messy, high-dimensional evidence, but also knows when to trust those numeric estimates and when to fall back to purely structural, qualitative reasoning.

Current implementation status:
- The pipeline and web app are implemented end to end, including interactive refinement for stages 1a, 1b, 4, and 6.
- The estimation stack currently targets single-subject or already-aggregated time series. Hierarchical multi-subject panel modeling is not implemented yet.

**Key Innovation: Continuous-Time Modeling**

Unlike traditional discrete-time approaches that require upfront aggregation, this framework uses continuous-time state-space modeling which:
- Handles irregularly-spaced observations natively via Kalman/particle filtering
- Avoids information loss from pre-aggregation
- Models dynamics via stochastic differential equations
- Keeps irregular-time dynamics explicit rather than forcing fixed-width bins up front
- Computes counterfactual effects via do-operator on CT steady states

## Start Here

- Curious readers: start with [`docs/index.md`](docs/index.md), [`docs/concepts/artifact-glossary.md`](docs/concepts/artifact-glossary.md), [`docs/concepts/pipeline-dimensions.md`](docs/concepts/pipeline-dimensions.md), and [`docs/pipeline.md`](docs/pipeline.md)
- Causal inference practitioners: then read [`docs/concepts/scope-and-timescales.md`](docs/concepts/scope-and-timescales.md), [`docs/concepts/assumptions.md`](docs/concepts/assumptions.md), and [`docs/model-runtime/estimation.md`](docs/model-runtime/estimation.md)
- Software engineers: start with [`docs/guides/dev_setup.md`](docs/guides/dev_setup.md), [`docs/guides/data_workflow.md`](docs/guides/data_workflow.md), and [`docs/guides/codegen.md`](docs/guides/codegen.md)

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

- **[Pipeline](docs/pipeline.md)** - Ordered stage map plus per-stage reference files
- **[Concepts](docs/concepts/)** - Cross-cutting domain concepts: artifacts, assumptions, temporal semantics, and scope
- **[Model Runtime](docs/model-runtime/)** - Stage 4 to Stage 6 handoff path: functional specification, compilation, estimation, and inference routing
- **[Runtime](docs/runtime/)** - Replay, restore, persistence, and web/internal exposure boundaries
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
│   │   │   ├── flows/             # Prefect pipeline stages (stage-0 → stage-6, including stage-1a/1b, stage-4b, and stage-5a/5b) + replay/resume orchestration
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
│           ├── app/               # Next.js app router pages + workspace-keyed analysis/API routes (+ colocated route tests)
│           │   ├── api/analysis/  # Server-side analysis manifest, stage log-source resolution, and resumed-run hydration
│           │   ├── api/refine/    # Shared interactive-LLM route handlers for stage refinement and terminal Stage 6 persistence
│           │   ├── api/results/[workspaceId]/[stage]/dataframe/  # Stage parquet download endpoint for full-data exploration
│           │   └── explore/[workspaceId]/[stage]/  # Standalone Perspective-powered full-data explorer
│           ├── components/        # React components (stages, charts, DAG, pipeline)
│           │   ├── pipeline/stage-contents/stage-4b-content.test.ts  # Verifies the Stage 4b surface renders T-rule payloads
│           │   ├── stages/inference/  # Inference diagnostics and treatment-effect ranking surfaces
│           │   └── stages/parametric-id/  # Stage 4b surfaces such as inference-structure-card.tsx and t-rule-card.tsx
│           └── lib/               # API clients, hooks, workspace-id helpers, types, utilities
│               ├── api/analysis.ts  # Shared typed contracts + client helpers for sessions, replay, and analysis manifests
│               ├── hooks/         # Prefect progress state, shared websocket transport, and stage log telemetry
│               │   ├── use-prefect-socket.ts  # Shared Prefect WebSocket auth/subscription hook used by events and logs
│               │   └── use-stage-logs.test.ts  # Covers Prefect log backlog/bootstrap routing and stream filter contracts
│               └── root-flow-runs.ts  # Shared root Prefect run lineage helpers used by sessions + manifest hydration
├── packages/
│   ├── api-types/                 # Generated TypeScript types + exported schema snapshots
│   └── typescript-config/         # Shared TS config
├── data/                          # Root data workspace shared by web + pipeline
│   ├── <WORKSPACE_ID>/            # Workspace: access.json, input/, query.txt, session.json, run/
│   ├── DEFAULT/                   # Tracked mock fixture workspace
│   ├── DOCTOLIB/                  # Tracked mock fixture workspace
│   ├── MEDICAL_SEMANTICS/         # Tracked stage 0-2 medical archive fixture workspace
│   │   ├── expected-stage2-raw-data.csv  # Expected full Stage 2 raw observation rows for fixture regression
│   │   ├── expected-stage2-model-data.csv  # Expected full Stage 2 model-ready rows for fixture regression
│   │   └── README.md              # Human-readable Stage 2 artifact shape and observation semantics contract
│   ├── GOLDEN/                    # Golden dataset submodule
│   ├── processed/                 # Preprocessed chunk files for eval/manual tools (gitignored)
│   ├── <WORKSPACE_ID>/access.json # Hashed workspace resume-code metadata stored separately from session lineage
│   └── <WORKSPACE_ID>/session.json # Per-workspace run lineage metadata persisted alongside query.txt
├── docs/                          # Project documentation (see docs/index.md)
│   ├── pipeline.md                # Ordered stage map with links to the per-stage docs
│   ├── pipeline/                  # One file per stage (0, 1a, 1b, 2, 3, 4, 4b, 5a, 5b, 6)
│   ├── concepts/                  # Cross-cutting domain concepts: artifacts, assumptions, timescales, and scope
│   ├── model-runtime/             # Stage 4 -> 6 handoff path: functional spec, compilation, estimation, inference routing
│   ├── runtime/                   # Replay, restore, persistence, and web/internal exposure
│   ├── guides/                    # Practical usage: dev setup, data workflow, evals, codegen, integration testing
│   └── literature.md              # Consolidated bibliography
└── scratchpad/                    # Temporary work files (gitignored)
```

The same workspace is addressed as `workspaceId` in the web app and `workspace_id` in pipeline/Prefect payloads:
- `apps/web/src/app/analysis/[workspaceId]/` and `apps/web/src/app/api/results/[workspaceId]/[stage]/`
- `apps/web/src/lib/workspace-id.ts` generates anonymous workspace IDs; access control is handled separately via workspace resume codes
