# Documentation Index

This index is the top-level map of the documentation for readers and coding agents.

## Structure

```text
docs/
├── index.md
├── pipeline.md
├── pipeline/
│   ├── 00-ingestion.md
│   ├── 01a-latent-model.md
│   ├── 01b-measurement-identifiability.md
│   ├── 02-indicator-extraction.md
│   ├── 03-extraction-validation.md
│   ├── 04-model-specification-priors.md
│   ├── 04b-parametric-identifiability.md
│   ├── 05a-svi-preflight.md
│   ├── 05b-inference-diagnostics.md
│   └── 06-intervention-analysis.md
├── primitives/
│   ├── latent-model/
│   │   ├── index.md
│   │   ├── constructs-and-edges.md
│   │   ├── temporal-semantics.md
│   │   └── assumptions.md
│   ├── measurement-model/
│   │   ├── index.md
│   │   ├── indicators.md
│   │   ├── windows-and-aggregation.md
│   │   └── assumptions.md
│   ├── causal-spec/
│   │   ├── index.md
│   │   ├── identifiability.md
│   │   └── handoff-contract.md
│   └── model-spec/
│       ├── index.md
│       ├── functional-specification.md
│       ├── parameters-likelihoods-and-priors.md
│       └── prior-elicitation.md
├── concepts/
│   ├── artifact-index.md
│   ├── causal-modeling-terminology.md
│   ├── pipeline-dimensions.md
│   ├── assumptions.md
│   └── scope-and-timescales.md
├── model-runtime/
│   ├── handoff-map.md
│   ├── compilation.md
│   ├── estimation.md
│   └── inference-routing.md
├── runtime/
│   ├── execution-and-replay.md
│   └── persistence-and-exposure.md
└── guides/
    ├── dev_setup.md
    ├── data_contract.md
    ├── data_workflow.md
    ├── running_evals.md
    ├── codegen.md
    └── agentic_integration_testing.md
```

References are kept in the docs that use them rather than in a shared bibliography file.

## Navigate by Primitive

- [`LatentModel`](primitives/latent-model/index.md): construct ontology, edge legality, and lag semantics
- [`MeasurementModel`](primitives/measurement-model/index.md): indicators, extraction semantics, windows, aggregation, and `model_clock`
- [`CausalSpec`](primitives/causal-spec/index.md): identifiability and the Stage 1b handoff contract
- [`ModelSpec`](primitives/model-spec/index.md): functional specification, parameter roles, likelihoods, and prior elicitation

## Quick Links by Task

**Understanding the pipeline:**

- Start with `pipeline.md` for the ordered stage map.
- Use `concepts/artifact-index.md` only if you know an artifact name but do not know which stage owns it.
- Start with `concepts/pipeline-dimensions.md` for the cross-cutting map.
- Then open the relevant file under `pipeline/` for stage-specific detail.

**Understanding the modeling approach:**

- Start with `concepts/causal-modeling-terminology.md` if you need the naming conventions.
- Use `concepts/scope-and-timescales.md` to route temporal questions to the owning primitive.
- Check `concepts/assumptions.md` to see which primitive owns each numbered assumption.
- Then open the relevant primitive section under `primitives/`.
- Use `model-runtime/handoff-map.md` for the Stage 4 -> Stage 6 chain.
- Use `model-runtime/compilation.md`, `model-runtime/estimation.md`, and `model-runtime/inference-routing.md` for the runtime path.

**Running the system:**

- `guides/dev_setup.md` for bootstrapping from a fresh clone
- `guides/data_contract.md` for practitioner-facing dataset requirements
- `guides/data_workflow.md` for workspace layout and data placement
- `guides/running_evals.md` for evaluation
- `guides/codegen.md` for TypeScript type generation from Python schemas
- `guides/agentic_integration_testing.md` for end-to-end integration testing

**Jump in by question:**

- "What is `CausalSpec`, `ModelSpec`, or the fitted artifact?" -> use `concepts/artifact-index.md` to find the owner stage, then open that stage doc.
- "What objects move through the pipeline?" -> `concepts/pipeline-dimensions.md`
- "What does one of the four domain primitives mean?" -> open the matching page under `primitives/`
- "What should my uploaded data look like?" -> `guides/data_contract.md`, then `guides/data_workflow.md`
- "What does each stage do?" -> `pipeline.md`, then the relevant file under `pipeline/`
- "Why do the docs say `latent model`, `topological structure`, and `functional specification` instead of `structural model`?" -> `concepts/causal-modeling-terminology.md`
- "How does time work across extraction, fitting, and interventions?" -> `concepts/pipeline-dimensions.md`, then `concepts/scope-and-timescales.md` and `model-runtime/estimation.md`
- "What gets validated where?" -> `concepts/pipeline-dimensions.md`, then `pipeline.md` and `primitives/model-spec/functional-specification.md`
- "How do resume, replay, overrides, and gates work?" -> `runtime/execution-and-replay.md`
- "What gets persisted, restored, or exposed to the web?" -> `runtime/persistence-and-exposure.md`, then `pipeline.md`

**Benchmarks:**

- `../apps/data-pipeline/benchmarks/results.md` for inference method parameter recovery results
