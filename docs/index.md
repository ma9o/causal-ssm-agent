# Documentation Index

This index is the top-level map of the documentation for readers and coding agents.

## Structure

```
docs/
├── index.md              # This file
├── pipeline.md           # Ordered pipeline map and links to each stage file
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
├── concepts/
│   ├── artifact-index.md       # Artifact -> owner-stage lookup table; definitions live in the stage docs
│   ├── pipeline-dimensions.md  # Cross-cutting pipeline map: artifact lineage, temporal semantics, validation surfaces, and control-flow semantics
│   ├── assumptions.md          # Core technical assumptions (A1-A9)
│   └── scope-and-timescales.md # Construct taxonomy, temporal granularity, cross-timescale rules, and scope boundaries
├── model-runtime/
│   ├── handoff-map.md          # Stage 4 -> Stage 6 artifact handoff chain
│   ├── functional-specification.md
│   ├── compilation.md
│   ├── estimation.md
│   └── inference-routing.md
├── runtime/
│   ├── execution-and-replay.md
│   └── persistence-and-exposure.md
├── guides/
│   ├── dev_setup.md      # Local development setup (bootstrapping from a fresh clone)
│   ├── data_workflow.md  # Data organization for users and evals
│   ├── running_evals.md  # Inspect AI evaluation framework
│   ├── codegen.md        # Python → TypeScript type generation
│   └── agentic_integration_testing.md  # E2E integration testing with browser automation
├── literature.md         # Consolidated bibliography (all papers referenced across docs)
```

## Quick Links by Task

**Understanding the pipeline:**
- Start with `pipeline.md` for the ordered stage map
- Use `concepts/artifact-index.md` only if you know an artifact name but do not know which stage owns it
- Start with `concepts/pipeline-dimensions.md` for the cross-cutting map: artifact lineage, temporal semantics, execution modality, assurance surfaces, and control-flow semantics
- Then open the relevant file under `pipeline/` for stage-specific detail

**Understanding the modeling approach:**
- Start with `concepts/scope-and-timescales.md` for construct taxonomy, ontology, temporal granularity, cross-timescale rules, and what's in/out of scope
- Check `concepts/assumptions.md` for specific technical assumptions (A1-A9)
- See `model-runtime/handoff-map.md` for the Stage 4 -> Stage 6 chain
- See `model-runtime/functional-specification.md`, `model-runtime/compilation.md`, `model-runtime/estimation.md`, and `model-runtime/inference-routing.md` for the downstream model-runtime path

**Running the system:**
- `guides/dev_setup.md` for bootstrapping from a fresh clone
- `guides/data_workflow.md` for data preprocessing
- `guides/running_evals.md` for evaluation
- `guides/codegen.md` for TypeScript type generation from Python schemas
- `guides/agentic_integration_testing.md` for end-to-end integration testing

**Jump in by question:**
- "What is `CausalSpec`, `ModelSpec`, or the fitted artifact?" -> use `concepts/artifact-index.md` to find the owner stage, then open that stage doc
- "What objects move through the pipeline?" -> `concepts/pipeline-dimensions.md`
- "What does each stage do?" -> `pipeline.md`, then the relevant file under `pipeline/`
- "How does time work across extraction, fitting, and interventions?" -> `concepts/pipeline-dimensions.md`, then `concepts/scope-and-timescales.md` and `model-runtime/estimation.md`
- "What gets validated where?" -> `concepts/pipeline-dimensions.md`, then `pipeline.md` and `model-runtime/functional-specification.md`
- "How do resume, replay, overrides, and gates work?" -> `runtime/execution-and-replay.md`
- "What gets persisted, restored, or exposed to the web?" -> `runtime/persistence-and-exposure.md`, then `pipeline.md`

**Benchmarks:**
- `../apps/data-pipeline/benchmarks/results.md` for inference method parameter recovery results
