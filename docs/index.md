# Documentation Index

This index is the top-level map of the documentation for readers and coding agents.

## Structure

```
docs/
├── index.md              # This file
├── architecture/
│   ├── artifact_glossary.md    # Short definitions for the core pipeline objects (`LatentModel`, `CausalSpec`, `ModelSpec`, fitted artifact, etc.)
│   ├── pipeline_dimensions.md  # Cross-cutting pipeline map: artifact lineage, temporal semantics, validation surfaces, and persistence/runtime semantics
│   └── runtime_semantics.md    # Execution DAG, restore/recompute behavior, replay/override semantics, and persistence surfaces
├── pipeline_stages.md    # Complete pipeline stage reference (inputs, outputs, logic for stage-0 → stage-6, including stage-1a/1b, stage-4b, and stage-5a/5b)
├── modeling/
│   ├── scope.md          # Construct taxonomy, temporal granularity, what's in/out of scope
│   ├── assumptions.md    # Core technical assumptions (A1-A9)
│   ├── estimation.md     # CT-SDE pipeline, discretization, likelihood backends, counterfactuals
│   ├── inference-strategies.md  # Inference routing (three axes, structural decision tree, 9 methods)
│   ├── functional_spec.md      # Stage 4 model specification (rules, LLM prior elicitation, parametric ID)
│   └── compilation.md    # SSM compilation pipeline (ModelSpec → SSMModel data flow)
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
- Start with `architecture/artifact_glossary.md` if you need the object vocabulary first
- Start with `architecture/pipeline_dimensions.md` for the cross-cutting map: artifact lineage, temporal semantics, execution modality, assurance surfaces, and persistence/runtime semantics
- Then read `architecture/runtime_semantics.md` for execution DAG, restore/recompute behavior, and replay/override semantics
- Then read `pipeline_stages.md` for the complete stage-by-stage reference: inputs, outputs, internal logic, gates, and resume behavior

**Understanding the modeling approach:**
- Start with `modeling/scope.md` for construct taxonomy, ontology, temporal granularity, cross-timescale rules, and what's in/out of scope
- Check `modeling/assumptions.md` for specific technical assumptions (A1-A9)
- See `modeling/estimation.md` for the estimation pipeline (CT-SDE, discretization, likelihood backends, counterfactual inference)
- See `modeling/inference-strategies.md` for inference routing (three orthogonal axes, structural decision tree, 9 methods)
- See `modeling/functional_spec.md` for Stage 4 model specification (rule-based constraints, LLM prior elicitation, parametric ID)
- See `modeling/compilation.md` for the SSM compilation pipeline (ModelSpec → SSMSpec → SSMModel)

**Running the system:**
- `guides/dev_setup.md` for bootstrapping from a fresh clone
- `guides/data_workflow.md` for data preprocessing
- `guides/running_evals.md` for evaluation
- `guides/codegen.md` for TypeScript type generation from Python schemas
- `guides/agentic_integration_testing.md` for end-to-end integration testing

**Jump in by question:**
- "What is `CausalSpec`, `ModelSpec`, or the fitted artifact?" -> `architecture/artifact_glossary.md`
- "What objects move through the pipeline?" -> `architecture/pipeline_dimensions.md`
- "What does each stage do?" -> `pipeline_stages.md`
- "How does time work across extraction, fitting, and interventions?" -> `architecture/pipeline_dimensions.md`, then `modeling/scope.md` and `modeling/estimation.md`
- "What gets validated where?" -> `architecture/pipeline_dimensions.md`, then `pipeline_stages.md` and `modeling/functional_spec.md`
- "How do resume, replay, overrides, and gates work?" -> `architecture/runtime_semantics.md`
- "What gets persisted, restored, or exposed to the web?" -> `architecture/runtime_semantics.md`, then `pipeline_stages.md`

**Benchmarks:**
- `../apps/data-pipeline/benchmarks/results.md` for inference method parameter recovery results
