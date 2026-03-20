# Documentation Index

This index helps coding agents navigate the documentation structure.

## Structure

```
docs/
├── index.md              # This file
├── pipeline_stages.md    # Complete pipeline stage reference (inputs, outputs, logic for stages 0–6)
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
- Start with `pipeline_stages.md` for a complete reference of all 10 stages: inputs, outputs, internal logic, gates, and resume behavior

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

**Benchmarks:**
- `../apps/data-pipeline/benchmarks/results.md` for inference method parameter recovery results
