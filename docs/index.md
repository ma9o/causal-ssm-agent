# Documentation Index

This index helps coding agents navigate the documentation structure.

## Structure

```
docs/
├── index.md           # This file
├── modeling/          # Theoretical foundations (scope, assumptions, estimation)
├── guides/            # Practical usage (data workflow, evals, codegen)
└── literature.md      # Reference papers (links + summaries)
```

## Quick Links by Task

**Understanding the modeling approach:**
- Start with `modeling/scope.md` for construct taxonomy, ontology, temporal granularity, cross-timescale rules, and what's in/out of scope
- Check `modeling/assumptions.md` for specific technical assumptions (A1-A9)
- See `modeling/estimation.md` for the estimation pipeline (CT-SDE, discretization, likelihood backends, counterfactual inference)
- See `modeling/inference-strategies.md` for inference routing (three orthogonal axes, structural decision tree, 9 methods)
- See `modeling/functional_spec.md` for Stage 4 model specification (rule-based constraints, LLM prior elicitation, parametric ID)

**Running the system:**
- `guides/data_workflow.md` for data preprocessing
- `guides/running_evals.md` for evaluation
