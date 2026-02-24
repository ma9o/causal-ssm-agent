# Documentation Index

This index helps coding agents navigate the documentation structure.

## Structure

```
docs/
├── index.md           # This file
├── modeling/          # Theoretical foundations (scope, assumptions, theory, estimation)
├── reference/         # Technical specs (schemas, pipeline, mplus-parity)
├── guides/            # Practical usage (quickstart, data, evals, prompting)
└── papers/            # Reference PDFs
```

## Quick Links by Task

**Understanding the modeling approach:**
- Start with `modeling/scope.md` for construct taxonomy, ontology, temporal granularity, cross-timescale rules, and what's in/out of scope
- Check `modeling/assumptions.md` for specific technical assumptions (A1-A9)
- See `modeling/theory.md` for design rationale (per-effect identification, confounders, measurement-to-causal bridge)
- See `modeling/estimation.md` for the estimation pipeline (CT-SDE, discretization, SSMSpec, likelihood backends, counterfactual inference)
- See `modeling/inference-strategies.md` for inference method selection (SVI, NUTS, NUTS-DA, Hess-MC², PGAS, Tempered SMC, etc.)
- See `modeling/functional_spec.md` for Stage 4 model specification (rule-based constraints, LLM prior elicitation, parametric ID)

**Implementing or modifying:**
- `reference/schema.md` for Pydantic models and validation
- `reference/pipeline.md` for stage-by-stage breakdown
- `reference/mplus-parity.md` for feature comparison with Asparouhov et al. (2017)

**Running the system:**
- `guides/quickstart.md` for installation and first run
- `guides/data_workflow.md` for data preprocessing
- `guides/running_evals.md` for evaluation

**LLM usage:**
- `guides/prompting.md` for LLM prompting best practices
