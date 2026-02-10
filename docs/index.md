# Documentation Index

This index helps coding agents navigate the documentation structure.

## Structure

```
docs/
├── index.md           # This file
├── modeling/          # Theoretical foundations (DSEM, assumptions, scope)
├── reference/         # Technical specs (schemas, pipeline)
└── guides/            # Practical usage (quickstart, data, evals)
```

## Quick Links by Task

**Understanding the modeling approach:**
- Start with `modeling/scope.md` for construct taxonomy, ontology, and what's in/out of scope
- Read `modeling/dsem_overview.md` for temporal granularity and cross-timescale rules
- Check `modeling/assumptions.md` for specific technical assumptions (A1-A9)
- See `modeling/theory.md` for theoretical foundations (identification locality, correlated errors, SEM/Pearl bridge, Bayesian unification)
- See `modeling/estimation.md` for the estimation pipeline (CT-SDE, discretization, SVI/NUTS/Hess-MC2/PGAS/Tempered-SMC inference, cuthbert integration)

**Implementing or modifying:**
- `reference/schema.md` for Pydantic models and validation
- `reference/pipeline.md` for stage-by-stage breakdown
- `reference/dsem-parity.md` for feature comparison with Asparouhov et al. (2017) Mplus DSEM

**Running the system:**
- `guides/quickstart.md` for installation and first run
- `guides/data_workflow.md` for data preprocessing
- `guides/running_evals.md` for evaluation

**Library patterns (extracted from AGENTS.md):**
- `guides/prefect.md` for Prefect v3 task/flow patterns
- `guides/inspect.md` for AISI inspect eval framework
- `guides/exa.md` for Exa literature search
