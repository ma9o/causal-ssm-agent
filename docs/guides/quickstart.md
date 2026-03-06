# Quickstart

## Running the Pipeline

**Option 1: Direct execution**

First, create a query file in `data/queries/` (e.g., `smoking-cancer.txt`):
```
What is the causal effect of smoking on lung cancer risk,
controlling for age and genetic predisposition?
```

Then run:
```bash
uv run python -c "
from causal_ssm_agent.flows.pipeline import causal_inference_pipeline

# Uses latest preprocessed file automatically
causal_inference_pipeline(
    query_file='smoking-cancer',
    target_effects=['smoking -> cancer'],
)

# Or specify input file explicitly
causal_inference_pipeline(
    query_file='smoking-cancer',
    target_effects=['smoking -> cancer'],
    input_file='export_20241208_153022.txt',
)
"
```

**Option 2: Serve with UI**
```bash
# Terminal 1: Start Prefect server
uv run prefect server start

# Terminal 2: Serve the flow
uv run python -m causal_ssm_agent.flows.pipeline
```

Then open http://localhost:4200 to trigger runs with custom parameters.
