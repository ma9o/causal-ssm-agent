1. Every time you commit make sure to split commits atomically, avoiding clumping multiple increments into a single one.

2. Every time you make a change to the file structure make sure to report it under Structure in the README

3. Every time you use a new function of a tool you need to look up the documented best practice way of doing it, and then add it to the CLAUDE.md instructions under that tool section:

------

# polars
Docs: https://docs.pola.rs/api/python/stable/reference/index.html

# uv
Docs: https://docs.astral.sh/uv/

# Prefect for pipeline orchestration
Docs: https://docs.prefect.io/v3/get-started

## Best Practices (v3)

### Tasks
- Use `retries` and `retry_delay_seconds` for fault tolerance
- Cache expensive ops with `cache_key_fn` and `cache_policy` (INPUTS, TASK_SOURCE, etc.)
- Set `timeout_seconds` to prevent runaway tasks
- Use `log_prints=True` to capture print statements as logs
- Name tasks explicitly with `name` parameter
- Use `task_run_name` with f-string patterns for observability: `@task(task_run_name="process-{chunk_id}")`

### Flows
- Nest flows for logical grouping - child flows appear in UI
- Use type hints on parameters for validation
- Set `timeout_seconds` on long-running flows
- Use `log_prints=True` on flows too

### Concurrency
- Use `task.map(items)` for parallel execution over iterables
- Native Python async/await supported for concurrent I/O

### Deployments
- Use `flow.serve()` to create deployment and start listener process
- Parameters with type hints get UI input forms via OpenAPI schema
- Set default parameters in `serve()`, users can override in UI
- Use `str | None = None` for optional params with smart defaults

```python
@flow(log_prints=True)
def pipeline(required_param: str, optional_file: str | None = None):
    ...

if __name__ == "__main__":
    pipeline.serve(
        name="my-deployment",
        tags=["tag1"],
        parameters={"required_param": "default"},  # defaults
    )
```

### Patterns
```python
@task(retries=3, retry_delay_seconds=10, cache_key_fn=task_input_hash)
def fetch_data(url: str):
    ...

@flow(log_prints=True, timeout_seconds=3600)
def pipeline():
    results = fetch_data.map(urls)  # parallel
```

# AISI's inspect agent framework
Docs: https://inspect.aisi.org.uk/

## Best Practices

### Model Interaction
- Use `get_model()` to get the configured model instance
- Use `model.generate(messages)` for single-turn generation
- Use `model.generate_loop(messages, tools=...)` for multi-turn with tool calling

### Messages
- `ChatMessageSystem(content=...)` for system prompts
- `ChatMessageUser(content=...)` for user messages
- Messages are lists passed to generate()

### Structured Output
- Use Pydantic models for response schemas
- Parse JSON from response, handle markdown code blocks
- Validate with `Model.model_validate(data)`

### Patterns
```python
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model

async def my_agent(prompt: str) -> dict:
    model = get_model()
    messages = [
        ChatMessageSystem(content="You are a helpful assistant."),
        ChatMessageUser(content=prompt),
    ]
    response = await model.generate(messages)
    return parse_json(response.completion)
```

### Sync Wrapper
```python
def sync_wrapper(prompt: str) -> dict:
    import asyncio
    return asyncio.run(my_agent(prompt))
```

# PyMC 
Docs: https://www.pymc.io/welcome.html

# DoWhy
Docs: https://www.pywhy.org/dowhy/v0.14/

## Best Practices

### Graph Format
- Use NetworkX DiGraph as primary format (preferred by DoWhy)
- Create from edge list: `nx.DiGraph([('X', 'Y'), ('Y', 'Z')])`
- Can also use GML string format for CausalModel

### CausalModel
```python
from dowhy import CausalModel
import networkx as nx

# From NetworkX graph
graph = nx.DiGraph([('treatment', 'outcome'), ('confounder', 'treatment'), ('confounder', 'outcome')])
model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='outcome',
    graph=graph,  # or GML string
)
```

### Identification & Estimation
```python
# Identify causal effect
identified = model.identify_effect()

# Estimate
estimate = model.estimate_effect(
    identified,
    method_name="backdoor.linear_regression"
)

# Refute (sensitivity analysis)
refutation = model.refute_estimate(
    identified, estimate,
    method_name="random_common_cause"
)
```

# NetworkX
Docs: https://networkx.org/documentation/stable/

## Best Practices
- Use `DiGraph` for causal DAGs (directed edges)
- Create from edge list: `nx.DiGraph([(cause, effect), ...])`
- Check for cycles: `nx.is_directed_acyclic_graph(G)`
- Add node attributes: `G.add_node(name, dtype='continuous', ...)`

# ArViz 
Docs: https://python.arviz.org/en/stable/index.html