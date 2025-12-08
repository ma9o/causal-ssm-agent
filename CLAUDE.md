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

# PyMC 
Docs: https://www.pymc.io/welcome.html

# DoWhy
Docs: https://www.pywhy.org/dowhy/v0.14/

# ArViz 
Docs: https://python.arviz.org/en/stable/index.html