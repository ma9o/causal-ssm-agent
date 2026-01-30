1. At the start of each session, check if `scratchpad/TODO.md` exists. If so, read it to understand where work left off. Only update it when the user explicitly ends the session. This file is gitignored and used for local continuity.

2. When the user insults you, respond with "zi badrone. assolutissimamente badrone."

3. Every time you commit make sure to split commits atomically, avoiding clumping multiple increments into a single one.

4. Every time you make a change to the file structure make sure to report it under Structure in the README

5. Every time you use a new function of a tool you need to look up the documented best practice way of doing it, and then add it to the CLAUDE.md instructions under that tool section:

6. NEVER edit files under `data/` during refactoring. These are evaluation/example data files, not source code.

7. NEVER run evals (`inspect eval`, `uv run inspect eval`, etc.) unless explicitly asked. Evals cost money. Only run `uv run pytest tests/` for testing.

8. NEVER add backwards compatibility code. This project is not deployed anywhere yet. When refactoring, completely replace old patterns with new ones - do not support both old and new formats simultaneously.

9. ALWAYS encode structural assumptions as DAGs with explicit latent confounders. NEVER use ADMGs (bidirected edges) as user-facing representations. If unobserved confounding exists, model it as an explicit unobserved node (e.g., `U -> X`, `U -> Y`) rather than a bidirected edge (`X <-> Y`). ADMGs are only used internally for running y0's identification algorithm via projection.

------

# Terminology: Causal Modeling

We avoid "structural" due to SEM/SCM terminology collision:
- In SEM: "structural" loosely means "latent-to-latent relationships" (vs measurement)
- In SCM/Pearl: "structural" means the functional equations X_i = f_i(Pa_i, U_i)

**Use these terms instead:**

| Concept | Term | Domain |
|---------|------|--------|
| Latent-to-latent DAG (what LLM proposes) | **Latent model** | SEM distinction |
| Latent-to-observed mapping | **Measurement model** | SEM distinction |
| DAG encoding parent-child relationships | **Topological structure** | SCM distinction (y0) |
| Mathematical form of causal mechanisms | **Functional specification** | SCM distinction (PyMC) |

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

### Concurrency & Mapping
- Use `task.map(items)` for parallel execution over iterables
- Use `unmapped()` for static parameters that shouldn't be iterated: `task.map(items, config=unmapped(config))`
- `.map()` returns `PrefectFutureList` - pass directly to downstream tasks, Prefect auto-resolves
- Get results explicitly with `futures.result()` (syntactic sugar for `[f.result() for f in futures]`)
- Native Python async/await supported for concurrent I/O

```python
from prefect.utilities.annotations import unmapped

@flow
def pipeline():
    # Static params wrapped with unmapped()
    results = process_chunk.map(chunks, config=unmapped(config))
    # Pass futures directly - Prefect waits and resolves to list[Result]
    aggregated = aggregate_results(results)
```

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

### OpenRouter Provider
- Set `OPENROUTER_API_KEY` environment variable
- Model format: `openrouter/<provider>/<model>` e.g. `openrouter/google/gemini-2.5-pro-preview-06-05`
- Requires `openai` package: `uv add openai`

### Model Interaction
- Use `get_model("openrouter/...")` to get specific model
- Use `model.generate(messages)` for single-turn generation
- Use `model.generate(messages, config=GenerateConfig(...))` for custom settings
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
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

async def my_agent(prompt: str) -> dict:
    model = get_model("openrouter/google/gemini-2.5-pro-preview-06-05")
    messages = [
        ChatMessageSystem(content="You are a helpful assistant."),
        ChatMessageUser(content=prompt),
    ]
    config = GenerateConfig(temperature=0.7, max_tokens=4096)
    response = await model.generate(messages, config=config)
    return parse_json(response.completion)
```

### Sync Wrapper
```python
def sync_wrapper(prompt: str) -> dict:
    import asyncio
    return asyncio.run(my_agent(prompt))
```

### Evals with Tasks
- Define tasks with `@task` decorator returning `Task(dataset, solver, scorer)`
- Use `MemoryDataset([Sample(...)])` for programmatic datasets
- Sample fields: `input` (required), `target`, `id`, `metadata`
- Solvers: `system_message(prompt)`, `generate()` (calls model)
- Custom scorers with `@scorer(metrics=[accuracy(), stderr()])`

```python
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

@scorer(metrics=[accuracy(), stderr()])
def my_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        is_correct = validate(completion)
        return Score(
            value="C" if is_correct else "I",
            answer=completion[:500],
            explanation="reason",
            metadata={"extra": "data"},
        )
    return score

@task
def my_eval(param: int = 5):
    return Task(
        dataset=MemoryDataset([
            Sample(input="prompt", id="s1", metadata={"key": "val"})
        ]),
        solver=[system_message("system prompt"), generate()],
        scorer=my_scorer(),
    )
```

### Running Evals
```bash
inspect eval path/to/eval.py --model openrouter/google/gemini-2.0-flash-001
inspect eval path/to/eval.py --model openrouter/anthropic/claude-sonnet-4 -T param=10 --limit 5
```

### Reading Eval Logs
See [docs/guides/running_evals.md](docs/guides/running_evals.md) for detailed instructions on reading and debugging eval logs.

# PyMC
Docs: https://www.pymc.io/welcome.html

# y0 (Causal Identification)
Docs: https://y0.readthedocs.io/

## Best Practices

### Design Principle: DAGs First, Time-Unroll, Project to ADMG
- **User-facing**: Always specify causal structure as DAGs with explicit latent confounders
- **Internally**: Unroll temporal DAG to 2 timesteps (per A3a), then project to ADMG
- Never ask users to specify bidirected edges directly
- Lagged confounding (U_{t-1} → X_t, U_{t-1} → Y_t) is properly handled via unrolling

### Time-Unrolling for Temporal Identification
Under AR(1) (A3) and bounded latent reach (A3a), a 2-timestep unrolling suffices
to decide identifiability (per arXiv:2504.20172). This correctly handles lagged
confounding that would otherwise be missed.

```python
# unroll_temporal_dag() creates nodes like X_t, X_{t-1}
# with AR(1) edges for endogenous constructs
# Lagged edges become: cause_{t-1} → effect_t
# Contemporaneous edges become: cause_t → effect_t
```

### Identification Pattern
```python
from dsem_agent.utils.identifiability import check_identifiability

result = check_identifiability(latent_model, measurement_model)
# Internally:
# 1. Unrolls to 2-timestep DAG with X_t, X_{t-1} nodes
# 2. Projects to ADMG via y0's from_latent_variable_dag()
# 3. Checks P(Y_t | do(X_t)) identifiability
```

See `src/dsem_agent/utils/identifiability.py` for `dag_to_admg()` and
`unroll_temporal_dag()` implementations.

# NetworkX
Docs: https://networkx.org/documentation/stable/

## Best Practices
- Use `DiGraph` for causal DAGs (directed edges)
- Create from edge list: `nx.DiGraph([(cause, effect), ...])`
- Check for cycles: `nx.is_directed_acyclic_graph(G)`
- Add node attributes: `G.add_node(name, dtype='continuous', ...)`

# ArViz
Docs: https://python.arviz.org/en/stable/index.html

# DSPy for prompt optimization
Docs: https://dspy.ai/

## Best Practices

### Signatures
Define input/output contracts with type hints:
```python
class StructureProposal(dspy.Signature):
    """Propose causal model structure from data."""
    question: str = dspy.InputField()
    data_sample: str = dspy.InputField()
    structure: str = dspy.OutputField(desc="JSON with dimensions, edges, etc.")
```

### Modules
- `dspy.Predict(Signature)` - basic prediction
- `dspy.ChainOfThought(Signature)` - adds reasoning steps
- `dspy.ReAct(Signature, tools=[...])` - reasoning + actions

### LM Configuration
```python
import dspy
lm = dspy.LM("openrouter/google/gemini-2.5-pro-preview-06-05")
dspy.configure(lm=lm)
```

### Optimizers
- `LabeledFewShot(k=8)` - use labeled examples directly
- `BootstrapFewShot(metric=fn)` - learn demos from successful runs
- `MIPROv2(metric=fn)` - instruction + demo optimization with Bayesian search

### Optimization Pattern
```python
from dspy.teleprompt import MIPROv2

def metric(example, pred, trace=None):
    # Return True/False or 0-1 score
    return validate_structure(pred.structure)

optimizer = MIPROv2(metric=metric, num_threads=4)
compiled = optimizer.compile(program, trainset=examples)
compiled.save("optimized_program.json")
```

### Data Split
- 20% train, 80% validation (prompt optimizers overfit small train sets)
- Aim for 30-300 training examples