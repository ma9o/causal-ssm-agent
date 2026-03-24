# Running Evaluations

## Cost and Workflow Warning

These evals consume paid model calls. In normal repo work, do **not** run them casually. If you are validating code changes rather than evaluating model quality, use `uv run pytest tests/` instead.

## Available Evals

The active checked-in eval surface mostly covers the pre-Stage-4 pipeline
areas, plus one tracked fixture-level orchestrator eval. Some eval filenames
keep older stage numbering. The current pipeline stages are `stage-0 → stage-1a → stage-1b → stage-2 → stage-3 → stage-4 → stage-4b → stage-5a → stage-5b → stage-6`.

| File | Current pipeline area | What it tests |
|------|-----------------------|---------------|
| `evals/single_model/eval1a_latent_model.py` | Stage 1a | Latent model proposal (orchestrator) |
| `evals/single_model/eval1b_measurement_model.py` | Stage 1b | Measurement model proposal |
| `evals/single_model/eval2_worker_extraction.py` | Stage 2 | Worker data extraction |
| `evals/multi_model/eval3_worker_measurement_adherence.py` | Stage 2 workers | Judge-based worker adherence to measurement instructions |
| `evals/multi_model/eval_medical_semantics_orchestrator.py` | Stages 1a -> 1b -> 2 | Judge-ranked orchestrator reproduction of the fixed `MEDICAL_SEMANTICS` fixture |

The old split Stage 4 evals were removed because they no longer match the
current agentic Stage 4 runtime. Reintroduce Stage 4 eval coverage as a single
end-to-end eval around `causal_ssm_agent.orchestrator.stage4.run_stage4()`.

`evals/scripts/run_parallel_evals.py` currently wraps only the Stage 1a
orchestrator eval and the Stage 2 worker eval. The `MEDICAL_SEMANTICS`
fixture eval is run directly with `inspect eval`.

Worker-facing evals load persisted workspace artifacts, not ad hoc preprocessed
text files. The default workspace is `GOLDEN` from `evals/config.yaml`, and you
can override it with `workspace_id`.

## Run all models in parallel

Only run these when you explicitly intend to spend tokens on evals.

```bash
# Stage 1a orchestrator eval (default) — runs configured models concurrently
uv run python evals/scripts/run_parallel_evals.py

# Stage 2 worker eval against the default GOLDEN workspace
uv run python evals/scripts/run_parallel_evals.py --eval worker
uv run python evals/scripts/run_parallel_evals.py --eval worker --workspace-id SMALLGOLDEN

# Run specific models using aliases
uv run python evals/scripts/run_parallel_evals.py --models claude gemini gpt
uv run python evals/scripts/run_parallel_evals.py --eval worker --models gemini haiku

# Customize worker sampling
uv run python evals/scripts/run_parallel_evals.py --eval worker -n 10 --seed 123

# Filter to specific Stage 1a questions
uv run python evals/scripts/run_parallel_evals.py -q 1,3

# Stage 1a aliases: claude, gemini, gpt, deepseek, kimi
# Stage 2 aliases:   kimi, deepseek, gemini, grok, haiku, minimax, gpt-oss
```

## Run individual models

Again: these are opt-in costful commands, not the default verification path for everyday development.

```bash
uv run inspect eval evals/single_model/eval1a_latent_model.py \
    --model openrouter/anthropic/claude-opus-4.6

uv run inspect eval evals/single_model/eval1b_measurement_model.py \
    --model openrouter/anthropic/claude-opus-4.6 \
    -T workspace_id=GOLDEN

uv run inspect eval evals/single_model/eval2_worker_extraction.py \
    --model openrouter/anthropic/claude-haiku-4.5 \
    -T workspace_id=SMALLGOLDEN

uv run inspect eval evals/multi_model/eval3_worker_measurement_adherence.py \
    -T workspace_id=SMALLGOLDEN

# MEDICAL_SEMANTICS fixture orchestrator eval
uv run inspect eval evals/multi_model/eval_medical_semantics_orchestrator.py
uv run inspect eval evals/multi_model/eval_medical_semantics_orchestrator.py \
    -T models=openrouter/anthropic/claude-opus-4.6,openrouter/openai/gpt-5.1

# View detailed results
uv run inspect view
```

## Tracked Fixture Workspaces

Evals and manual prompt-sampling tools read the same persisted workspace artifacts
that the pipeline uses. By default they load `data/GOLDEN/`, but you can point
them at any workspace with compatible `query.txt` and `run/stage-*.json` outputs.
For the full workspace directory layout, see [Agentic Integration Testing](agentic_integration_testing.md#workspace-layout).

## Log directories

- `inspect eval` saves logs to the default `logs/` directory.
- `run_parallel_evals.py` saves logs to `logs/{eval}-{timestamp}/` (e.g. `logs/orchestrator-20260305-143000/`). Override with `--log-dir`.

## Reading Eval Logs

### Quick summary with `tools/read_eval_log.py`

```bash
uv run python tools/read_eval_log.py                    # Latest log
uv run python tools/read_eval_log.py -e aggregation     # Latest by eval name
uv run python tools/read_eval_log.py -f <filename>      # Specific file
uv run python tools/read_eval_log.py --list             # List available logs
```

### Programmatic access

```python
from inspect_ai.log import read_eval_log

log = read_eval_log('logs/xxx.eval')

# Basic info
print(f'Model: {log.eval.model}')
print(f'Samples: {len(log.samples)}')

# Sample scores
for s in log.samples:
    print(s.id, s.scores)
```

### Debugging with Events

Each sample has events showing the full execution trace:

```python
for s in log.samples:
    for event in s.events:
        event_type = type(event).__name__

        # Tool calls and results
        if 'Tool' in event_type:
            print(f'Tool: {event.function}, Result: {event.result}')

        # Model completions
        if 'Model' in event_type and hasattr(event, 'output'):
            print(f'Completion ({len(event.output.completion)} chars)')
```

### Accessing Attachments

Large arguments (like JSON payloads) are stored as attachments:

```python
for s in log.samples:
    for k, v in s.attachments.items():
        print(f'{k}: {str(v)[:500]}')
```

### Model Usage Stats

`model_usage` is a dict keyed by model name:

```python
for model_name, usage in log.samples[0].model_usage.items():
    print(f'{model_name}: {usage.input_tokens} in, {usage.output_tokens} out')
```
