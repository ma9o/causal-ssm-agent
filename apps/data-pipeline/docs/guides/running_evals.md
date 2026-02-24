# Running Evaluations

Evaluate LLM performance on structure proposal tasks using Inspect AI. Only top-tier models with max thinking budget are used.

## Run all models in parallel

```bash
# Run all 5 models concurrently (recommended)
uv run python evals/scripts/run_parallel_evals.py

# Run specific models using aliases
uv run python evals/scripts/run_parallel_evals.py --models claude gemini gpt

# Customize parameters
uv run python evals/scripts/run_parallel_evals.py -n 10 --seed 123

# Available aliases: claude, gemini, gpt, deepseek, kimi
```

## Run individual models

```bash
uv run inspect eval evals/single_model/eval1a_latent_model.py \
    --model openrouter/anthropic/claude-opus-4.6

# View detailed results
uv run inspect view
```

Logs are saved to `logs/` directory. The eval scores models on cumulative points for valid causal model structures.

## Reading Eval Logs

Eval logs are compressed files. Use `read_eval_log` to parse them:

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

### Debugging with Transcript Events

Each sample has a transcript with events showing the full execution trace:

```python
for s in log.samples:
    for event in s.transcript.events:
        event_type = type(event).__name__

        # Tool calls and results
        if 'Tool' in event_type:
            print(f'Tool: {event.function}, Result: {event.result}')

        # Model completions
        if 'Model' in event_type and hasattr(event, 'output'):
            print(f'Completion ({len(event.output.completion)} chars)')
```

### Accessing Tool Call Arguments

Large arguments (like JSON payloads) are stored as attachments:

```python
for s in log.samples:
    for k, v in s.attachments.items():
        print(f'{k}: {str(v)[:500]}')
```

### Model Usage Stats

```python
print(log.samples[0].model_usage)
# ModelUsage(input_tokens=..., output_tokens=..., reasoning_tokens=...)
```
