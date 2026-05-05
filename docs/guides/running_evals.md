# Running Evaluations

| File | Current pipeline area | What it tests |
|------|-----------------------|---------------|
| `evals/single_model/eval1a_latent_model.py` | Stage 1a | Latent model proposal (orchestrator) |
| `evals/single_model/eval1b_measurement_model.py` | Stage 1b | Measurement model proposal |
| `evals/single_model/eval2_worker_extraction.py` | Stage 2 | Worker data extraction |
| `evals/multi_model/eval3_worker_measurement_adherence.py` | Stage 2 workers | Judge-based worker adherence to measurement instructions |
| `evals/multi_model/eval_medical_semantics_orchestrator.py` | Stages 1a -> 1b -> 2 | Judge-ranked orchestrator reproduction of the fixed `MEDICAL_SEMANTICS` fixture |

Worker-facing evaluations load persisted workspace artifacts, not ad hoc preprocessed
text files. The default workspace is `GOLDEN` from `evals/config.yaml`, and you
can override it with `workspace_id`.

## Run all models in parallel

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

```bash
# General pattern: pick an eval from the table above
uv run inspect eval <eval_file> --model <model> [-T workspace_id=<WS>]

# Example: Stage 1a with a specific model
uv run inspect eval evals/single_model/eval1a_latent_model.py \
    --model openrouter/anthropic/claude-opus-4.6

# MEDICAL_SEMANTICS fixture eval (multi-model, no --model flag)
uv run inspect eval evals/multi_model/eval_medical_semantics_orchestrator.py \
    [-T models=openrouter/anthropic/claude-opus-4.6,openrouter/openai/gpt-5.1]

# View results in browser
uv run inspect view
```

Worker and measurement evaluations require `-T workspace_id=GOLDEN` (or `SMALLGOLDEN`).
Multi-model evaluations (`eval3_*`, `eval_medical_*`) don't take `--model`; they configure models internally.

## Tracked Fixture Workspaces

Evaluations and manual prompt-sampling tools read the same persisted workspace artifacts
that the pipeline uses. By default they load `data/GOLDEN/`, but you can point
them at any workspace with compatible `query.txt` and `run/stage-*.json` outputs.
For the full workspace directory layout, see [Agentic Integration Testing](agentic_integration_testing.md#workspace-layout).

## Log directories

- `inspect eval` saves logs to the default `logs/` directory.
- `run_parallel_evals.py` saves logs to `logs/{eval}-{timestamp}/` (e.g. `logs/orchestrator-20260305-143000/`). Override with `--log-dir`.

## Reading Eval Logs

```bash
uv run python tools/read_eval_log.py                    # Latest log
uv run python tools/read_eval_log.py -e aggregation     # Latest by eval name
uv run python tools/read_eval_log.py -f <filename>      # Specific file
uv run python tools/read_eval_log.py --list             # List available logs
```

For programmatic access to log samples, scores, events, and attachments, see the [inspect_ai.log API docs](https://inspect.aisi.org.uk/log-files.html).
