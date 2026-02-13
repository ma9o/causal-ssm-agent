#!/usr/bin/env python
"""Run evals across all models in parallel using Inspect's eval_set.

Usage:
    # Orchestrator eval (default)
    uv run python evals/scripts/run_parallel_evals.py
    uv run python evals/scripts/run_parallel_evals.py --models claude gemini

    # Worker eval
    uv run python evals/scripts/run_parallel_evals.py --eval worker
    uv run python evals/scripts/run_parallel_evals.py --eval worker --models gemini haiku

    # Common options
    uv run python evals/scripts/run_parallel_evals.py -n 10 --seed 123
    uv run python evals/scripts/run_parallel_evals.py --max-tasks 8

    # Filter to specific questions
    uv run python evals/scripts/run_parallel_evals.py --eval worker -q 1
    uv run python evals/scripts/run_parallel_evals.py -q 1,4
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import load_eval_config
from inspect_ai import eval_set
from inspect_ai.log import EvalLog

# Load config
CONFIG = load_eval_config()

# Eval configurations
EVAL_CONFIGS = {
    "orchestrator": {
        "file": "evals/eval1a_latent_model.py",
        "models": {m["id"]: m["alias"] for m in CONFIG["orchestrator_models"]},
        "task_params": ["n_chunks", "seed", "input_file", "questions"],
    },
    "worker": {
        "file": "evals/eval2_worker_extraction.py",
        "models": {m["id"]: m["alias"] for m in CONFIG["worker_models"]},
        "task_params": ["n_chunks", "seed", "input_file", "question"],
    },
}


def short_model_name(model: str, models_dict: dict) -> str:
    """Get short display name for model."""
    return models_dict.get(model, model.split("/")[-1])


def print_summary(logs: list[EvalLog], models_dict: dict):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Extract results from logs
    results = []
    for log in logs:
        model = log.eval.model
        status = log.status

        # Get mean score from results
        mean_score = None
        stderr = None
        if log.results and log.results.scores:
            for score in log.results.scores:
                if score.metrics:
                    mean_metric = score.metrics.get("mean")
                    if mean_metric:
                        mean_score = mean_metric.value
                    stderr_metric = score.metrics.get("stderr")
                    if stderr_metric:
                        stderr = stderr_metric.value
                    break

        results.append({
            "model": model,
            "status": status,
            "mean_score": mean_score,
            "stderr": stderr,
        })

    # Sort by mean score descending (failures at bottom)
    sorted_results = sorted(
        results,
        key=lambda r: (r["mean_score"] is not None, r["mean_score"] or 0),
        reverse=True,
    )

    print(f"{'Model':<15} {'Score':>10} {'Stderr':>10} {'Status':<10}")
    print("-" * 70)

    for r in sorted_results:
        name = short_model_name(r["model"], models_dict)
        score = f"{r['mean_score']:.1f}" if r["mean_score"] is not None else "-"
        stderr = f"{r['stderr']:.2f}" if r["stderr"] is not None else "-"
        status = r["status"]
        print(f"{name:<15} {score:>10} {stderr:>10} {status:<10}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run evals in parallel using eval_set")
    parser.add_argument(
        "--eval",
        choices=["orchestrator", "worker"],
        default="orchestrator",
        help="Which eval to run (default: orchestrator)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to eval (use aliases or full model IDs)",
    )
    parser.add_argument("-n", "--n-chunks", type=int, default=5, help="Chunks per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-i", "--input-file", help="Specific input file name")
    parser.add_argument(
        "-q", "--question",
        help="Question selector (prefix ID or slug). For worker eval: single question. "
             "For orchestrator eval: comma-separated list.",
    )
    parser.add_argument("--max-tasks", type=int, help="Max parallel tasks (default: max(4, num_models))")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Max retry attempts (default: 3)")
    parser.add_argument("--log-dir", help="Log directory (default: logs/<eval>-<timestamp>)")
    args = parser.parse_args()

    # Get eval config
    config = EVAL_CONFIGS[args.eval]
    models_dict = config["models"]
    alias_to_model = {alias: model_id for model_id, alias in models_dict.items()}

    # Resolve model names
    if args.models:
        models = []
        for m in args.models:
            if m in alias_to_model:
                models.append(alias_to_model[m])
            else:
                models.append(m)
    else:
        models = list(models_dict.keys())

    # Build task params string for -T flags
    task_params = {}
    task_params["n_chunks"] = args.n_chunks
    task_params["seed"] = args.seed
    if args.input_file:
        task_params["input_file"] = args.input_file
    if args.question:
        if args.eval == "worker":
            task_params["question"] = args.question
        else:
            task_params["questions"] = args.question

    # Set log directory
    if args.log_dir:
        log_dir = args.log_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/{args.eval}-{timestamp}"

    print(f"Running {args.eval} eval for {len(models)} models...", file=sys.stderr)
    print(f"Config: n_chunks={args.n_chunks}, seed={args.seed}", file=sys.stderr)
    print(f"Models: {', '.join(short_model_name(m, models_dict) for m in models)}", file=sys.stderr)
    print(f"Log dir: {log_dir}", file=sys.stderr)
    print(file=sys.stderr)

    # Run eval_set
    success, logs = eval_set(
        tasks=[config["file"]],
        model=models,
        task_args=task_params,
        log_dir=log_dir,
        max_tasks=args.max_tasks,
        retry_attempts=args.retry_attempts,
    )

    print_summary(logs, models_dict)

    # Exit with error if any failed
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
