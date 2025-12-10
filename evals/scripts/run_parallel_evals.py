#!/usr/bin/env python
"""Run orchestrator evals across all models in parallel.

Usage:
    uv run python evals/scripts/run_parallel_evals.py
    uv run python evals/scripts/run_parallel_evals.py --models claude gemini
    uv run python evals/scripts/run_parallel_evals.py -n 10 --seed 123
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass

# Import model registry from main eval module (single source of truth)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from orchestrator_structure import MODELS

# Reverse mapping: alias -> model ID
ALIAS_TO_MODEL = {alias: model_id for model_id, alias in MODELS.items()}


@dataclass
class EvalResult:
    model: str
    success: bool
    duration: float
    output: str
    mean_score: float | None = None
    stderr: float | None = None


def parse_results(output: str) -> tuple[float | None, float | None]:
    """Extract mean and stderr from inspect output."""
    mean_score = None
    stderr = None

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("mean"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mean_score = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith("stderr"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    stderr = float(parts[1])
                except ValueError:
                    pass

    return mean_score, stderr


def short_model_name(model: str) -> str:
    """Get short display name for model."""
    return MODELS.get(model, model.split("/")[-1])


async def run_eval(
    model: str,
    n_chunks: int,
    seed: int,
    input_file: str | None,
) -> EvalResult:
    """Run a single eval and return results."""
    short_name = short_model_name(model)
    print(f"[{short_name}] Starting...", file=sys.stderr)

    cmd = ["uv", "run", "inspect", "eval", "evals/orchestrator_structure.py", "--model", model]
    cmd.extend(["-T", f"n_chunks={n_chunks}"])
    cmd.extend(["-T", f"seed={seed}"])
    if input_file:
        cmd.extend(["-T", f"input_file={input_file}"])

    start = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    stdout, _ = await proc.communicate()
    duration = time.time() - start
    output = stdout.decode()

    success = proc.returncode == 0
    mean_score, stderr = parse_results(output) if success else (None, None)

    status = "done" if success else "FAILED"
    score_str = f" (mean: {mean_score:.1f})" if mean_score is not None else ""
    print(f"[{short_name}] {status} in {duration:.0f}s{score_str}", file=sys.stderr)

    return EvalResult(
        model=model,
        success=success,
        duration=duration,
        output=output,
        mean_score=mean_score,
        stderr=stderr,
    )


async def run_all_evals(
    models: list[str],
    n_chunks: int,
    seed: int,
    input_file: str | None,
) -> list[EvalResult]:
    """Run all evals in parallel."""
    tasks = [
        run_eval(model, n_chunks, seed, input_file)
        for model in models
    ]
    return await asyncio.gather(*tasks)


def print_summary(results: list[EvalResult]) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Sort by mean score descending (failures at bottom)
    sorted_results = sorted(
        results,
        key=lambda r: (r.mean_score is not None, r.mean_score or 0),
        reverse=True,
    )

    print(f"{'Model':<15} {'Score':>10} {'Stderr':>10} {'Time':>10} {'Status':<10}")
    print("-" * 60)

    for r in sorted_results:
        name = short_model_name(r.model)
        score = f"{r.mean_score:.1f}" if r.mean_score is not None else "-"
        stderr = f"{r.stderr:.2f}" if r.stderr is not None else "-"
        time_str = f"{r.duration:.0f}s"
        status = "OK" if r.success else "FAILED"
        print(f"{name:<15} {score:>10} {stderr:>10} {time_str:>10} {status:<10}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run orchestrator evals in parallel")
    parser.add_argument(
        "--models",
        nargs="+",
        help=f"Models to eval (aliases: {', '.join(MODELS.values())})",
    )
    parser.add_argument("-n", "--n-chunks", type=int, default=5, help="Chunks per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-i", "--input-file", help="Specific input file name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full output")
    args = parser.parse_args()

    # Resolve model names
    if args.models:
        models = []
        for m in args.models:
            if m in ALIAS_TO_MODEL:
                models.append(ALIAS_TO_MODEL[m])
            else:
                models.append(m)
    else:
        models = list(MODELS.keys())

    print(f"Running evals for {len(models)} models in parallel...", file=sys.stderr)
    print(f"Config: n_chunks={args.n_chunks}, seed={args.seed}", file=sys.stderr)
    print(file=sys.stderr)

    results = asyncio.run(run_all_evals(models, args.n_chunks, args.seed, args.input_file))

    if args.verbose:
        for r in results:
            print(f"\n{'=' * 60}")
            print(f"MODEL: {r.model}")
            print("=" * 60)
            print(r.output)

    print_summary(results)

    # Exit with error if any failed
    if not all(r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
