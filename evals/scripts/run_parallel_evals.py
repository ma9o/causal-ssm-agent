#!/usr/bin/env python
"""Run orchestrator evals across all models in parallel.

Usage:
    uv run python evals/scripts/run_parallel_evals.py
    uv run python evals/scripts/run_parallel_evals.py --models claude gemini
    uv run python evals/scripts/run_parallel_evals.py -n 10 --seed 123
    uv run python evals/scripts/run_parallel_evals.py --viz  # Open DAG viz for successful results
"""

import argparse
import asyncio
import json
import re
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

# Import model registry from main eval module (single source of truth)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from orchestrator_structure import MODELS

# Reverse mapping: alias -> model ID
ALIAS_TO_MODEL = {alias: model_id for model_id, alias in MODELS.items()}

# Path to visualizer
VIZ_PATH = Path(__file__).parent.parent.parent / "tools" / "dag_visualizer.html"


@dataclass
class EvalResult:
    model: str
    success: bool
    duration: float
    output: str
    mean_score: float | None = None
    stderr: float | None = None
    log_path: Path | None = None
    structures: list[str] = field(default_factory=list)  # Valid JSON structures


def parse_results(output: str) -> tuple[float | None, float | None, Path | None]:
    """Extract mean, stderr, and log path from inspect output."""
    mean_score = None
    stderr = None
    log_path = None

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
        elif line.startswith("logs/") and line.endswith(".eval"):
            log_path = Path(line)

    return mean_score, stderr, log_path


def extract_structures_from_log(log_path: Path) -> list[str]:
    """Extract valid JSON structures from Inspect log file."""
    structures = []
    if not log_path or not log_path.exists():
        return structures

    try:
        with open(log_path) as f:
            log_data = json.load(f)

        for sample in log_data.get("samples", []):
            # Check if score > 0
            score_value = sample.get("scores", {}).get("dsem_structure_scorer", {}).get("value", 0)
            if isinstance(score_value, (int, float)) and score_value > 0:
                # Get the answer (JSON structure)
                answer = sample.get("scores", {}).get("dsem_structure_scorer", {}).get("answer", "")
                if answer and not answer.startswith("["):
                    # Remove truncation suffix if present
                    if answer.endswith("..."):
                        # Try to find the full JSON in the model output
                        for msg in reversed(sample.get("messages", [])):
                            if msg.get("role") == "assistant":
                                content = msg.get("content", "")
                                if isinstance(content, str):
                                    # Extract JSON from content
                                    match = re.search(r'\{[\s\S]*\}', content)
                                    if match:
                                        try:
                                            json.loads(match.group())
                                            answer = match.group()
                                            break
                                        except json.JSONDecodeError:
                                            pass
                    structures.append(answer)
    except Exception as e:
        print(f"Warning: Could not parse log {log_path}: {e}", file=sys.stderr)

    return structures


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
    mean_score, stderr, log_path = parse_results(output) if success else (None, None, None)
    structures = extract_structures_from_log(log_path) if log_path else []

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
        log_path=log_path,
        structures=structures,
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


def print_summary(results: list[EvalResult]) -> list[EvalResult]:
    """Print summary table of results. Returns sorted results."""
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
    return sorted_results


def open_visualizer(structure_json: str, model_name: str) -> None:
    """Open DAG visualizer with the given structure."""
    encoded = quote(structure_json, safe='')
    url = f"file://{VIZ_PATH.resolve()}?data={encoded}"
    print(f"Opening visualizer for {model_name}...", file=sys.stderr)
    webbrowser.open(url)


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
    parser.add_argument("--viz", action="store_true", help="Open DAG visualizer for successful structures")
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

    sorted_results = print_summary(results)

    # Open visualizer for best model's first successful structure
    if args.viz:
        for r in sorted_results:
            if r.structures:
                open_visualizer(r.structures[0], short_model_name(r.model))
                break
        else:
            print("No valid structures to visualize", file=sys.stderr)

    # Exit with error if any failed
    if not all(r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
