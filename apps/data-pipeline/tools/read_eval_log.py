"""Tool for reading and extracting data from inspect eval log files.

Usage:
    uv run python tools/read_eval_log.py                    # Latest log
    uv run python tools/read_eval_log.py -e aggregation     # Latest by eval name
    uv run python tools/read_eval_log.py -f <filename>      # Specific file
    uv run python tools/read_eval_log.py --list             # List available logs
"""

from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, read_eval_log


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    # tools/ -> project root -> logs/
    return Path(__file__).parent.parent / "logs"


def list_log_files(eval_name: str | None = None) -> list[Path]:
    """List all log files, optionally filtered by eval name.

    Args:
        eval_name: If provided, filter logs containing this eval name.

    Returns:
        List of log file paths sorted by modification time (newest first).
    """
    logs_dir = get_logs_dir()
    if not logs_dir.exists():
        return []

    files = list(logs_dir.glob("*.eval"))

    if eval_name:
        # Filter by eval name (appears between timestamp and ID in filename)
        files = [f for f in files if eval_name in f.stem]

    # Sort by modification time, newest first
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def read_log(
    filename: str | None = None,
    eval_name: str | None = None,
) -> EvalLog:
    """Read an inspect eval log file.

    Args:
        filename: Specific log filename (with or without path).
        eval_name: Eval name to find latest log for.
            If neither provided, returns the latest log overall.

    Returns:
        EvalLog object containing the log data.

    Raises:
        FileNotFoundError: If no matching log file found.
        ValueError: If both filename and eval_name are provided.
    """
    if filename and eval_name:
        raise ValueError("Provide either filename or eval_name, not both")

    if filename:
        # Handle both absolute paths and just filenames
        path = Path(filename)
        if not path.is_absolute():
            path = get_logs_dir() / filename
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")
        return read_eval_log(str(path))

    # Find by eval_name or get latest
    files = list_log_files(eval_name)
    if not files:
        if eval_name:
            raise FileNotFoundError(f"No logs found for eval: {eval_name}")
        raise FileNotFoundError("No log files found in logs directory")

    return read_eval_log(str(files[0]))


def extract_samples_summary(log: EvalLog) -> list[dict[str, Any]]:
    """Extract a summary of samples from a log.

    Args:
        log: EvalLog object.

    Returns:
        List of dicts with sample id, scores, and metadata.
    """
    summaries = []
    if log.samples:
        for sample in log.samples:
            summary: dict[str, Any] = {
                "id": sample.id,
                "scores": {},
            }
            if sample.scores:
                for scorer_name, score in sample.scores.items():
                    summary["scores"][scorer_name] = {
                        "value": score.value,
                        "answer": score.answer[:200] if score.answer else None,
                        "explanation": score.explanation,
                    }
            if sample.metadata:
                summary["metadata"] = sample.metadata
            summaries.append(summary)
    return summaries


def extract_log_summary(log: EvalLog) -> dict[str, Any]:
    """Extract a comprehensive summary from a log.

    Args:
        log: EvalLog object.

    Returns:
        Dict with eval info, results, and sample summaries.
    """
    summary: dict[str, Any] = {
        "eval": {
            "task": log.eval.task,
            "model": log.eval.model,
            "created": log.eval.created,
        },
        "status": log.status,
        "samples_count": len(log.samples) if log.samples else 0,
    }

    if log.results:
        summary["results"] = {
            "scores": [
                {
                    "name": s.name,
                    "metrics": {name: m.value for name, m in s.metrics.items()},
                }
                for s in log.results.scores
            ]
            if log.results.scores
            else []
        }

    summary["samples"] = extract_samples_summary(log)

    return summary


def print_log_summary(
    filename: str | None = None,
    eval_name: str | None = None,
) -> None:
    """Print a human-readable summary of a log file.

    Args:
        filename: Specific log filename.
        eval_name: Eval name to find latest log for.
    """
    import json

    log = read_log(filename=filename, eval_name=eval_name)
    summary = extract_log_summary(log)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read and summarize inspect eval log files")
    parser.add_argument("-f", "--filename", help="Specific log filename")
    parser.add_argument("-e", "--eval-name", help="Eval name to find latest log for")
    parser.add_argument("--list", action="store_true", help="List available log files")

    args = parser.parse_args()

    if args.list:
        files = list_log_files(args.eval_name)
        if not files:
            print("No log files found")
        else:
            for f in files:
                print(f.name)
    else:
        print_log_summary(filename=args.filename, eval_name=args.eval_name)
