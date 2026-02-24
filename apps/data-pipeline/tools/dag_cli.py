"""CLI interface for DAG diagnostics.

Usage:
    uv run python tools/dag_cli.py data/eval/questions/1_resolve-errors-faster/causal_spec.json

The CLI emits machine-friendly JSON by default so agents can read the same
diagnostics shown in the Streamlit UI. Use ``--format text`` for a human summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dag_diagnostics import load_model_file, run_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect DAG diagnostics.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a causal model JSON file (latent + measurement).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format. JSON includes structured diagnostics (default).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent level (use 0 for compact output).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = load_model_file(args.path)
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    diagnostics = run_diagnostics(data)

    if args.format == "json":
        payload = diagnostics.to_dict()
        payload["input_path"] = str(args.path.resolve())
        indent = None if args.indent <= 0 else args.indent
        print(json.dumps(payload, indent=indent))
    else:
        print(diagnostics.identifiability_report)
        print()
        print(diagnostics.marginalization_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
