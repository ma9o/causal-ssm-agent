"""Validate ``config.yaml``.

Reports schema errors (enum values, field-harness compatibility, Stage 2
harness lock) and optionally runtime prereqs (binary on PATH, required env
vars).

Usage::

    uv run python scripts/validate_config.py
    uv run python scripts/validate_config.py --config path/to/config.yaml
    uv run python scripts/validate_config.py --runtime

Exits with code 1 if any errors are found, 0 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nof1_causal_lab.utils import config as config_mod
from nof1_causal_lab.utils.config import (
    load_config,
    validate_config,
    validate_runtime_prereqs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the pipeline config.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: auto-discover)",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Also check runtime prerequisites (binaries on PATH, env vars)",
    )
    args = parser.parse_args(argv)

    if args.config is not None:
        if not args.config.exists():
            print(f"error: config file not found: {args.config}", file=sys.stderr)
            return 1
        load_config.cache_clear()
        config_mod._find_config_path = lambda: args.config  # type: ignore[attr-defined]

    print(f"Validating: {config_mod._find_config_path()}")

    try:
        config = load_config()
    except ValueError as exc:
        print("\nSchema errors:")
        print(exc)
        return 1

    # load_config already raises on schema errors, but run validate_config again
    # for completeness in case someone bypasses load_config.
    schema_errors = validate_config(config)
    if schema_errors:
        print("\nSchema errors:")
        for err in schema_errors:
            print(f"  - {err}")
        return 1

    print("Schema: OK")

    if args.runtime:
        runtime_errors = validate_runtime_prereqs(config)
        if runtime_errors:
            print("\nRuntime prereq errors:")
            for err in runtime_errors:
                print(f"  - {err}")
            return 1
        print("Runtime prereqs: OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
