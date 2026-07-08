"""Validate ``config.yaml`` and jaxtyping runtime-check wiring.

Reports schema errors (enum values, field-harness compatibility, extraction
harness lock) and optionally runtime prereqs (binary on PATH, required env
vars). Also verifies that every ``src/`` module using jaxtyping shape
annotations is registered in ``--jaxtyping-packages`` (so its runtime shape
checks actually run) and has not silently disabled them via
``from __future__ import annotations``. ``--fix`` auto-syncs that list so it
never needs hand-editing.

Usage::

    uv run python scripts/validate_config.py
    uv run python scripts/validate_config.py --config path/to/config.yaml
    uv run python scripts/validate_config.py --runtime
    uv run python scripts/validate_config.py --fix   # auto-sync --jaxtyping-packages

Exits with code 1 if any errors are found, 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

from nof1_causal_lab.utils import config as config_mod
from nof1_causal_lab.utils.config import (
    load_config,
    validate_config,
    validate_runtime_prereqs,
)

# jaxtyping array/dtype annotation types whose per-call shape checks the
# ``--jaxtyping-packages`` import hook installs. ``shapes.py`` re-exports these and
# is the canonical import source; ``FloatScalar`` is its local scalar alias.
_JAXTYPING_TYPES = frozenset(
    {
        "Array",
        "ArrayLike",
        "Bool",
        "Complex",
        "Float",
        "FloatScalar",
        "Inexact",
        "Int",
        "Integer",
        "Key",
        "Num",
        "PRNGKeyArray",
        "Real",
        "Scalar",
        "Shaped",
        "UInt",
    }
)
_SHAPES_MODULE = "nof1_causal_lab.models.ssm.shapes"


def _path_to_module(py_path: Path, src_root: Path) -> str:
    parts = list(py_path.relative_to(src_root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_shape_names(tree: ast.Module) -> set[str]:
    """Local names bound to a jaxtyping shape type (via shapes or jaxtyping)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == _SHAPES_MODULE:
                names.update(alias.asname or alias.name for alias in node.names)
            elif node.module == "jaxtyping":
                names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in _JAXTYPING_TYPES
                )
    return names


def _has_jaxtyping_annotation(tree: ast.Module, shape_names: set[str]) -> bool:
    """True if any function signature in the module is annotated with a shape type."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = node.args
        annotations = [
            arg.annotation
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
            if arg is not None and arg.annotation is not None
        ]
        if node.returns is not None:
            annotations.append(node.returns)
        for annotation in annotations:
            if any(
                isinstance(sub, ast.Name) and sub.id in shape_names for sub in ast.walk(annotation)
            ):
                return True
    return False


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _wired_jaxtyping_packages(pyproject: Path) -> set[str]:
    addopts = tomllib.loads(pyproject.read_text())["tool"]["pytest"]["ini_options"]["addopts"]
    match = re.search(r"--jaxtyping-packages=(\S+)", addopts)
    if match is None:
        return set()
    return {pkg for pkg in match.group(1).split(",") if pkg and pkg != "beartype.beartype"}


def _scan_jaxtyping_modules(src_root: Path) -> tuple[set[str], set[str]]:
    """Scan ``src/`` for modules with shape-annotated function signatures.

    Returns ``(annotated, future_disabled)``: modules whose function signatures use
    jaxtyping shape types, and the subset that also use ``from __future__ import
    annotations`` (which silently turns their runtime checks into no-ops).
    """
    annotated: set[str] = set()
    future_disabled: set[str] = set()
    for py_path in src_root.rglob("*.py"):
        module = _path_to_module(py_path, src_root)
        if module == _SHAPES_MODULE:
            continue  # the type-alias definitions module is not instrumented
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        shape_names = _imported_shape_names(tree)
        if shape_names and _has_jaxtyping_annotation(tree, shape_names):
            annotated.add(module)
            if _has_future_annotations(tree):
                future_disabled.add(module)
    return annotated, future_disabled


def validate_jaxtyping_wiring(repo_root: Path) -> tuple[list[str], list[str]]:
    """Check that jaxtyping runtime checks actually cover the modules that need them.

    Returns ``(errors, notes)``. Errors (fatal) are the cases where shape checks
    silently do not run: a module annotated but missing from ``--jaxtyping-packages``,
    or a listed module that disabled its checks via ``from __future__ import
    annotations``. Notes (non-fatal) are benign stale list entries. ``--fix``
    resolves the missing-module errors and the stale notes; the ``__future__`` case
    needs a source edit.
    """
    src_root = repo_root / "src"
    pyproject = repo_root / "pyproject.toml"
    if not src_root.is_dir() or not pyproject.is_file():
        return []

    wired = _wired_jaxtyping_packages(pyproject)
    annotated, future_disabled = _scan_jaxtyping_modules(src_root)

    errors = [
        f"{module}: uses jaxtyping shape annotations but is missing from "
        "--jaxtyping-packages (shape checks silently skipped); run `bun run lint:fix`"
        for module in sorted(annotated - wired)
    ]
    errors += [
        f"{module}: instrumented but uses `from __future__ import annotations`, "
        "which makes its jaxtyping shape checks silently no-op"
        for module in sorted(future_disabled & wired)
    ]
    notes = [
        f"{module}: listed in --jaxtyping-packages but has no runtime-checkable "
        "jaxtyping annotations; `bun run lint:fix` will drop it"
        for module in sorted(wired - annotated)
    ]
    return errors, notes


def sync_jaxtyping_wiring(repo_root: Path) -> str | None:
    """Rewrite ``--jaxtyping-packages`` in pyproject.toml to the detected modules.

    Returns a summary of the change, or ``None`` if it was already in sync. The
    ``beartype.beartype`` typechecker entry is preserved as the final element. Does
    not touch the ``from __future__ import annotations`` footgun — fixing that needs
    a source edit, so it stays a check-only error.
    """
    src_root = repo_root / "src"
    pyproject = repo_root / "pyproject.toml"
    annotated, _future_disabled = _scan_jaxtyping_modules(src_root)
    desired_value = ",".join([*sorted(annotated), "beartype.beartype"])

    text = pyproject.read_text()
    match = re.search(r'--jaxtyping-packages=[^\s"]+', text)
    if match is None:
        raise SystemExit(
            "error: --jaxtyping-packages not found in pyproject.toml [tool.pytest] addopts"
        )
    current_value = match.group(0).split("=", 1)[1]
    if current_value == desired_value:
        return None

    pyproject.write_text(
        text[: match.start()] + "--jaxtyping-packages=" + desired_value + text[match.end() :]
    )
    current_mods = {m for m in current_value.split(",") if m != "beartype.beartype"}
    changes = []
    if added := sorted(annotated - current_mods):
        changes.append("added " + ", ".join(added))
    if removed := sorted(current_mods - annotated):
        changes.append("removed " + ", ".join(removed))
    return "; ".join(changes) if changes else "reordered to canonical order"


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
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-sync --jaxtyping-packages in pyproject.toml to the modules that "
        "use jaxtyping annotations (rewrites pyproject.toml; used by lint:fix)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent

    if args.fix:
        change = sync_jaxtyping_wiring(repo_root)
        print(
            "Jaxtyping wiring: already in sync"
            if change is None
            else f"Jaxtyping wiring: updated --jaxtyping-packages ({change})"
        )
        return 0

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

    jaxtyping_errors, jaxtyping_notes = validate_jaxtyping_wiring(repo_root)
    for note in jaxtyping_notes:
        print(f"  note: {note}")
    if jaxtyping_errors:
        print("\nJaxtyping wiring errors:")
        for err in jaxtyping_errors:
            print(f"  - {err}")
        return 1
    print("Jaxtyping wiring: OK")

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
