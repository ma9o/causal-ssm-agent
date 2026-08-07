"""Enforce the dependency direction of the structural and SSM architecture.

The rules in this module protect the four promoted seams:

1. The SSM compiler consumes ``StructuralPlan`` instead of reaching into
   structural planning or identification.
2. The numerical SSM layer consumes ``PriorPlan`` and never worker schemas.
3. Compilation and runtime hydration do not call each other.
4. The executable model surface is independent of inference algorithms, while
   inference consumes that surface and cannot reach back through runtime.

Imports guarded by ``TYPE_CHECKING`` are excluded because these rules constrain
runtime ownership and initialization, not type annotation dependencies.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import override

_PACKAGE = "nof1_causal_lab"
_SSM = f"{_PACKAGE}.models.ssm"
_COMPILER = f"{_SSM}.compile"
_EXECUTION = f"{_SSM}.execution"
_INFERENCE = f"{_SSM}.inference"
_RUNTIME = f"{_SSM}.runtime"


@dataclass(frozen=True, slots=True)
class ImportRef:
    """One runtime import between project modules."""

    path: Path
    line: int
    importer: str
    imported: str


@dataclass(frozen=True, slots=True)
class Violation:
    """One forbidden dependency."""

    ref: ImportRef
    code: str
    message: str

    def diagnostic(self, source_root: Path) -> str:
        path = self.ref.path.relative_to(source_root.parent).as_posix()
        return f"{path}:{self.ref.line}: {self.code} {self.message}"


def _is_type_checking_test(node: ast.expr) -> bool:
    return bool(
        (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING")
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "typing"
            and node.attr == "TYPE_CHECKING"
        )
    )


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path, importer: str, package: str) -> None:
        self.path = path
        self.importer = importer
        self.package = package
        self.type_checking_depth = 0
        self.refs: list[ImportRef] = []

    def _append(self, imported: str, line: int) -> None:
        if not self.type_checking_depth and imported.startswith(_PACKAGE):
            self.refs.append(
                ImportRef(
                    path=self.path,
                    line=line,
                    importer=self.importer,
                    imported=imported,
                )
            )

    @override
    def visit_If(self, node: ast.If) -> None:
        if not _is_type_checking_test(node.test):
            self.generic_visit(node)
            return

        self.type_checking_depth += 1
        for statement in node.body:
            self.visit(statement)
        self.type_checking_depth -= 1
        for statement in node.orelse:
            self.visit(statement)

    @override
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._append(alias.name, node.lineno)

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            relative_name = "." * node.level + (node.module or "")
            base = importlib.util.resolve_name(relative_name, self.package)
        else:
            base = node.module or ""
        self._append(base, node.lineno)


def _module_name(source_root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(source_root).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    suffix = ".".join(parts)
    module = _PACKAGE if not suffix else f"{_PACKAGE}.{suffix}"
    package = module if is_package else module.rpartition(".")[0]
    return module, package


def collect_imports(source_root: Path) -> tuple[ImportRef, ...]:
    """Parse runtime imports below the package source root."""
    refs: list[ImportRef] = []
    for path in sorted(source_root.rglob("*.py")):
        importer, package = _module_name(source_root, path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _ImportVisitor(path=path, importer=importer, package=package)
        visitor.visit(tree)
        refs.extend(visitor.refs)
    return tuple(refs)


def _is_module(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def find_violations(source_root: Path) -> tuple[Violation, ...]:
    """Return every forbidden runtime dependency below ``source_root``."""
    violations: list[Violation] = []
    structural_owners = (
        f"{_PACKAGE}.models.structural",
        f"{_PACKAGE}.utils.causal_design",
        f"{_PACKAGE}.utils.identifiability",
    )

    for ref in collect_imports(source_root):
        if _is_module(ref.importer, _COMPILER) and any(
            _is_module(ref.imported, owner) for owner in structural_owners
        ):
            violations.append(
                Violation(
                    ref,
                    "ARCH001",
                    "the SSM compiler must consume StructuralPlan, not structural planners",
                )
            )

        if _is_module(ref.importer, _SSM) and _is_module(ref.imported, f"{_PACKAGE}.workers"):
            violations.append(
                Violation(
                    ref,
                    "ARCH002",
                    "the SSM layer must consume typed artifacts, not worker schemas",
                )
            )

        if _is_module(ref.importer, _COMPILER) and _is_module(ref.imported, _RUNTIME):
            violations.append(
                Violation(
                    ref,
                    "ARCH003",
                    "the compiler must serialize output without calling runtime hydration",
                )
            )

        if ref.importer == _RUNTIME and _is_module(ref.imported, _COMPILER):
            violations.append(
                Violation(
                    ref,
                    "ARCH004",
                    "runtime hydration must not call compiler implementation modules",
                )
            )

        if (
            ref.importer in {f"{_SSM}.model", _RUNTIME} or _is_module(ref.importer, _EXECUTION)
        ) and _is_module(ref.imported, _INFERENCE):
            violations.append(
                Violation(
                    ref,
                    "ARCH005",
                    "the executable SSM surface and hydration must not depend on inference",
                )
            )

        if _is_module(ref.importer, _INFERENCE) and _is_module(ref.imported, _RUNTIME):
            violations.append(
                Violation(
                    ref,
                    "ARCH006",
                    "inference must consume the executable SSM surface, not runtime adapters",
                )
            )

    return tuple(violations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root",
        nargs="?",
        type=Path,
        default=Path("src/nof1_causal_lab"),
    )
    args = parser.parse_args()

    violations = find_violations(args.source_root)
    for violation in violations:
        print(violation.diagnostic(args.source_root), file=sys.stderr)
    if violations:
        print(f"{len(violations)} architecture boundary violation(s)", file=sys.stderr)
        return 1

    print("Architecture boundaries: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
