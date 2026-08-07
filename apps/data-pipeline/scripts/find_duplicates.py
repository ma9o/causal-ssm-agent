"""Rank likely duplicate classes and functions for local agent review.

The audit is deliberately advisory. By default it compares definitions touched
since the merge base with ``origin/master`` against the entire source tree. A
token-clone pass catches copied blocks in Python and TypeScript, while a Python
AST pass catches short alpha-equivalent functions and data models with strongly
overlapping fields.

Usage from the repository root::

    bun run duplicates
    bun run duplicates --deep
    bun run duplicates --all
    bun run duplicates --include-reviewed
    bun run duplicates apps/data-pipeline/src/nof1_causal_lab/models/ssm

Candidates are retrieval results, not claims of semantic equivalence. The
reviewing agent decides whether to consolidate the pair, preserve an intentional
boundary mirror, or leave related implementations separate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOTS = (
    Path("apps/data-pipeline/src"),
    Path("apps/web/src"),
    Path("packages/api-types/src"),
)
PYTHON_SOURCE_ROOT = Path("apps/data-pipeline/src")
SUPPORTED_SUFFIXES = frozenset({".py", ".ts", ".tsx"})
IGNORED_PARTS = frozenset({"generated", "node_modules", "__pycache__", ".next"})
IGNORED_NAME_MARKERS = (".test.", ".spec.", ".stories.")

DEFAULT_LIMIT = 30
DEEP_LIMIT = 60
DEFAULT_CLASS_THRESHOLD = 0.78
DEEP_CLASS_THRESHOLD = 0.68
DEFAULT_FUNCTION_THRESHOLD = 0.82
DEEP_FUNCTION_THRESHOLD = 0.72
DEFAULT_MIN_FUNCTION_NODES = 35
DEEP_MIN_FUNCTION_NODES = 25
REVIEWED_PAIRS_PATH = REPO_ROOT / "apps/data-pipeline/scripts/duplicate_reviews.json"

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
_WORD_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NAME_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "build",
        "by",
        "create",
        "for",
        "from",
        "get",
        "in",
        "is",
        "make",
        "of",
        "on",
        "or",
        "run",
        "set",
        "the",
        "to",
        "with",
    }
)
_UNINFORMATIVE_CALLS = frozenset(
    {
        "abs",
        "all",
        "any",
        "array",
        "asarray",
        "bool",
        "cast",
        "dict",
        "enumerate",
        "float",
        "get",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "items",
        "len",
        "list",
        "max",
        "min",
        "range",
        "set",
        "str",
        "sum",
        "tuple",
        "where",
        "zip",
    }
)
_CONTROL_NODE_NAMES = frozenset(
    {
        "Assert",
        "AsyncFor",
        "AsyncWith",
        "Await",
        "Break",
        "Continue",
        "For",
        "GeneratorExp",
        "If",
        "IfExp",
        "ListComp",
        "Match",
        "Raise",
        "Return",
        "SetComp",
        "Try",
        "TryStar",
        "While",
        "With",
        "Yield",
        "YieldFrom",
    }
)


class DuplicateAuditError(RuntimeError):
    """An operational failure that prevents a trustworthy audit."""


@dataclass(frozen=True, slots=True)
class LineRange:
    """Inclusive source-line range."""

    start: int
    end: int

    def overlaps(self, start: int, end: int) -> bool:
        return self.start <= end and start <= self.end


@dataclass(frozen=True)
class SourceSelection:
    """Definitions that seed comparisons against the full repository."""

    ranges: dict[str, tuple[LineRange, ...]] | None
    description: str

    @property
    def is_all(self) -> bool:
        return self.ranges is None

    @property
    def path_count(self) -> int:
        return len(self.ranges) if self.ranges is not None else 0

    def includes(self, path: str, start: int, end: int) -> bool:
        if self.ranges is None:
            return True
        return any(line_range.overlaps(start, end) for line_range in self.ranges.get(path, ()))


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    annotation: str
    default: str


@dataclass(frozen=True)
class PythonDefinition:
    kind: str
    path: str
    name: str
    qualname: str
    start: int
    end: int
    fields: tuple[FieldSpec, ...] = ()
    methods: frozenset[str] = frozenset()
    bases: frozenset[str] = frozenset()
    fingerprint: str = ""
    node_counts: tuple[tuple[str, int], ...] = ()
    control_counts: tuple[tuple[str, int], ...] = ()
    calls: frozenset[str] = frozenset()
    signature: tuple[str, ...] = ()
    node_count: int = 0
    is_method: bool = False
    parent_class: str | None = None

    @property
    def identity(self) -> tuple[str, int, str]:
        return (self.path, self.start, self.qualname)


@dataclass(frozen=True, slots=True)
class Location:
    path: str
    start: int
    end: int
    label: str = ""

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return (self.path, self.start, self.end, self.label)


@dataclass(frozen=True)
class Candidate:
    category: str
    score: float
    first: Location
    second: Location
    reason: str
    first_selected: bool
    second_selected: bool
    review_fingerprint: str = ""

    @property
    def level(self) -> str:
        return "HIGH" if self.score >= 0.9 else "MEDIUM"

    @property
    def identity(self) -> tuple[str, tuple[str, int, int, str], tuple[str, int, int, str]]:
        first, second = sorted((self.first.identity, self.second.identity))
        return (self.category, first, second)


@dataclass(frozen=True, slots=True)
class ReviewedPair:
    category: str
    first: str
    second: str
    fingerprint: str
    classification: str
    rationale: str

    @property
    def identity(self) -> tuple[str, str, str]:
        first, second = sorted((self.first, self.second))
        return (self.category, first, second)


def _run_command(args: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DuplicateAuditError(f"command failed ({' '.join(args)}): {detail}")
    return completed.stdout


def _git(args: Sequence[str], *, repo_root: Path) -> str:
    return _run_command(("git", *args), cwd=repo_root)


def _is_source_path(path: Path) -> bool:
    if path.suffix not in SUPPORTED_SUFFIXES:
        return False
    if any(part in IGNORED_PARTS for part in path.parts):
        return False
    if any(marker in path.name for marker in IGNORED_NAME_MARKERS):
        return False
    return not (path.suffix == ".py" and path.name.startswith("test_"))


def _is_below_source_root(path: Path) -> bool:
    return any(path == root or path.is_relative_to(root) for root in SOURCE_ROOTS)


def source_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return every production Python/TypeScript source path."""
    paths: list[Path] = []
    for relative_root in SOURCE_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            continue
        paths.extend(
            path.relative_to(repo_root)
            for path in root.rglob("*")
            if path.is_file() and _is_source_path(path.relative_to(repo_root))
        )
    return tuple(sorted(set(paths)))


def parse_changed_ranges(diff_text: str) -> tuple[LineRange, ...]:
    """Extract inclusive new-file ranges from a zero-context unified diff."""
    ranges: list[LineRange] = []
    for line in diff_text.splitlines():
        match = _HUNK_HEADER.match(line)
        if match is None:
            continue
        start = int(match.group("start"))
        count_text = match.group("count")
        count = 1 if count_text is None else int(count_text)
        end = start if count == 0 else start + count - 1
        ranges.append(LineRange(start=max(1, start), end=max(1, end)))
    return tuple(ranges)


def changed_selection(repo_root: Path, *, base_ref: str) -> SourceSelection:
    """Select changed source lines relative to the merge base with ``base_ref``."""
    try:
        merge_base = _git(("merge-base", "HEAD", base_ref), repo_root=repo_root).strip()
    except DuplicateAuditError as exc:
        raise DuplicateAuditError(
            f"cannot resolve merge base with {base_ref!r}; pass --base or use --all"
        ) from exc
    if not merge_base:
        raise DuplicateAuditError(f"git returned no merge base for {base_ref!r}")

    tracked_text = _git(
        ("diff", "--name-only", "--diff-filter=ACMR", merge_base, "--"),
        repo_root=repo_root,
    )
    untracked_text = _git(
        ("ls-files", "--others", "--exclude-standard"),
        repo_root=repo_root,
    )
    tracked = {Path(line) for line in tracked_text.splitlines() if line}
    untracked = {Path(line) for line in untracked_text.splitlines() if line}
    selected: dict[str, tuple[LineRange, ...]] = {}

    for path in sorted(tracked | untracked):
        if not _is_below_source_root(path) or not _is_source_path(path):
            continue
        relative = path.as_posix()
        if path in untracked:
            selected[relative] = (LineRange(1, sys.maxsize),)
            continue
        diff_text = _git(
            ("diff", "--unified=0", "--no-color", merge_base, "--", relative),
            repo_root=repo_root,
        )
        ranges = parse_changed_ranges(diff_text)
        if ranges:
            selected[relative] = ranges

    return SourceSelection(
        ranges=selected,
        description=f"changed definitions since merge base with {base_ref}",
    )


def explicit_selection(repo_root: Path, targets: Sequence[str]) -> SourceSelection:
    """Select all lines in explicit source files or directories."""
    selected: dict[str, tuple[LineRange, ...]] = {}
    resolved_root = repo_root.resolve()
    for target_text in targets:
        target = (repo_root / target_text).resolve()
        try:
            relative_target = target.relative_to(resolved_root)
        except ValueError as exc:
            raise DuplicateAuditError(f"target is outside the repository: {target_text}") from exc
        if not target.exists():
            raise DuplicateAuditError(f"target does not exist: {target_text}")
        candidates = (target,) if target.is_file() else target.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(resolved_root)
            if _is_below_source_root(relative) and _is_source_path(relative):
                selected[relative.as_posix()] = (LineRange(1, sys.maxsize),)
        if target.is_file() and relative_target.as_posix() not in selected:
            raise DuplicateAuditError(f"target is not a production source file: {target_text}")
    return SourceSelection(
        ranges=selected,
        description=f"{len(selected)} explicitly selected source file(s)",
    )


def _terminal_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _terminal_name(node.value)
    return ast.unparse(node)


def _name_words(name: str) -> frozenset[str]:
    snake_case = _WORD_BOUNDARY.sub(r"\1_\2", name).lower()
    return frozenset(
        word
        for word in re.split(r"[^a-z0-9]+", snake_case)
        if len(word) > 1 and word not in _NAME_STOP_WORDS
    )


def _jaccard(left: frozenset[str] | set[str], right: frozenset[str] | set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _cosine(
    left_items: tuple[tuple[str, int], ...],
    right_items: tuple[tuple[str, int], ...],
) -> float:
    left = dict(left_items)
    right = dict(right_items)
    dot = sum(left[key] * right[key] for key in left.keys() & right.keys())
    denominator = math.sqrt(
        sum(value * value for value in left.values())
        * sum(value * value for value in right.values())
    )
    return dot / denominator if denominator else 0.0


def _scope_nodes(root: ast.AST) -> Iterator[ast.AST]:
    """Walk one executable scope without descending into nested definitions."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        if node is not root and isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _function_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.stmt, ...]:
    """Return executable statements, excluding a leading documentation string."""
    body = tuple(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _function_body_nodes(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Walk executable function bodies without counting signatures or annotations."""
    wrapper = ast.Module(body=list(_function_body(node)), type_ignores=[])
    for child in _scope_nodes(wrapper):
        if child is not wrapper:
            yield child


def _is_function_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a definition is an interface or abstract placeholder."""
    body = _function_body(node)
    if not body:
        return True
    if len(body) != 1:
        return False

    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return statement.value.value is Ellipsis
    if isinstance(statement, ast.Return):
        return isinstance(statement.value, ast.Name) and statement.value.id == "NotImplemented"
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return False

    exception = statement.exc.func if isinstance(statement.exc, ast.Call) else statement.exc
    return _terminal_name(exception) == "NotImplementedError"


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    def annotation(arg: ast.arg) -> str:
        return ast.unparse(arg.annotation) if arg.annotation is not None else ""

    args = node.args
    result = [annotation(arg) for arg in args.posonlyargs]
    result.append("/")
    result.extend(annotation(arg) for arg in args.args if arg.arg not in {"self", "cls"})
    if args.vararg is not None:
        result.append(f"*{annotation(args.vararg)}")
    else:
        result.append("*")
    result.extend(annotation(arg) for arg in args.kwonlyargs)
    if args.kwarg is not None:
        result.append(f"**{annotation(args.kwarg)}")
    result.append(f"->{ast.unparse(node.returns) if node.returns is not None else ''}")
    return tuple(result)


def _local_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    ordered: list[str] = []

    def add(name: str) -> None:
        if name not in ordered:
            ordered.append(name)

    args = node.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        add(arg.arg)
    if args.vararg is not None:
        add(args.vararg.arg)
    if args.kwarg is not None:
        add(args.kwarg.arg)
    for child in _scope_nodes(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            add(child.id)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                add(alias.asname or alias.name.split(".", maxsplit=1)[0])
        elif isinstance(child, ast.ExceptHandler) and child.name is not None:
            add(child.name)
    return {name: f"_local_{index}" for index, name in enumerate(ordered)}


def _normalized_ast_value(
    value: object,
    *,
    locals_by_name: dict[str, str],
    root: ast.AST,
) -> object:
    if isinstance(value, ast.Name):
        return (
            "Name",
            locals_by_name.get(value.id, value.id),
            type(value.ctx).__name__,
        )
    if isinstance(value, ast.arg):
        return (
            "arg",
            locals_by_name.get(value.arg, value.arg),
            _normalized_ast_value(value.annotation, locals_by_name=locals_by_name, root=root)
            if value.annotation is not None
            else None,
        )
    if isinstance(value, ast.Constant):
        normalized = (
            value.value
            if value.value is None or isinstance(value.value, bool)
            else type(value.value).__name__
        )
        return ("Constant", normalized)
    if (
        isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and value is not root
    ):
        return (type(value).__name__, "nested-definition")
    if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = value.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        return (
            type(value).__name__,
            _normalized_ast_value(value.args, locals_by_name=locals_by_name, root=root),
            _normalized_ast_value(value.returns, locals_by_name=locals_by_name, root=root),
            _normalized_ast_value(value.decorator_list, locals_by_name=locals_by_name, root=root),
            _normalized_ast_value(body, locals_by_name=locals_by_name, root=root),
        )
    if isinstance(value, ast.AST):
        return (
            type(value).__name__,
            tuple(
                (
                    field,
                    _normalized_ast_value(child, locals_by_name=locals_by_name, root=root),
                )
                for field, child in ast.iter_fields(value)
                if field != "type_comment"
            ),
        )
    if isinstance(value, list):
        return tuple(
            _normalized_ast_value(child, locals_by_name=locals_by_name, root=root)
            for child in value
        )
    return value


def _function_fingerprint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = _normalized_ast_value(node, locals_by_name=_local_names(node), root=node)
    return hashlib.sha256(repr(normalized).encode()).hexdigest()


def _field_default(value: ast.expr | None) -> str:
    return ast.dump(value, include_attributes=False) if value is not None else ""


def _self_fields(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[FieldSpec]:
    for child in _scope_nodes(node):
        targets: Sequence[ast.expr]
        annotation = ""
        value: ast.expr | None
        if isinstance(child, ast.Assign):
            targets = child.targets
            value = child.value
        elif isinstance(child, ast.AnnAssign):
            targets = (child.target,)
            annotation = ast.unparse(child.annotation)
            value = child.value
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                yield FieldSpec(target.attr, annotation, _field_default(value))


def _class_definition(
    node: ast.ClassDef,
    *,
    path: str,
    qualname: str,
) -> PythonDefinition:
    fields: dict[str, FieldSpec] = {}
    methods: set[str] = set()
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            fields[statement.target.id] = FieldSpec(
                statement.target.id,
                ast.unparse(statement.annotation),
                _field_default(statement.value),
            )
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.add(statement.name)
            if statement.name == "__init__":
                for field in _self_fields(statement):
                    fields.setdefault(field.name, field)
    return PythonDefinition(
        kind="class",
        path=path,
        name=node.name,
        qualname=qualname,
        start=node.lineno,
        end=node.end_lineno or node.lineno,
        fields=tuple(sorted(fields.values(), key=lambda field: field.name)),
        methods=frozenset(methods),
        bases=frozenset(_terminal_name(base) for base in node.bases),
    )


def _function_definition(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    path: str,
    qualname: str,
    is_method: bool,
    parent_class: str | None,
) -> PythonDefinition:
    nodes = tuple(_function_body_nodes(node))
    node_counts = Counter(type(child).__name__ for child in nodes)
    control_counts = Counter(
        type(child).__name__ for child in nodes if type(child).__name__ in _CONTROL_NODE_NAMES
    )
    calls = frozenset(
        name
        for child in nodes
        if isinstance(child, ast.Call)
        for name in (_call_name(child.func),)
        if name
    )
    return PythonDefinition(
        kind="function",
        path=path,
        name=node.name,
        qualname=qualname,
        start=node.lineno,
        end=node.end_lineno or node.lineno,
        fingerprint=_function_fingerprint(node),
        node_counts=tuple(sorted(node_counts.items())),
        control_counts=tuple(sorted(control_counts.items())),
        calls=calls,
        signature=_function_signature(node),
        node_count=len(nodes),
        is_method=is_method,
        parent_class=parent_class,
    )


class _DefinitionCollector(ast.NodeVisitor):
    def __init__(self, *, path: str) -> None:
        self.path = path
        self.scope: list[tuple[str, str]] = []
        self.definitions: list[PythonDefinition] = []

    def _qualname(self, name: str) -> str:
        return ".".join((*[scope_name for _, scope_name in self.scope], name))

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.definitions.append(
            _class_definition(node, path=self.path, qualname=self._qualname(node.name))
        )
        self.scope.append(("class", node.name))
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_method = bool(self.scope and self.scope[-1][0] == "class")
        parent_class = self.scope[-1][1] if is_method else None
        if not _is_function_stub(node):
            self.definitions.append(
                _function_definition(
                    node,
                    path=self.path,
                    qualname=self._qualname(node.name),
                    is_method=is_method,
                    parent_class=parent_class,
                )
            )
        self.scope.append(("function", node.name))
        self.generic_visit(node)
        self.scope.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def collect_python_definitions(
    repo_root: Path,
    paths: Iterable[Path],
) -> tuple[PythonDefinition, ...]:
    """Parse Python classes and executable scopes below ``paths``."""
    definitions: list[PythonDefinition] = []
    for relative_path in sorted(path for path in paths if path.suffix == ".py"):
        path = repo_root / relative_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise DuplicateAuditError(f"cannot parse {relative_path}: {exc}") from exc
        collector = _DefinitionCollector(path=relative_path.as_posix())
        collector.visit(tree)
        definitions.extend(collector.definitions)
    return tuple(definitions)


def _ordered_locations(
    first: PythonDefinition,
    second: PythonDefinition,
) -> tuple[Location, Location]:
    first_location = Location(first.path, first.start, first.end, first.qualname)
    second_location = Location(second.path, second.start, second.end, second.qualname)
    if first_location.identity <= second_location.identity:
        return first_location, second_location
    return second_location, first_location


def _definition_content_fingerprint(definition: PythonDefinition) -> str:
    if definition.kind == "function":
        return definition.fingerprint
    payload = (
        tuple((field.name, field.annotation, field.default) for field in definition.fields),
        tuple(sorted(definition.methods)),
        tuple(sorted(definition.bases)),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def _candidate_review_fingerprint(
    first: PythonDefinition,
    second: PythonDefinition,
) -> str:
    """Fingerprint both definition bodies so source edits invalidate reviews."""
    content = sorted(
        (_definition_content_fingerprint(first), _definition_content_fingerprint(second))
    )
    return hashlib.sha256(f"{first.kind}:{content[0]}:{content[1]}".encode()).hexdigest()


def _class_candidate(
    first: PythonDefinition,
    second: PythonDefinition,
    *,
    deep: bool,
    selection: SourceSelection,
) -> Candidate | None:
    first_fields = {field.name: field for field in first.fields}
    second_fields = {field.name: field for field in second.fields}
    shared_names = set(first_fields) & set(second_fields)
    minimum_shared = 2 if deep else 3
    if len(shared_names) < minimum_shared:
        return None
    if first.name in second.bases or second.name in first.bases:
        return None

    names_union = set(first_fields) | set(second_fields)
    field_jaccard = len(shared_names) / len(names_union)
    containment = len(shared_names) / min(len(first_fields), len(second_fields))
    if containment < (0.65 if deep else 0.75):
        return None
    annotation_matches = sum(
        first_fields[name].annotation == second_fields[name].annotation for name in shared_names
    )
    default_matches = sum(
        first_fields[name].default == second_fields[name].default for name in shared_names
    )
    annotation_score = annotation_matches / len(shared_names)
    default_score = default_matches / len(shared_names)
    method_score = _jaccard(first.methods, second.methods)
    name_score = _jaccard(_name_words(first.name), _name_words(second.name))
    score = (
        0.50 * field_jaccard
        + 0.15 * containment
        + 0.15 * annotation_score
        + 0.10 * method_score
        + 0.05 * default_score
        + 0.05 * name_score
    )
    if first.path == second.path and first.bases & second.bases:
        score -= 0.08
    threshold = DEEP_CLASS_THRESHOLD if deep else DEFAULT_CLASS_THRESHOLD
    if score < threshold:
        return None

    location_first, location_second = _ordered_locations(first, second)
    selected_by_identity = {
        (first.path, first.start, first.end, first.qualname): selection.includes(
            first.path, first.start, first.end
        ),
        (second.path, second.start, second.end, second.qualname): selection.includes(
            second.path, second.start, second.end
        ),
    }
    return Candidate(
        category="class schema",
        score=min(1.0, score),
        first=location_first,
        second=location_second,
        reason=(
            f"{len(shared_names)}/{len(names_union)} shared fields; "
            f"{annotation_matches} matching annotations; {default_matches} matching defaults"
        ),
        first_selected=selected_by_identity[location_first.identity],
        second_selected=selected_by_identity[location_second.identity],
        review_fingerprint=_candidate_review_fingerprint(first, second),
    )


def _function_candidate(
    first: PythonDefinition,
    second: PythonDefinition,
    *,
    deep: bool,
    selection: SourceSelection,
) -> Candidate | None:
    minimum_nodes = DEEP_MIN_FUNCTION_NODES if deep else DEFAULT_MIN_FUNCTION_NODES
    if first.node_count < minimum_nodes or second.node_count < minimum_nodes:
        return None
    if first.path == second.path and (
        (first.start <= second.start and second.end <= first.end)
        or (second.start <= first.start and first.end <= second.end)
    ):
        return None

    exact_structure = first.fingerprint == second.fingerprint
    size_ratio = min(first.node_count, second.node_count) / max(first.node_count, second.node_count)
    if not exact_structure and size_ratio < (0.5 if deep else 0.6):
        return None

    first_calls = first.calls - _UNINFORMATIVE_CALLS
    second_calls = second.calls - _UNINFORMATIVE_CALLS
    shared_calls = first_calls & second_calls
    call_score = _jaccard(first_calls, second_calls)
    name_score = _jaccard(_name_words(first.name), _name_words(second.name))
    shape_score = _cosine(first.node_counts, second.node_counts)
    control_score = _cosine(first.control_counts, second.control_counts)
    signature_score = float(first.signature == second.signature)

    if exact_structure:
        score = 0.98
    else:
        if len(shared_calls) < 2 and not (shared_calls and name_score >= 0.5):
            return None
        score = (
            0.40 * call_score
            + 0.25 * shape_score
            + 0.15 * control_score
            + 0.10 * name_score
            + 0.10 * signature_score
        )

    sibling_methods = (
        first.is_method
        and second.is_method
        and first.name == second.name
        and first.parent_class != second.parent_class
    )
    if sibling_methods:
        score -= 0.18
    threshold = DEEP_FUNCTION_THRESHOLD if deep else DEFAULT_FUNCTION_THRESHOLD
    if score < threshold:
        return None

    location_first, location_second = _ordered_locations(first, second)
    selected_by_identity = {
        (first.path, first.start, first.end, first.qualname): selection.includes(
            first.path, first.start, first.end
        ),
        (second.path, second.start, second.end, second.qualname): selection.includes(
            second.path, second.start, second.end
        ),
    }
    if exact_structure:
        reason = "alpha-normalized AST match"
    else:
        reason = (
            f"AST shape {shape_score:.2f}; control flow {control_score:.2f}; "
            f"call overlap {call_score:.2f}"
        )
    if shared_calls:
        reason += f"; shared calls: {', '.join(sorted(shared_calls)[:6])}"
    return Candidate(
        category="function behavior",
        score=min(1.0, score),
        first=location_first,
        second=location_second,
        reason=reason,
        first_selected=selected_by_identity[location_first.identity],
        second_selected=selected_by_identity[location_second.identity],
        review_fingerprint=_candidate_review_fingerprint(first, second),
    )


def ast_candidates(
    definitions: Sequence[PythonDefinition],
    *,
    selection: SourceSelection,
    deep: bool,
) -> tuple[Candidate, ...]:
    """Compare selected Python definitions with all repository definitions."""
    selected = [
        definition
        for definition in definitions
        if selection.includes(definition.path, definition.start, definition.end)
    ]
    by_kind = {
        "class": [definition for definition in definitions if definition.kind == "class"],
        "function": [definition for definition in definitions if definition.kind == "function"],
    }
    exact_function_anchors: dict[str, tuple[str, int, str]] = {}
    for definition in by_kind["function"]:
        anchor = exact_function_anchors.get(definition.fingerprint)
        if anchor is None or definition.identity < anchor:
            exact_function_anchors[definition.fingerprint] = definition.identity
    candidates: dict[
        tuple[str, tuple[str, int, int, str], tuple[str, int, int, str]], Candidate
    ] = {}
    seen_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    for first in selected:
        for second in by_kind[first.kind]:
            if first.identity == second.identity:
                continue
            if (
                first.kind == "function"
                and first.fingerprint == second.fingerprint
                and exact_function_anchors[first.fingerprint]
                not in {first.identity, second.identity}
            ):
                # Report alpha-equivalent groups as a star around one stable
                # representative instead of emitting every O(n²) pair.
                continue
            pair = tuple(sorted((first.identity, second.identity)))
            typed_pair = cast(
                "tuple[tuple[str, int, str], tuple[str, int, str]]",
                pair,
            )
            if typed_pair in seen_pairs:
                continue
            seen_pairs.add(typed_pair)
            if first.kind == "class":
                candidate = _class_candidate(
                    first,
                    second,
                    deep=deep,
                    selection=selection,
                )
            else:
                candidate = _function_candidate(
                    first,
                    second,
                    deep=deep,
                    selection=selection,
                )
            if candidate is not None:
                candidates[candidate.identity] = candidate
    return tuple(sorted(candidates.values(), key=_candidate_sort_key))


def _report_path(
    repo_root: Path,
    *,
    name: str,
    format_name: str,
) -> str:
    raw = Path(name)
    if raw.is_absolute():
        try:
            return raw.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError as exc:
            raise DuplicateAuditError(
                f"jscpd reported a path outside the repository: {name}"
            ) from exc
    roots = (
        (PYTHON_SOURCE_ROOT,)
        if format_name == "python"
        else tuple(root for root in SOURCE_ROOTS if root != PYTHON_SOURCE_ROOT)
    )
    matches = [root / raw for root in roots if (repo_root / root / raw).exists()]
    if len(matches) != 1:
        raise DuplicateAuditError(
            f"cannot uniquely resolve jscpd path {name!r} ({format_name}); matches={matches}"
        )
    return matches[0].as_posix()


def _report_location(
    repo_root: Path,
    raw: object,
    *,
    format_name: str,
) -> Location:
    if not isinstance(raw, dict):
        raise DuplicateAuditError("jscpd report contains a non-object file location")
    name = raw.get("name")
    start = raw.get("start")
    end = raw.get("end")
    if not isinstance(name, str) or not isinstance(start, int) or not isinstance(end, int):
        raise DuplicateAuditError("jscpd report contains an invalid file location")
    return Location(
        path=_report_path(repo_root, name=name, format_name=format_name),
        start=start,
        end=end,
    )


def _python_range_is_import_scaffolding(repo_root: Path, location: Location) -> bool:
    """Return whether a clone range contains only module imports or its docstring."""
    try:
        tree = ast.parse((repo_root / location.path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False

    overlapping = [
        statement
        for statement in tree.body
        if statement.lineno <= location.end
        and (statement.end_lineno or statement.lineno) >= location.start
    ]
    if not overlapping:
        return False

    return all(
        isinstance(statement, (ast.Import, ast.ImportFrom))
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
        for statement in overlapping
    )


def parse_jscpd_report(
    report_path: Path,
    *,
    repo_root: Path,
    selection: SourceSelection,
) -> tuple[Candidate, ...]:
    """Parse and selection-filter a jscpd JSON report."""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DuplicateAuditError(f"cannot read jscpd report {report_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("duplicates"), list):
        raise DuplicateAuditError("jscpd report has no duplicates array")

    candidates: list[Candidate] = []
    for raw_duplicate in payload["duplicates"]:
        if not isinstance(raw_duplicate, dict):
            raise DuplicateAuditError("jscpd report contains a non-object duplicate")
        format_name = raw_duplicate.get("format")
        lines = raw_duplicate.get("lines")
        tokens = raw_duplicate.get("tokens")
        if (
            not isinstance(format_name, str)
            or not isinstance(lines, int)
            or not isinstance(tokens, int)
        ):
            raise DuplicateAuditError("jscpd report contains invalid duplicate metadata")
        first = _report_location(
            repo_root,
            raw_duplicate.get("firstFile"),
            format_name=format_name,
        )
        second = _report_location(
            repo_root,
            raw_duplicate.get("secondFile"),
            format_name=format_name,
        )
        if (
            format_name == "python"
            and _python_range_is_import_scaffolding(repo_root, first)
            and _python_range_is_import_scaffolding(repo_root, second)
        ):
            continue
        first_selected = selection.includes(first.path, first.start, first.end)
        second_selected = selection.includes(second.path, second.start, second.end)
        if not (first_selected or second_selected):
            continue
        score = min(0.99, 0.76 + min(tokens, 240) / 1200 + min(lines, 30) / 300)
        candidates.append(
            Candidate(
                category="token clone",
                score=score,
                first=first,
                second=second,
                reason=f"{lines} duplicated lines; {tokens} duplicated tokens ({format_name})",
                first_selected=first_selected,
                second_selected=second_selected,
            )
        )
    return tuple(sorted(candidates, key=_candidate_sort_key))


def run_jscpd(
    repo_root: Path,
    *,
    selection: SourceSelection,
    deep: bool,
) -> tuple[Candidate, ...]:
    """Run the repository-local jscpd and return selected clone pairs."""
    minimum_tokens = 45 if deep else 65
    minimum_lines = 5 if deep else 7
    with tempfile.TemporaryDirectory(prefix="duplicate-audit-") as output_dir_text:
        output_dir = Path(output_dir_text)
        command = (
            "bunx",
            "jscpd",
            *(root.as_posix() for root in SOURCE_ROOTS),
            "--min-tokens",
            str(minimum_tokens),
            "--min-lines",
            str(minimum_lines),
            "--mode",
            "weak",
            "--format",
            "python,typescript,tsx",
            "--ignore",
            "**/generated/**,**/*.test.*,**/*.spec.*,**/*.stories.*",
            "--reporters",
            "json",
            "--output",
            output_dir_text,
            "--no-colors",
            "--no-tips",
        )
        _run_command(command, cwd=repo_root)
        return parse_jscpd_report(
            output_dir / "jscpd-report.json",
            repo_root=repo_root,
            selection=selection,
        )


def _ranges_overlap(first: Location, second: Location) -> bool:
    return first.path == second.path and first.start <= second.end and second.start <= first.end


def _covered_by_token_clone(candidate: Candidate, token_clones: Sequence[Candidate]) -> bool:
    for clone in token_clones:
        if (
            _ranges_overlap(candidate.first, clone.first)
            and _ranges_overlap(candidate.second, clone.second)
        ) or (
            _ranges_overlap(candidate.first, clone.second)
            and _ranges_overlap(candidate.second, clone.first)
        ):
            return True
    return False


def _candidate_sort_key(candidate: Candidate) -> tuple[float, str, tuple[str, int, int, str]]:
    return (-candidate.score, candidate.category, candidate.first.identity)


def select_candidates(candidates: Sequence[Candidate], *, limit: int) -> tuple[Candidate, ...]:
    """Cap output while reserving space for every populated candidate category."""
    if limit <= 0:
        return ()
    deduplicated = {candidate.identity: candidate for candidate in candidates}
    by_category: dict[str, list[Candidate]] = {}
    for candidate in sorted(deduplicated.values(), key=_candidate_sort_key):
        by_category.setdefault(candidate.category, []).append(candidate)
    if len(deduplicated) <= limit:
        return tuple(sorted(deduplicated.values(), key=_candidate_sort_key))
    if limit < len(by_category):
        return tuple(sorted(deduplicated.values(), key=_candidate_sort_key)[:limit])

    quota = max(1, min(5, limit // max(1, len(by_category))))
    selected: dict[tuple[str, tuple[str, int, int, str], tuple[str, int, int, str]], Candidate] = {}
    for group in by_category.values():
        for candidate in group[:quota]:
            selected[candidate.identity] = candidate
    remainder = [
        candidate for candidate in deduplicated.values() if candidate.identity not in selected
    ]
    for candidate in sorted(remainder, key=_candidate_sort_key):
        if len(selected) >= limit:
            break
        selected[candidate.identity] = candidate
    return tuple(sorted(selected.values(), key=_candidate_sort_key))


def load_reviewed_pairs(path: Path) -> tuple[ReviewedPair, ...]:
    """Load stable semantic-pair classifications from the checked-in registry."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DuplicateAuditError(
            f"cannot load reviewed duplicate pairs from {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
        raise DuplicateAuditError(f"{path} must contain a top-level 'reviews' list")

    reviews: list[ReviewedPair] = []
    identities: set[tuple[str, str, str]] = set()
    for index, record in enumerate(payload["reviews"]):
        if not isinstance(record, dict):
            raise DuplicateAuditError(f"{path}: review {index} must be an object")
        values = tuple(
            record.get(field)
            for field in (
                "category",
                "first",
                "second",
                "fingerprint",
                "classification",
                "rationale",
            )
        )
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise DuplicateAuditError(
                f"{path}: review {index} requires non-empty category, first, second, "
                "fingerprint, classification, and rationale strings"
            )
        category, first, second, fingerprint, classification, rationale = cast(
            "tuple[str, str, str, str, str, str]", values
        )
        if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            raise DuplicateAuditError(
                f"{path}: review {index} fingerprint must be a lowercase SHA-256 digest"
            )
        review = ReviewedPair(category, first, second, fingerprint, classification, rationale)
        if review.identity in identities:
            raise DuplicateAuditError(f"{path}: duplicate reviewed pair at index {index}")
        identities.add(review.identity)
        reviews.append(review)
    return tuple(reviews)


def partition_reviewed_candidates(
    candidates: Sequence[Candidate],
    reviews: Sequence[ReviewedPair],
) -> tuple[tuple[Candidate, ...], tuple[Candidate, ...]]:
    """Separate reviewed semantic definitions from candidates needing attention."""
    reviewed_fingerprints = {review.identity: review.fingerprint for review in reviews}
    pending: list[Candidate] = []
    reviewed: list[Candidate] = []
    for candidate in candidates:
        first = f"{candidate.first.path}::{candidate.first.label}"
        second = f"{candidate.second.path}::{candidate.second.label}"
        identity = (candidate.category, *sorted((first, second)))
        target = (
            reviewed
            if reviewed_fingerprints.get(identity) == candidate.review_fingerprint
            else pending
        )
        target.append(candidate)
    return tuple(pending), tuple(reviewed)


def _format_location(location: Location, *, selected: bool, mark_selected: bool) -> str:
    marker = "*" if selected and mark_selected else " "
    line = (
        str(location.start)
        if location.start == location.end
        else f"{location.start}-{location.end}"
    )
    label = f" {location.label}" if location.label else ""
    return f"  {marker} {location.path}:{line}{label}"


def render_report(
    *,
    selection: SourceSelection,
    candidates: Sequence[Candidate],
    shown: Sequence[Candidate],
    definitions: Sequence[PythonDefinition],
    reviewed_count: int = 0,
) -> str:
    """Render concise output intended for a coding agent."""
    classes = sum(definition.kind == "class" for definition in definitions)
    functions = sum(definition.kind == "function" for definition in definitions)
    lines = [
        "Duplicate audit (advisory)",
        f"Scope: {selection.description}",
        f"Indexed: {classes} Python classes; {functions} Python functions/methods",
    ]
    if reviewed_count:
        lines.append(
            f"Suppressed {reviewed_count} reviewed pair(s); pass --include-reviewed to show them."
        )
    if not candidates:
        lines.append("No likely duplicates touched the selected code.")
        return "\n".join(lines)

    lines.append(f"Showing {len(shown)} of {len(candidates)} candidate pair(s).")
    if not selection.is_all:
        lines.append("* marks a definition or clone range in the selected code.")
    category_titles = {
        "token clone": "Token/near-copy clones",
        "class schema": "Class/schema overlaps",
        "function behavior": "Function behavior overlaps",
    }
    for category in ("token clone", "class schema", "function behavior"):
        group = [candidate for candidate in shown if candidate.category == category]
        if not group:
            continue
        lines.extend(("", category_titles[category]))
        for candidate in group:
            lines.append(f"{candidate.level} {candidate.score:.0%} — {candidate.reason}")
            lines.append(
                _format_location(
                    candidate.first,
                    selected=candidate.first_selected,
                    mark_selected=not selection.is_all,
                )
            )
            lines.append(
                _format_location(
                    candidate.second,
                    selected=candidate.second_selected,
                    mark_selected=not selection.is_all,
                )
            )
    lines.extend(
        (
            "",
            "Review each pair as: consolidate, intentional mirror/parity check, "
            "related implementation, or false positive.",
        )
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        help="source files/directories to compare against the repository instead of using git diff",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="audit every pair in the production source tree",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="lower similarity and clone-size thresholds",
    )
    parser.add_argument(
        "--base",
        default="origin/master",
        help="git ref used to find the default comparison merge base",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=f"maximum pairs to show (default {DEFAULT_LIMIT}, deep {DEEP_LIMIT})",
    )
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="show pairs classified in scripts/duplicate_reviews.json",
    )
    args = parser.parse_args(argv)
    if args.all and args.targets:
        parser.error("--all cannot be combined with explicit targets")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.all:
            selection = SourceSelection(ranges=None, description="all production source")
        elif args.targets:
            selection = explicit_selection(REPO_ROOT, cast("Sequence[str]", args.targets))
        else:
            selection = changed_selection(REPO_ROOT, base_ref=cast("str", args.base))
        if not selection.is_all and not selection.ranges:
            print(
                "Duplicate audit (advisory)\n"
                f"Scope: {selection.description}\n"
                "No changed production source found; pass paths or use --all."
            )
            return 0

        paths = source_paths(REPO_ROOT)
        definitions = collect_python_definitions(REPO_ROOT, paths)
        token_clones = run_jscpd(REPO_ROOT, selection=selection, deep=bool(args.deep))
        semantic = ast_candidates(definitions, selection=selection, deep=bool(args.deep))
        semantic = tuple(
            candidate
            for candidate in semantic
            if candidate.category != "function behavior"
            or not _covered_by_token_clone(candidate, token_clones)
        )
        candidates = tuple(sorted((*token_clones, *semantic), key=_candidate_sort_key))
        reviewed: tuple[Candidate, ...] = ()
        if not args.include_reviewed:
            candidates, reviewed = partition_reviewed_candidates(
                candidates,
                load_reviewed_pairs(REVIEWED_PAIRS_PATH),
            )
        limit = cast("int | None", args.limit) or (DEEP_LIMIT if args.deep else DEFAULT_LIMIT)
        shown = select_candidates(candidates, limit=limit)
        print(
            render_report(
                selection=selection,
                candidates=candidates,
                shown=shown,
                definitions=definitions,
                reviewed_count=len(reviewed),
            )
        )
        return 0
    except DuplicateAuditError as exc:
        print(f"duplicate audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
