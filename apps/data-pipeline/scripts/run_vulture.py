"""Run vulture with notebooks (.ipynb) converted to .py first.

vulture only scans .py files. This wrapper extracts code cells from every
notebook under ``notebooks/`` into a temporary ``.vulture_cache/`` tree of
``.py`` shadows, then runs vulture against the config paths plus the cache,
then removes the cache.

Three kinds of phantom usage are emitted into the cache so vulture stops
flagging legitimate-but-statically-invisible references:

* Identifiers wrapped in backticks inside notebook markdown cells (e.g.
  ``foo_bar``) — covers documented swap-in hooks.
* Class-body annotated field names on Pydantic models, TypedDicts,
  NamedTuples, Protocols, and ``@dataclass``-decorated classes — covers
  fields read via attribute access that vulture's flow analysis misses.
* Identifiers inside string-quoted type expressions — ``cast("X")``,
  ``Annotated["X", ...]``, forward annotations like ``def foo() -> "X"``.
  Vulture treats strings as opaque, so these references are otherwise
  invisible.

Additionally, vulture's built-in treatment of ``__all__`` entries as "uses" is
disabled (see ``_run_vulture``) so that symbols which are only ever re-exported
from a package ``__init__`` — but never actually referenced — are reported as
dead instead of being kept alive by their ``__all__`` listing.

Usage:
    cd apps/data-pipeline
    uv run python scripts/run_vulture.py                # standard run
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

import nbformat
import vulture.core as vulture_core

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
CACHE_DIR = REPO_ROOT / ".vulture_cache"
PYPROJECT = REPO_ROOT / "pyproject.toml"

BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")
IDENT = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

PYDANTIC_LIKE_BASES = {
    "BaseModel",
    "RootModel",
    "GenericModel",
    "TypedDict",
    "NamedTuple",
    "Protocol",
}
DATACLASS_DECORATORS = {"dataclass", "pydantic_dataclass"}
VALIDATOR_DECORATORS = {
    "field_validator",
    "model_validator",
    "validator",
    "root_validator",
    "computed_field",
}
TYPING_CAST_FUNCS = {"cast", "assert_type", "reveal_type"}


def _extract_backtick_idents(text: str) -> set[str]:
    refs: set[str] = set()
    for span in BACKTICK_SPAN.findall(text):
        refs.update(IDENT.findall(span))
    return refs


def _convert_notebooks() -> set[str]:
    """Convert .ipynb code cells to .py shadows; return markdown identifier refs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    markdown_refs: set[str] = set()
    for nb_path in NOTEBOOKS_DIR.rglob("*.ipynb"):
        out_path = CACHE_DIR / nb_path.relative_to(NOTEBOOKS_DIR).with_suffix(".py")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nb = nbformat.read(nb_path, as_version=4)
        sources: list[str] = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                sources.append(cell.source)
            elif cell.cell_type == "markdown":
                markdown_refs.update(_extract_backtick_idents(cell.source))
        out_path.write_text("\n\n".join(sources))
    return markdown_refs


def _decorator_name(dec: ast.expr) -> str | None:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _base_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Subscript):
        return _base_name(base.value)
    return None


def _has_validator_or_model_config(node: ast.ClassDef) -> bool:
    """Heuristic: Pydantic v2 models have model_config or @field_validator-style methods."""
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "model_config":
                    return True
        elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            for dec in stmt.decorator_list:
                if _decorator_name(dec) in VALIDATOR_DECORATORS:
                    return True
    return False


def _is_data_class_like(node: ast.ClassDef, known: set[str]) -> bool:
    for dec in node.decorator_list:
        if _decorator_name(dec) in DATACLASS_DECORATORS:
            return True
    for base in node.bases:
        name = _base_name(base)
        if name and (name in PYDANTIC_LIKE_BASES or name in known):
            return True
    return _has_validator_or_model_config(node)


def _collect_data_class_fields(paths: list[Path]) -> set[str]:
    """Walk .py files; return annotated field names on Pydantic/dataclass/TypedDict classes."""
    classes_by_file: dict[Path, list[ast.ClassDef]] = {}
    for root in paths:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for py_path in root_path.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            classes_by_file[py_path] = [
                n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
            ]

    known: set[str] = set()
    while True:
        added = False
        for classes in classes_by_file.values():
            for cls in classes:
                if cls.name in known:
                    continue
                if _is_data_class_like(cls, known):
                    known.add(cls.name)
                    added = True
        if not added:
            break

    fields: set[str] = set()
    for classes in classes_by_file.values():
        for cls in classes:
            if cls.name not in known:
                continue
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
    return fields


def _scan_string_type_node(node: ast.AST | None, refs: set[str]) -> None:
    if node is None:
        return
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            refs.update(IDENT.findall(sub.value))


def _collect_string_type_refs(paths: list[Path]) -> set[str]:
    """Extract identifiers from string-quoted type positions (cast/Annotated/forward annotations)."""
    refs: set[str] = set()
    for root in paths:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for py_path in root_path.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    _scan_string_type_node(node.returns, refs)
                    args = node.args
                    for arg in (*args.args, *args.posonlyargs, *args.kwonlyargs):
                        _scan_string_type_node(arg.annotation, refs)
                elif isinstance(node, ast.AnnAssign):
                    _scan_string_type_node(node.annotation, refs)
                elif isinstance(node, ast.Call):
                    func = node.func
                    func_name = None
                    if isinstance(func, ast.Name):
                        func_name = func.id
                    elif isinstance(func, ast.Attribute):
                        func_name = func.attr
                    if func_name in TYPING_CAST_FUNCS and node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Constant) and isinstance(
                            first_arg.value, str
                        ):
                            refs.update(IDENT.findall(first_arg.value))
    return refs


def _write_phantom(filename: str, refs: set[str]) -> None:
    if not refs:
        return
    body = "\n".join(f"_ = {name}" for name in sorted(refs))
    (CACHE_DIR / filename).write_text(body + "\n")


def _collect_top_level_defs(roots: list[str]) -> dict[str, list[tuple[str, int, str]]]:
    """Map name -> [(file, lineno, kind)] for module-level bindings under ``roots``.

    Only direct children of a module are recorded (no nested/local defs), since
    those are what vulture's flat name matching confuses across files. ``kind`` is
    "def" for top-level functions/classes and "assign" for module-level name
    bindings (the latter catches factory assignments like ``foo = make_task(...)``).
    """
    found: dict[str, list[tuple[str, int, str]]] = {}
    for root in roots:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for py_path in root_path.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                targets: list[tuple[str, str]] = []
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    targets.append((node.name, "def"))
                elif isinstance(node, ast.Assign):
                    targets += [(t.id, "assign") for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    targets.append((node.target.id, "assign"))
                for name, kind in targets:
                    found.setdefault(name, []).append((str(py_path), node.lineno, kind))
    return found


def _warn_name_collisions(vulture, defs: dict[str, list[tuple[str, int, str]]]) -> None:
    """Warn about referenced names with module-level definitions in >1 file.

    Vulture's unused check is ``item.name not in used_names`` against one flat set,
    so a single use of a name shields *every* same-named definition across all
    modules. We surface names that (a) are bound at module level in more than one
    file with at least one being a function/class, and (b) are referenced somewhere
    — vulture reports none of them as unused, yet one may be dead. Advisory; this is
    an irreducible consequence of vulture's name matching (vulture #366 / #271).
    """
    flagged = 0
    for name in sorted(defs):
        sites = defs[name]
        files = {f for f, _, _ in sites}
        if len(files) < 2 or name not in vulture.used_names:
            continue
        if not any(kind == "def" for _, _, kind in sites):
            continue  # assign-only collisions (constants) are noisy and rarely API
        flagged += 1
        joined = ", ".join(f"{f}:{ln}" for f, ln, _ in sorted(sites))
        print(
            f"name-collision: '{name}' bound at module level in {len(files)} files "
            f"({joined}) and is referenced — vulture cannot tell which are live, so a "
            f"dead one is masked.",
            file=sys.stderr,
        )
    if flagged:
        print(
            f"\n{flagged} top-level name collision(s) may hide dead code that vulture's "
            f"name matching cannot detect (vulture #366/#271) — review manually.",
            file=sys.stderr,
        )


def _run_vulture(argv: list[str]) -> int:
    """Run vulture in-process with two local adjustments.

    1. ``__all__`` entries are NOT counted as uses, so re-exported-but-unused
       symbols surface (vulture otherwise treats every ``__all__`` name as a use
       via ``core.visit_Assign`` -> ``_assigns_special_variable__all__``).
    2. After the normal report, warn about referenced names defined in more than
       one file (see ``_warn_name_collisions``).

    Replicates ``vulture.core.main`` rather than calling it, so we hold the
    ``Vulture`` instance for the collision pass; config discovery, exit codes, and
    ``--make-whitelist`` output are unchanged. Coupled to vulture internals
    (pinned via ``vulture>=2.14``).
    """
    # setattr (not direct assignment) so vulture doesn't read this monkeypatch as a
    # defined-but-unused module attribute and flag our own override as dead code.
    setattr(vulture_core, "_assigns_special_variable__all__", lambda _node: False)  # noqa: B010
    saved_argv, saved_cwd = sys.argv, Path.cwd()
    sys.argv = ["vulture", *argv]
    os.chdir(REPO_ROOT)
    try:
        try:
            config = vulture_core.make_config()
        except vulture_core.InputError as exc:
            print(exc, file=sys.stderr)
            return int(vulture_core.ExitCode.InvalidCmdlineArguments)
        vulture = vulture_core.Vulture(
            verbose=config["verbose"],
            ignore_names=config["ignore_names"],
            ignore_decorators=config["ignore_decorators"],
        )
        vulture.scavenge(config["paths"], exclude=config["exclude"])
        exit_code = vulture.report(
            min_confidence=config["min_confidence"],
            sort_by_size=config["sort_by_size"],
            make_whitelist=config["make_whitelist"],
        )
        if not config["make_whitelist"]:
            _warn_name_collisions(vulture, _collect_top_level_defs(["src"]))
        return int(exit_code)
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)


def main() -> int:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        markdown_refs = _convert_notebooks()
        _write_phantom("_notebook_markdown_refs.py", markdown_refs)
        config = tomllib.loads(PYPROJECT.read_text())
        paths = config["tool"]["vulture"]["paths"]
        data_class_fields = _collect_data_class_fields(paths)
        _write_phantom("_data_class_field_refs.py", data_class_fields)
        string_type_refs = _collect_string_type_refs(paths)
        _write_phantom("_string_type_refs.py", string_type_refs)
        return _run_vulture([*paths, CACHE_DIR.name, *sys.argv[1:]])
    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
