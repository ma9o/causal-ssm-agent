"""Tests for the advisory local duplication audit."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_checker() -> Any:
    module_name = "find_duplicates_under_test"
    path = Path(__file__).resolve().parents[2] / "scripts" / "find_duplicates.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_source(repo_root: Path, relative_path: str, source: str) -> Path:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return Path(relative_path)


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_zero_context_diff_ranges_include_added_lines_and_deletion_anchor() -> None:
    checker = _load_checker()

    ranges = checker.parse_changed_ranges(
        """
@@ -4,2 +4,0 @@
@@ -10 +8,3 @@
@@ -20 +21 @@
"""
    )

    assert ranges == (
        checker.LineRange(4, 4),
        checker.LineRange(8, 10),
        checker.LineRange(21, 21),
    )


def test_default_selection_includes_tracked_hunks_and_untracked_sources(tmp_path: Path) -> None:
    checker = _load_checker()
    tracked_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/tracked.py",
        "def transform(value: int) -> int:\n    return value + 1\n",
    )
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "duplicate-audit@example.test")
    _git(tmp_path, "config", "user.name", "Duplicate Audit")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "base")
    _git(tmp_path, "branch", "audit-base")

    _write_source(
        tmp_path,
        tracked_path.as_posix(),
        "def transform(value: int) -> int:\n    return value + 2\n",
    )
    untracked_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/untracked.py",
        "def load() -> int:\n    return 1\n",
    )

    selection = checker.changed_selection(tmp_path, base_ref="audit-base")

    assert selection.includes(tracked_path.as_posix(), 2, 2)
    assert selection.includes(untracked_path.as_posix(), 1, 1000)
    assert selection.path_count == 2


def test_schema_overlap_finds_transport_and_runtime_mirrors(tmp_path: Path) -> None:
    checker = _load_checker()
    runtime_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/runtime.py",
        """
from dataclasses import dataclass

@dataclass
class RuntimeConfig:
    model: str
    timeout: int | None = None
    effort: str | None = None
""",
    )
    transport_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/transport.py",
        """
from typing import Literal
from pydantic import BaseModel

class TransportConfig(BaseModel):
    model: str
    timeout: int | None = None
    effort: Literal["low", "high"] | None = None
""",
    )
    selection = checker.SourceSelection(
        ranges={runtime_path.as_posix(): (checker.LineRange(1, sys.maxsize),)},
        description="runtime config",
    )

    definitions = checker.collect_python_definitions(
        tmp_path,
        (runtime_path, transport_path),
    )
    candidates = checker.ast_candidates(definitions, selection=selection, deep=False)

    schema_candidates = [
        candidate for candidate in candidates if candidate.category == "class schema"
    ]
    assert len(schema_candidates) == 1
    assert {schema_candidates[0].first.label, schema_candidates[0].second.label} == {
        "RuntimeConfig",
        "TransportConfig",
    }
    assert "3/3 shared fields" in schema_candidates[0].reason


def test_function_overlap_normalizes_local_names(tmp_path: Path) -> None:
    checker = _load_checker()
    first_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/first.py",
        """
def encode_failure(error: Exception) -> Envelope:
    if isinstance(error, DomainError):
        return Envelope(
            str(error),
            error.diagnostics,
            kind=type(error).__name__,
            terminal=True,
        )
    return Envelope(str(error), kind=type(error).__name__, terminal=True)
""",
    )
    second_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/second.py",
        """
def convert_failure(exc: Exception) -> Envelope:
    if isinstance(exc, DomainError):
        return Envelope(
            str(exc),
            exc.diagnostics,
            kind=type(exc).__name__,
            terminal=True,
        )
    return Envelope(str(exc), kind=type(exc).__name__, terminal=True)
""",
    )
    selection = checker.SourceSelection(
        ranges={first_path.as_posix(): (checker.LineRange(1, sys.maxsize),)},
        description="first helper",
    )

    definitions = checker.collect_python_definitions(tmp_path, (first_path, second_path))
    candidates = checker.ast_candidates(definitions, selection=selection, deep=False)

    function_candidates = [
        candidate for candidate in candidates if candidate.category == "function behavior"
    ]
    assert len(function_candidates) == 1
    assert function_candidates[0].score == 0.98
    assert function_candidates[0].reason.startswith("alpha-normalized AST match")


def test_similar_shape_with_different_calls_is_not_reported(tmp_path: Path) -> None:
    checker = _load_checker()
    first_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/first.py",
        """
def calculate_alpha(value: float) -> float:
    prepared = prepare_alpha(value)
    if alpha_is_valid(prepared):
        return finish_alpha(prepared)
    return default_alpha(prepared)
""",
    )
    second_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/second.py",
        """
def calculate_beta(value: float) -> float:
    prepared = prepare_beta(value)
    if beta_is_ready(prepared):
        return finish_beta(prepared)
    return default_beta(prepared)
""",
    )
    selection = checker.SourceSelection(
        ranges={first_path.as_posix(): (checker.LineRange(1, sys.maxsize),)},
        description="first helper",
    )

    definitions = checker.collect_python_definitions(tmp_path, (first_path, second_path))

    assert checker.ast_candidates(definitions, selection=selection, deep=True) == ()


def test_alpha_equivalent_groups_use_one_stable_representative(tmp_path: Path) -> None:
    checker = _load_checker()
    paths = tuple(
        _write_source(
            tmp_path,
            f"apps/data-pipeline/src/domain/{name}.py",
            f"""
def {name}(error: Exception) -> Envelope:
    if isinstance(error, DomainError):
        return Envelope(
            str(error),
            error.diagnostics,
            kind=type(error).__name__,
            terminal=True,
        )
    return Envelope(str(error), kind=type(error).__name__, terminal=True)
""",
        )
        for name in ("alpha", "beta", "gamma")
    )
    selection = checker.SourceSelection(ranges=None, description="all")

    definitions = checker.collect_python_definitions(tmp_path, paths)
    candidates = checker.ast_candidates(definitions, selection=selection, deep=False)

    function_candidates = [
        candidate for candidate in candidates if candidate.category == "function behavior"
    ]
    assert len(function_candidates) == 2
    assert all(candidate.first.label == "alpha" for candidate in function_candidates)


def test_function_collection_ignores_interface_and_abstract_stubs(tmp_path: Path) -> None:
    checker = _load_checker()
    source_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/interfaces.py",
        '''
from typing import Protocol

class Reader(Protocol):
    def ellipsis_stub(self, value: int) -> str: ...

    def pass_stub(self, value: int) -> str:
        """Document the interface."""
        pass

    def raise_stub(self, value: int) -> str:
        raise NotImplementedError("implemented by adapters")

    def sentinel_stub(self, value: int) -> str:
        return NotImplemented
''',
    )

    definitions = checker.collect_python_definitions(tmp_path, (source_path,))

    assert [definition.qualname for definition in definitions] == ["Reader"]


def test_function_size_counts_executable_body_not_large_annotations(tmp_path: Path) -> None:
    checker = _load_checker()
    first_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/annotated_first.py",
        """
def first(
    value: tuple[dict[str, list[tuple[int, float, str]]], ...],
    context: dict[str, tuple[list[int], list[float], list[str]]],
    options: tuple[str, int, float, bool, bytes, complex],
) -> tuple[dict[str, list[tuple[int, float, str]]], ...]:
    return value
""",
    )
    second_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/domain/annotated_second.py",
        """
def second(
    item: tuple[dict[str, list[tuple[int, float, str]]], ...],
    metadata: dict[str, tuple[list[int], list[float], list[str]]],
    settings: tuple[str, int, float, bool, bytes, complex],
) -> tuple[dict[str, list[tuple[int, float, str]]], ...]:
    return item
""",
    )
    selection = checker.SourceSelection(ranges=None, description="all")

    definitions = checker.collect_python_definitions(tmp_path, (first_path, second_path))

    assert {definition.node_count for definition in definitions} == {3}
    assert checker.ast_candidates(definitions, selection=selection, deep=True) == ()


def test_jscpd_report_is_filtered_to_selected_clone_ranges(tmp_path: Path) -> None:
    checker = _load_checker()
    first_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/nof1_causal_lab/first.py",
        "\n" * 50,
    )
    second_path = _write_source(
        tmp_path,
        "apps/data-pipeline/src/nof1_causal_lab/second.py",
        "\n" * 50,
    )
    report_path = tmp_path / "jscpd-report.json"
    report_path.write_text(
        json.dumps(
            {
                "duplicates": [
                    {
                        "format": "python",
                        "lines": 8,
                        "tokens": 70,
                        "firstFile": {
                            "name": "nof1_causal_lab/first.py",
                            "start": 10,
                            "end": 17,
                        },
                        "secondFile": {
                            "name": "nof1_causal_lab/second.py",
                            "start": 10,
                            "end": 17,
                        },
                    },
                    {
                        "format": "python",
                        "lines": 8,
                        "tokens": 70,
                        "firstFile": {
                            "name": "nof1_causal_lab/first.py",
                            "start": 30,
                            "end": 37,
                        },
                        "secondFile": {
                            "name": "nof1_causal_lab/second.py",
                            "start": 30,
                            "end": 37,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    selection = checker.SourceSelection(
        ranges={first_path.as_posix(): (checker.LineRange(12, 12),)},
        description="one changed hunk",
    )

    candidates = checker.parse_jscpd_report(
        report_path,
        repo_root=tmp_path,
        selection=selection,
    )

    assert len(candidates) == 1
    assert candidates[0].first.path == first_path.as_posix()
    assert candidates[0].second.path == second_path.as_posix()
    assert candidates[0].first_selected is True
    assert candidates[0].second_selected is False


def test_jscpd_report_ignores_import_only_python_clones(tmp_path: Path) -> None:
    checker = _load_checker()
    source = '''"""Activity module."""

from __future__ import annotations

import json
from package import first, second

VALUE = 1
'''
    _write_source(
        tmp_path,
        "apps/data-pipeline/src/nof1_causal_lab/first.py",
        source,
    )
    _write_source(
        tmp_path,
        "apps/data-pipeline/src/nof1_causal_lab/second.py",
        source,
    )
    report_path = tmp_path / "jscpd-report.json"
    report_path.write_text(
        json.dumps(
            {
                "duplicates": [
                    {
                        "format": "python",
                        "lines": 6,
                        "tokens": 70,
                        "firstFile": {
                            "name": "nof1_causal_lab/first.py",
                            "start": 1,
                            "end": 6,
                        },
                        "secondFile": {
                            "name": "nof1_causal_lab/second.py",
                            "start": 1,
                            "end": 6,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = checker.parse_jscpd_report(
        report_path,
        repo_root=tmp_path,
        selection=checker.SourceSelection(ranges=None, description="all"),
    )

    assert candidates == ()


def test_output_cap_reserves_space_for_each_candidate_category() -> None:
    checker = _load_checker()
    candidates = []
    for category_index, category in enumerate(("token clone", "class schema", "function behavior")):
        for index in range(4):
            candidates.append(
                checker.Candidate(
                    category=category,
                    score=0.99 - category_index * 0.1 - index * 0.01,
                    first=checker.Location(f"first-{category_index}-{index}.py", 1, 5),
                    second=checker.Location(f"second-{category_index}-{index}.py", 1, 5),
                    reason="fixture",
                    first_selected=True,
                    second_selected=False,
                )
            )

    selected = checker.select_candidates(candidates, limit=3)

    assert len(selected) == 3
    assert {candidate.category for candidate in selected} == {
        "token clone",
        "class schema",
        "function behavior",
    }
    assert len(checker.select_candidates(candidates, limit=2)) == 2


def test_reviewed_pairs_are_suppressed_by_stable_definition_identity(tmp_path: Path) -> None:
    checker = _load_checker()
    registry = tmp_path / "duplicate_reviews.json"
    registry.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "category": "function behavior",
                        "first": "src/first.py::first",
                        "second": "src/second.py::second",
                        "fingerprint": "a" * 64,
                        "classification": "related implementation",
                        "rationale": "The shared shape is intentional.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate = checker.Candidate(
        category="function behavior",
        score=0.95,
        first=checker.Location("src/second.py", 20, 30, "second"),
        second=checker.Location("src/first.py", 10, 15, "first"),
        reason="fixture",
        first_selected=True,
        second_selected=False,
        review_fingerprint="a" * 64,
    )

    reviews = checker.load_reviewed_pairs(registry)
    pending, reviewed = checker.partition_reviewed_candidates((candidate,), reviews)

    assert pending == ()
    assert reviewed == (candidate,)

    stale_candidate = dataclasses.replace(candidate, review_fingerprint="b" * 64)
    pending, reviewed = checker.partition_reviewed_candidates((stale_candidate,), reviews)
    assert pending == (stale_candidate,)
    assert reviewed == ()


def test_review_registry_rejects_duplicate_pair_orderings(tmp_path: Path) -> None:
    checker = _load_checker()
    registry = tmp_path / "duplicate_reviews.json"
    common = {
        "category": "class schema",
        "fingerprint": "a" * 64,
        "classification": "intentional mirror/parity check",
        "rationale": "Protocol and model have different responsibilities.",
    }
    registry.write_text(
        json.dumps(
            {
                "reviews": [
                    {**common, "first": "a.py::A", "second": "b.py::B"},
                    {**common, "first": "b.py::B", "second": "a.py::A"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(checker.DuplicateAuditError, match="duplicate reviewed pair"):
        checker.load_reviewed_pairs(registry)
