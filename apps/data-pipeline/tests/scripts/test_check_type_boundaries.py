"""Tests for the project-specific type-boundary checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_checker() -> Any:
    module_name = "check_type_boundaries_under_test"
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_type_boundaries.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_any_member_makes_union_a_violation_even_inside_generic() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any

def parse(value: tuple[str, Any | None]) -> Any | int:
    ...
""",
        path="src/example.py",
    )

    assert [violation.code for violation in violations] == ["CUSTOM001", "CUSTOM001"]
    assert {violation.annotation for violation in violations} == {
        "Any | None",
        "Any | int",
    }


def test_typing_unions_optional_and_import_aliases_are_checked() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
import typing as t
from typing import Any as Dynamic
from typing import Optional as Maybe
from typing import Union as Either

type Loose = Dynamic
type Looser = Loose

def parse(
    first: Either[str, Dynamic],
    second: t.Optional[t.Any],
    third: Maybe[Looser],
    fourth: Looser | int,
    valid: list[Dynamic] | None,
) -> None:
    ...
""",
        path="src/example.py",
    )

    assert [violation.annotation for violation in violations] == [
        "Either[str, Dynamic]",
        "t.Optional[t.Any]",
        "Maybe[Looser]",
        "Looser | int",
    ]


def test_legacy_explicit_type_alias_value_is_checked() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any, TypeAlias, Union

Loose: TypeAlias = Union[str, Any]
""",
        path="src/example.py",
    )

    assert [violation.code for violation in violations] == ["CUSTOM001"]
    assert violations[0].target == "alias:Loose"
    assert violations[0].annotation == "Union[str, Any]"


def test_rules_can_be_selected_independently() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any

def compile_plan(
    plan: StructuralPlan | dict[str, Any],
    fallback: Any | None,
) -> None:
    ...
""",
        path="tests/example.py",
        rules=frozenset({"CUSTOM001"}),
    )

    assert [violation.code for violation in violations] == ["CUSTOM001"]
    assert violations[0].target == "parameter:fallback"


def test_domain_model_must_not_be_union_member_with_dict() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any, Union
from pydantic import BaseModel

class Indicator(BaseModel):
    name: str

def compile_plan(
    plan: StructuralPlan | dict[str, Any] | None,
    indicator: Indicator | dict[str, Any],
    legacy: Union[Indicator, dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> None:
    ...
        """,
        path="src/example.py",
        rules=frozenset({"CUSTOM002"}),
    )

    assert [violation.code for violation in violations] == [
        "CUSTOM002",
        "CUSTOM002",
        "CUSTOM002",
    ]
    assert {violation.target for violation in violations} == {
        "parameter:indicator",
        "parameter:legacy",
        "parameter:plan",
    }


def test_recursive_json_type_alias_is_not_treated_as_domain_union() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

def register(value: dict | Callable) -> None:
    ...
""",
        path="src/example.py",
    )

    assert violations == []


def test_anonymous_any_dictionary_requires_a_named_boundary() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any

def parse(
    payload: dict[str, Any],
    nested: list[dict[str, Any]] | None,
    indirect: dict[str, tuple[Any, Any]],
) -> dict[str, Any]:
    local: dict[str, Any] = {}
    return local
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM004"}),
    )

    assert [violation.target for violation in violations] == [
        "parameter:payload",
        "parameter:nested",
        "parameter:indirect",
        "return",
        "variable:local",
    ]
    assert all(violation.code == "CUSTOM004" for violation in violations)


def test_unsafe_dictionary_requires_an_explicitly_unsafe_or_domain_name() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any

type JsonObject = dict[str, Any]
type UncheckedJsonObject = dict[str, Any]
type RuntimeMap = dict[str, Any]

def parse(payload: UncheckedJsonObject) -> RuntimeMap:
    return payload
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM004"}),
    )

    assert [violation.target for violation in violations] == ["alias:JsonObject"]


def test_unchecked_json_usage_is_aggregated_into_a_per_file_budget() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from nof1_causal_lab.json_types import UncheckedJsonObject as RawJson

type DomainPayload = RawJson

def parse(payload: RawJson) -> list[RawJson]:
    return [payload]
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM005"}),
    )

    assert len(violations) == 1
    assert violations[0].code == "CUSTOM005"
    assert violations[0].target == "unchecked-json-usage"
    assert violations[0].annotation == "UncheckedJsonObject[3]"


def test_reject_only_optional_parameter_is_checked() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        '''
from typing import Optional as Maybe

def compile_plan(plan: Maybe[StructuralPlan]) -> None:
    """Compile an already validated plan."""
    if plan is None:
        raise ValueError("plan is required")

def compile_spec(spec: StatisticalModelSpec | None = None) -> None:
    if None is spec:
        raise ValueError("spec is required")

def compile_after_audit(plan: StructuralPlan | None) -> None:
    audit_request()
    audit_complete = True
    if plan is None:
        raise ValueError("plan is required")
''',
        path="src/example.py",
    )

    assert [violation.code for violation in violations] == [
        "CUSTOM003",
        "CUSTOM003",
        "CUSTOM003",
    ]
    assert {violation.target for violation in violations} == {
        "parameter:plan",
        "parameter:spec",
    }


def test_conditional_or_non_rejecting_optional_parameter_is_allowed() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
def conditionally_require_plan(
    plan: StructuralPlan | None,
    *,
    enabled: bool,
) -> None:
    if enabled:
        if plan is None:
            raise ValueError("enabled compilation requires a plan")

def default_plan(plan: StructuralPlan | None = None) -> None:
    if plan is None:
        return

def return_before_rejection(plan: StructuralPlan | None, enabled: bool) -> None:
    if not enabled:
        return
    if plan is None:
        raise ValueError("plan is required")

def replace_before_rejection(plan: StructuralPlan | None) -> None:
    if plan is None:
        plan = default_plan()
    if plan is None:
        raise ValueError("plan is required")
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM003"}),
    )

    assert violations == []


def test_baseline_must_match_exact_violation_identity() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        "def compile_plan(plan: StructuralPlan | dict) -> None: ...",
        path="src/example.py",
    )
    identity = violations[0].identity

    assert checker.compare_with_baseline(violations, {identity}) == ([], [])

    new, stale = checker.compare_with_baseline(violations, set())
    assert [violation.identity for violation in new] == [identity]
    assert stale == []

    new, stale = checker.compare_with_baseline([], {identity})
    assert new == []
    assert stale == [identity]
