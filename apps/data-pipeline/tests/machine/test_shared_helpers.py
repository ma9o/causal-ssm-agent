import pytest
from pydantic import BaseModel, ValidationError

from nof1_causal_lab.machine.errors import TransitionExecutionError
from nof1_causal_lab.machine.model_contracts import filter_model_fields, project_model_fields
from nof1_causal_lab.machine.temporal.activity_errors import (
    as_non_retryable_application_error,
)


class _ExampleContract(BaseModel):
    retained: int


class _NestedContract(BaseModel):
    value: int


class _ValidatedExampleContract(BaseModel):
    nested: _NestedContract


def test_filter_model_fields_drops_undeclared_values():
    assert filter_model_fields(_ExampleContract, {"retained": 1, "transient": 2}) == {"retained": 1}


def test_project_model_fields_drops_transients_and_serializes_validated_values():
    assert project_model_fields(
        _ValidatedExampleContract,
        {"nested": {"value": "2"}, "transient": object()},
    ) == {"nested": {"value": 2}}


def test_project_model_fields_rejects_invalid_nested_values():
    with pytest.raises(ValidationError):
        project_model_fields(
            _ValidatedExampleContract,
            {"nested": {"value": "not-an-integer"}},
        )


def test_application_error_preserves_transition_diagnostics():
    error = TransitionExecutionError(
        "model failed",
        transition_id="posterior",
        diagnostics={"reason": "diverged"},
    )

    converted = as_non_retryable_application_error(error)

    assert converted.message == "model failed"
    assert converted.type == "TransitionExecutionError"
    assert converted.non_retryable is True
    assert converted.details == ({"reason": "diverged"},)


def test_application_error_omits_diagnostics_for_untyped_failures():
    converted = as_non_retryable_application_error(ValueError("invalid input"))

    assert converted.message == "invalid input"
    assert converted.type == "ValueError"
    assert converted.non_retryable is True
    assert converted.details == ()
