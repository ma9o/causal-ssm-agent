import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nof1_causal_lab.machine.temporal import llm_tool_adapters
from nof1_causal_lab.machine.temporal import model_spec_checkpoints as checkpoints
from nof1_causal_lab.machine.temporal import statistical_model_spec_activities as activities
from nof1_causal_lab.machine.temporal.llm_subroutine_workflow import (
    _executes_model_spec_simulation,
    _openrouter_turn_executes_model_spec_simulation,
)
from nof1_causal_lab.machine.temporal.messages import (
    LLMToolSpec,
    OpenRouterCallResult,
    StatisticalModelSpecAttemptFinalizeInput,
    ToolCallSummary,
)
from tests.helpers import run_async


class _FailingConstructState:
    def __init__(self) -> None:
        self.submission_made = False
        self.last_report = None
        self.current_construct = "early_life_adversity"
        self.search_queries = {}
        self.search_cache = {}
        self.admitted_contributions = {}
        self.n_draws = 200
        self.seed = 0

    def submit_construct(self, **_kwargs):
        self.submission_made = True
        raise ValueError("ordered_logistic requires at least two observed levels")


class _FeedbackConstructState:
    def __init__(self) -> None:
        self.last_report = None
        self.current_construct = "early_life_adversity"
        self.search_queries = {}
        self.search_cache = {}
        self.admitted_contributions = {}
        self.n_draws = 200
        self.seed = 0

    def submit_construct(self, **_kwargs):
        return "Revise the submitted priors."


def test_only_construct_submission_is_routed_to_the_simulation_lane():
    submit = LLMToolSpec(
        name="submit_construct",
        description="submit",
        executor="model_spec_submit_construct",
    )
    search = LLMToolSpec(
        name="search_literature",
        description="search",
        executor="model_spec_search_literature",
    )
    call = OpenRouterCallResult(
        conversation_ref="conversation.json",
        assistant_ref="assistant.json",
        model="test",
        time=0.0,
        completion_preview="",
        tool_calls=[ToolCallSummary(index=0, id="call-1", name="submit_construct")],
    )

    assert _executes_model_spec_simulation(submit)
    assert not _executes_model_spec_simulation(search)
    assert _openrouter_turn_executes_model_spec_simulation(call, [search, submit])


@pytest.mark.parametrize(
    ("state", "expected_feedback", "raises"),
    [
        (
            _FailingConstructState(),
            "ordered_logistic requires at least two observed levels",
            True,
        ),
        (_FeedbackConstructState(), "Revise the submitted priors.", False),
    ],
)
def test_submit_construct_adapter_persists_tool_feedback(
    monkeypatch,
    state,
    expected_feedback,
    raises,
):
    saved = {}
    context = {
        "workspace_id": "ws",
        "checkpoint_ref": "checkpoint:0",
        "attempt": 1,
        "attempt_result_ref": "run/attempt-result.json",
        "search_state_ref": "run/search-state.json",
    }
    monkeypatch.setattr(
        llm_tool_adapters,
        "read_subroutine_json",
        lambda ref: (
            context if ref == "context.json" else {"search_queries": {}, "search_cache": {}}
        ),
    )
    monkeypatch.setattr(
        llm_tool_adapters,
        "write_subroutine_json",
        lambda ref, value: saved.__setitem__(ref, value),
    )
    monkeypatch.setattr(llm_tool_adapters.storage, "exists", lambda _ref: False)
    monkeypatch.setattr(
        checkpoints,
        "load_checkpoint_construct_state",
        lambda *_args, **_kwargs: (
            SimpleNamespace(input_pins={}, accepted_constructs=[]),
            state,
        ),
    )
    monkeypatch.setattr(checkpoints, "existing_accepted_checkpoint_ref", lambda *_args: None)
    args = {
        "construct": "early_life_adversity",
        "indicators": [],
        "priors": {},
    }

    if raises:
        with pytest.raises(ValueError, match="at least two observed levels"):
            llm_tool_adapters._execute_model_spec_submit_construct(
                "context.json", args, "submission-1"
            )
    else:
        llm_tool_adapters._execute_model_spec_submit_construct("context.json", args, "submission-1")

    assert saved["run/attempt-result.json"]["feedback"] == expected_feedback


def test_model_spec_submission_runs_off_the_async_worker_loop(monkeypatch):
    caller_thread = threading.get_ident()
    execution_threads = []

    def fake_submit(_context_ref, _args, _request_id):
        execution_threads.append(threading.get_ident())
        return "submitted", None

    monkeypatch.setattr(
        llm_tool_adapters,
        "_execute_model_spec_submit_construct",
        fake_submit,
    )

    output = run_async(
        llm_tool_adapters.execute_subroutine_tool(
            input=cast("Any", SimpleNamespace(context_ref="context.json")),
            tool=cast("Any", SimpleNamespace(executor="model_spec_submit_construct")),
            args={"construct": "X", "indicators": [], "priors": {}},
            result_ref="result.json",
            request_id="submission-1",
        )
    )

    assert output == ("submitted", None)
    assert execution_threads
    assert execution_threads[0] != caller_thread


def test_admitted_submission_persists_and_returns_the_new_checkpoint(monkeypatch):
    class AdmittedState:
        def __init__(self):
            self.current_construct = "sleep"
            self.search_queries = {}
            self.search_cache = {}
            self.last_report = None
            self.last_coupled_results = ()
            self.admitted_contributions = {}
            self.n_draws = 200
            self.seed = 0

        def submit_construct(self, *, construct, **_kwargs):
            assert construct == "sleep"
            self.current_construct = None
            self.last_report = SimpleNamespace(
                name="sleep",
                admitted=True,
                annotations=(),
                outcome="ADMITTED",
                results=(),
            )
            return "accepted"

    saved = {}
    context = {
        "workspace_id": "ws",
        "checkpoint_ref": "checkpoint:0",
        "attempt": 1,
        "attempt_result_ref": "run/attempt-result.json",
        "search_state_ref": "run/search-state.json",
    }
    monkeypatch.setattr(
        llm_tool_adapters,
        "read_subroutine_json",
        lambda ref: (
            context if ref == "context.json" else {"search_queries": {}, "search_cache": {}}
        ),
    )
    monkeypatch.setattr(
        llm_tool_adapters,
        "write_subroutine_json",
        lambda ref, value: saved.__setitem__(ref, value),
    )
    monkeypatch.setattr(llm_tool_adapters.storage, "exists", lambda _ref: False)
    monkeypatch.setattr(
        checkpoints,
        "load_checkpoint_construct_state",
        lambda *_args, **_kwargs: (
            SimpleNamespace(input_pins={}, accepted_constructs=[]),
            AdmittedState(),
        ),
    )
    monkeypatch.setattr(checkpoints, "existing_accepted_checkpoint_ref", lambda *_args: None)
    monkeypatch.setattr(
        checkpoints,
        "write_accepted_model_spec_checkpoint",
        lambda **_kwargs: "checkpoint:1",
    )

    output = llm_tool_adapters._execute_model_spec_submit_construct(
        "context.json",
        {"construct": "sleep", "indicators": [], "priors": {}},
        "submission-1",
    )

    assert output == ("accepted", None)
    assert saved["run/attempt-result.json"]["admitted"] is True
    assert saved["run/attempt-result.json"]["checkpoint_ref"] == "checkpoint:1"


def test_semantically_identical_submissions_reuse_admission_evaluation(monkeypatch):
    class AdmittedState:
        def __init__(self):
            self.current_construct = "sleep"
            self.search_queries = {}
            self.search_cache = {}
            self.last_report = None
            self.last_coupled_results = ()
            self.admitted_contributions = {}
            self.n_draws = 200
            self.seed = 0

        def submit_construct(self, *, construct, **_kwargs):
            submit_calls.append(construct)
            self.current_construct = None
            self.last_report = SimpleNamespace(
                name=construct,
                admitted=True,
                annotations=(),
                outcome="ADMITTED",
                results=(),
            )
            return "accepted"

    saved = {
        "context.json": {
            "workspace_id": "ws",
            "checkpoint_ref": "checkpoint:0",
            "attempt": 1,
            "attempt_result_ref": "run/attempt-result.json",
            "search_state_ref": "run/search-state.json",
        },
        "run/search-state.json": {"search_queries": {}, "search_cache": {}},
    }
    submit_calls = []
    checkpoint_submissions = []
    parent = SimpleNamespace(input_pins={}, accepted_constructs=[])
    monkeypatch.setattr(llm_tool_adapters, "read_subroutine_json", saved.__getitem__)
    monkeypatch.setattr(
        llm_tool_adapters,
        "write_subroutine_json",
        lambda ref, value: saved.__setitem__(ref, value),
    )
    monkeypatch.setattr(llm_tool_adapters.storage, "exists", lambda ref: ref in saved)
    monkeypatch.setattr(
        checkpoints,
        "load_checkpoint_construct_state",
        lambda *_args, **_kwargs: (parent, AdmittedState()),
    )
    monkeypatch.setattr(checkpoints, "existing_accepted_checkpoint_ref", lambda *_args: None)

    def _write_checkpoint(**kwargs):
        submission_id = kwargs["accepted"].submission_id
        checkpoint_submissions.append(submission_id)
        return f"checkpoint:{submission_id}"

    monkeypatch.setattr(checkpoints, "write_accepted_model_spec_checkpoint", _write_checkpoint)
    args = {"construct": "sleep", "indicators": [], "priors": {}}

    first = llm_tool_adapters._execute_model_spec_submit_construct(
        "context.json", args, "submission-1"
    )
    second = llm_tool_adapters._execute_model_spec_submit_construct(
        "context.json", args, "submission-2"
    )

    assert first == second == ("accepted", None)
    assert submit_calls == ["sleep"]
    assert checkpoint_submissions == ["submission-1", "submission-2"]
    assert saved["run/attempt-result.json"]["checkpoint_ref"] == "checkpoint:submission-2"


def test_finalize_attempt_reads_the_attempt_result(monkeypatch):
    monkeypatch.setattr(activities.storage, "exists", lambda _ref: True)
    monkeypatch.setattr(
        activities,
        "_read_model_spec_json",
        lambda _ref: {
            "submission_id": "submission-1",
            "construct_name": "early_life_adversity",
            "admitted": False,
            "outcome": "ordered_logistic requires at least two observed levels",
            "feedback": "ordered_logistic requires at least two observed levels",
            "checkpoint_ref": None,
            "error": None,
        },
    )

    result = run_async(
        activities.finalize_statistical_model_spec_attempt_activity(
            StatisticalModelSpecAttemptFinalizeInput(
                result_ref="attempt-result.json",
                construct_name="early_life_adversity",
                attempt=1,
            )
        )
    )

    assert result.admitted is False
    assert result.outcome == "ordered_logistic requires at least two observed levels"
