from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
from nof1_causal_lab.machine.temporal.messages import (
    LLMBackendConfig,
    StatisticalModelSpecAdmissionUnit,
    StatisticalModelSpecFrontierMergeInput,
    StatisticalModelSpecPlan,
)
from nof1_causal_lab.machine.temporal.model_spec_checkpoints import (
    AcceptedConstructCheckpoint,
    read_model_spec_checkpoint,
    write_accepted_model_spec_checkpoint,
    write_initial_model_spec_checkpoint,
)
from nof1_causal_lab.machine.temporal.statistical_model_spec_activities import (
    _barrier_reopen_constructs,
    merge_statistical_model_spec_frontier_activity,
)
from nof1_causal_lab.machine.temporal.statistical_model_spec_workflow import (
    _ready_constructs,
)
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionState,
    CheckResult,
    ConstructContribution,
    build_construct_units,
    validate_full_admission_state,
)
from tests.helpers import make_structural_plan, run_async

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.construct_admission import DesignInfo


def _structural_plan() -> StructuralPlan:
    return StructuralPlan.model_validate(
        make_structural_plan(
            ["A", "B", "X", "Y", "Z"],
            [
                ("A", "X"),
                ("B", "X"),
                ("X", "Y"),
                ("Y", "X"),
                ("Y", "Z"),
            ],
        )
    )


def _plan() -> StatisticalModelSpecPlan:
    units = build_construct_units(_structural_plan())
    return StatisticalModelSpecPlan(
        workspace_id="workspace",
        run_id="seq-000001",
        checkpoint_ref="checkpoint",
        context_ref="context",
        pins={},
        units=[
            StatisticalModelSpecAdmissionUnit(
                unit_id=unit.unit_id,
                constructs=list(unit.constructs),
                predecessors=list(unit.predecessors),
            )
            for unit in units
        ],
        accepted_constructs=[],
        llm=LLMBackendConfig(harness="pi", model="gpt-test"),
        max_tool_turns=10,
        max_attempts_per_construct=4,
    )


def test_ready_frontier_fans_out_and_serializes_feedback_members():
    plan = _plan()

    assert _ready_constructs(plan, set()) == ["A", "B"]
    assert _ready_constructs(plan, {"A"}) == ["B"]
    assert _ready_constructs(plan, {"A", "B"}) == ["X"]
    assert _ready_constructs(plan, {"A", "B", "X"}) == ["Y"]
    assert _ready_constructs(plan, {"A", "B", "X", "Y"}) == ["Z"]


def test_completed_path_can_advance_while_unrelated_root_remains_in_flight():
    plan = StatisticalModelSpecPlan(
        workspace_id="workspace",
        run_id="seq-000001",
        checkpoint_ref="checkpoint",
        context_ref="context",
        pins={},
        units=[
            StatisticalModelSpecAdmissionUnit(unit_id="A", constructs=["A"], predecessors=[]),
            StatisticalModelSpecAdmissionUnit(unit_id="B", constructs=["B"], predecessors=[]),
            StatisticalModelSpecAdmissionUnit(unit_id="C", constructs=["C"], predecessors=["A"]),
        ],
        accepted_constructs=[],
        llm=LLMBackendConfig(harness="pi", model="gpt-test"),
        max_tool_turns=10,
        max_attempts_per_construct=4,
    )

    assert _ready_constructs(plan, set()) == ["A", "B"]
    assert _ready_constructs(plan, {"A"}) == ["B", "C"]


def test_barrier_reopens_failed_feedback_suffix_and_descendants():
    units = build_construct_units(_structural_plan())

    assert _barrier_reopen_constructs(units, ["Y"]) == {"Y", "Z"}
    assert _barrier_reopen_constructs(units, ["X"]) == {"X", "Y", "Z"}
    assert _barrier_reopen_constructs(units, ["A"]) == {"A", "X", "Y", "Z"}


def test_full_barrier_shares_one_exact_simulation(monkeypatch):
    from nof1_causal_lab.models.ssm import construct_admission

    calls: dict[str, Any] = {"compile": 0, "sample": 0, "battery": []}

    def fake_compile(*_args):
        calls["compile"] += 1
        return object(), object()

    def fake_sample(*_args):
        calls["sample"] += 1
        return {"latents": object()}

    def fake_battery(_spec, _pred, _design, target):
        calls["battery"].append(target.name)
        return (
            [CheckResult("C1a finiteness", target.name, "0%", "0%", True, "ok")],
            [],
        )

    monkeypatch.setattr(construct_admission, "_compile_partial", fake_compile)
    monkeypatch.setattr(construct_admission, "_sample_partial", fake_sample)
    monkeypatch.setattr(construct_admission, "_run_battery", fake_battery)
    monkeypatch.setattr(construct_admission.jax, "block_until_ready", lambda value: value)

    validation = validate_full_admission_state(
        AdmissionState(names=("A", "B")),
        (ConstructContribution(name="A"), ConstructContribution(name="B")),
        _structural_plan(),
        cast("DesignInfo", object()),
    )

    assert calls == {"compile": 1, "sample": 1, "battery": ["A", "B"]}
    assert all(report.admitted for report in validation.reports)


def test_frontier_merge_is_single_writer_and_deterministic(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    workspace_id = "parallel-checkpoint"
    initial_ref = write_initial_model_spec_checkpoint(
        workspace_id=workspace_id,
        run_id="seq-000001",
        seq=1,
        pins={},
        accepted_constructs=[],
        search_queries={},
        search_cache={},
        repair_feedback={},
        parent_ref=None,
        rebase=None,
    )
    initial = read_model_spec_checkpoint(workspace_id, initial_ref)

    branch_refs = {}
    for name in ("B", "A"):
        branch_refs[name] = write_accepted_model_spec_checkpoint(
            parent_ref=initial_ref,
            parent=initial,
            accepted=AcceptedConstructCheckpoint(
                submission_id=f"submission-{name}",
                construct_name=name,
                outcome="ADMITTED",
                feedback="accepted",
            ),
            search_queries={},
            search_cache={},
        )

    merged_a = run_async(
        merge_statistical_model_spec_frontier_activity(
            StatisticalModelSpecFrontierMergeInput(
                workspace_id=workspace_id,
                checkpoint_ref=initial_ref,
                branch_checkpoint_refs=[branch_refs["A"]],
                construct_order=["A", "B"],
            )
        )
    )
    merged = run_async(
        merge_statistical_model_spec_frontier_activity(
            StatisticalModelSpecFrontierMergeInput(
                workspace_id=workspace_id,
                checkpoint_ref=merged_a.checkpoint_ref,
                branch_checkpoint_refs=[branch_refs["B"]],
                construct_order=["A", "B"],
            )
        )
    )

    checkpoint = read_model_spec_checkpoint(workspace_id, merged.checkpoint_ref)
    assert checkpoint.parent_ref == merged_a.checkpoint_ref
    assert checkpoint.checkpoint_index == 2
    assert [item.construct_name for item in checkpoint.accepted_constructs] == ["A", "B"]
    assert merged.accepted_constructs == ["A", "B"]
    assert checkpoint.full_model_validated is False
