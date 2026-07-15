from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from nof1_causal_lab.machine.moves import RunArtifact
from nof1_causal_lab.machine.store import EpisodeJournal, TransitionRecord
from nof1_causal_lab.machine.temporal.model_spec_checkpoints import (
    AcceptedConstructCheckpoint,
    ModelSpecAdmissionEvaluation,
    ModelSpecCheckpoint,
    latest_failed_model_spec_checkpoint_ref,
    model_spec_admission_evaluation_key,
    model_spec_admission_evaluation_path,
    read_model_spec_admission_evaluation,
    read_model_spec_checkpoint,
    rebase_accepted_constructs,
    restore_construct_state,
    write_accepted_model_spec_checkpoint,
    write_initial_model_spec_checkpoint,
    write_model_spec_admission_evaluation,
)

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId


def _workspace(monkeypatch, tmp_path) -> str:
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    return "checkpoint-test"


def test_accepted_checkpoint_is_immutable_and_idempotent(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    pins: dict[ArtifactId, int] = {
        "question": 1,
        "causal_design": 2,
        "identification_report": 2,
        "panel": 3,
        "validation_report": 3,
    }
    initial_ref = write_initial_model_spec_checkpoint(
        workspace_id=workspace_id,
        run_id="seq-000004",
        seq=4,
        pins=pins,
        accepted_constructs=[],
        search_queries={},
        search_cache={},
        repair_feedback={},
        parent_ref=None,
        rebase=None,
    )
    initial = read_model_spec_checkpoint(workspace_id, initial_ref)
    accepted = AcceptedConstructCheckpoint(
        submission_id="tool-call-1",
        construct_name="sleep",
        indicators=[],
        priors={"rho_sleep": {"distribution": "Beta", "params": {"alpha": 5, "beta": 2}}},
        results=[
            {
                "check": "C2 latent scale",
                "target": "sleep",
                "value": "median scale 1.0",
                "band": "[0.33, 3.0]",
                "passed": True,
                "note": "ok",
                "diagnosis": [],
                "mode": "soft",
            }
        ],
        outcome="ADMITTED",
        feedback="accepted",
    )

    first_ref = write_accepted_model_spec_checkpoint(
        parent_ref=initial_ref,
        parent=initial,
        accepted=accepted,
        search_queries={},
        search_cache={},
    )
    retry_ref = write_accepted_model_spec_checkpoint(
        parent_ref=initial_ref,
        parent=initial,
        accepted=accepted,
        search_queries={},
        search_cache={},
    )

    assert retry_ref == first_ref
    checkpoint = read_model_spec_checkpoint(workspace_id, first_ref)
    assert checkpoint.parent_ref == initial_ref
    assert checkpoint.checkpoint_index == 1
    assert checkpoint.input_pins == pins
    assert [item.construct_name for item in checkpoint.accepted_constructs] == ["sleep"]
    assert checkpoint.accepted_constructs[0].results[0]["target"] == "sleep"


def test_admission_evaluation_key_is_scoped_to_causal_ancestors(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    accepted_a = AcceptedConstructCheckpoint(
        submission_id="submission-a",
        construct_name="A",
        priors={"rho_A": {"distribution": "Normal", "params": {"mu": 0.2}}},
        outcome="ADMITTED",
        feedback="accepted",
    )
    accepted_b = AcceptedConstructCheckpoint(
        submission_id="submission-b",
        construct_name="B",
        priors={"rho_B": {"distribution": "Normal", "params": {"mu": 0.3}}},
        outcome="ADMITTED",
        feedback="accepted",
    )
    checkpoint = ModelSpecCheckpoint(
        workspace_id=workspace_id,
        run_id="seq-000001",
        seq=1,
        checkpoint_index=2,
        input_pins={"causal_design": 2, "panel": 3},
        accepted_constructs=[accepted_a, accepted_b],
        created_at="2026-07-13T00:00:00+00:00",
    )
    proposal = {
        "ancestor_constructs": {"A"},
        "construct_name": "X",
        "indicators": [],
        "priors": {"rho_X": {"distribution": "Normal", "params": {"mu": 0.5}}},
        "accept": [],
        "n_draws": 200,
        "seed": 0,
    }

    key = model_spec_admission_evaluation_key(checkpoint, **proposal)
    changed_sibling = checkpoint.model_copy(
        update={
            "accepted_constructs": [
                accepted_a,
                accepted_b.model_copy(
                    update={
                        "priors": {
                            "rho_B": {
                                "distribution": "Normal",
                                "params": {"mu": 99.0},
                            }
                        }
                    }
                ),
            ]
        }
    )
    changed_ancestor = checkpoint.model_copy(
        update={
            "accepted_constructs": [
                accepted_a.model_copy(
                    update={
                        "priors": {
                            "rho_A": {
                                "distribution": "Normal",
                                "params": {"mu": 99.0},
                            }
                        }
                    }
                ),
                accepted_b,
            ]
        }
    )

    assert model_spec_admission_evaluation_key(changed_sibling, **proposal) == key
    assert model_spec_admission_evaluation_key(changed_ancestor, **proposal) != key

    path = model_spec_admission_evaluation_path(workspace_id, key)
    evaluation = ModelSpecAdmissionEvaluation(
        evaluation_key=key,
        construct_name="X",
        admitted=False,
        outcome="NEEDS REVISION",
        feedback="revise",
    )
    write_model_spec_admission_evaluation(path, evaluation)
    write_model_spec_admission_evaluation(path, evaluation)
    assert read_model_spec_admission_evaluation(path) == evaluation


def test_latest_failed_stage_four_checkpoint_is_the_resume_source(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    journal = EpisodeJournal(workspace_id)
    journal.append(
        TransitionRecord(
            seq=1,
            ts="2026-07-11T00:00:00+00:00",
            move=RunArtifact(artifact_id="statistical_model_spec"),
            status="raised",
            diagnostics={"checkpoint_ref": "model-spec-checkpoint:checkpoint-test/run/old.json"},
        )
    )
    journal.append(
        TransitionRecord(
            seq=2,
            ts="2026-07-11T00:01:00+00:00",
            move=RunArtifact(artifact_id="posterior"),
            status="raised",
        )
    )
    journal.append(
        TransitionRecord(
            seq=3,
            ts="2026-07-11T00:02:00+00:00",
            move=RunArtifact(artifact_id="statistical_model_spec"),
            status="raised",
            diagnostics={"checkpoint_ref": "model-spec-checkpoint:checkpoint-test/run/new.json"},
        )
    )

    assert (
        latest_failed_model_spec_checkpoint_ref(workspace_id)
        == "model-spec-checkpoint:checkpoint-test/run/new.json"
    )

    journal.append(
        TransitionRecord(
            seq=4,
            ts="2026-07-11T00:03:00+00:00",
            move=RunArtifact(artifact_id="statistical_model_spec"),
            status="applied",
        )
    )
    assert latest_failed_model_spec_checkpoint_ref(workspace_id) is None


def test_target_restore_uses_only_its_causal_ancestor_closure(monkeypatch):
    from nof1_causal_lab.flows.transitions.model_spec.agentic import construct_flow
    from nof1_causal_lab.models.ssm import construct_admission
    from nof1_causal_lab.models.ssm.construct_admission import AdmissionState

    class FakeState:
        def __init__(self, *, order, **_kwargs):
            self.order = order
            self.cursor = 0
            self.catalog = object()
            self.admission = AdmissionState()
            self.admitted_contributions = {}
            self.search_queries = {}
            self.search_cache = {}
            self.last_tool_feedback = None

        @property
        def current_construct(self):
            return self.order[self.cursor]

    monkeypatch.setattr(construct_flow, "ConstructBuildState", FakeState)
    monkeypatch.setattr(
        construct_flow,
        "contribution_from_payload",
        lambda _design, payload, _catalog: payload["construct"],
    )
    monkeypatch.setattr(
        construct_admission,
        "trial_admission_state",
        lambda admission, _contribution: admission,
    )
    checkpoint = ModelSpecCheckpoint(
        workspace_id="workspace",
        run_id="seq-000001",
        seq=1,
        checkpoint_index=2,
        input_pins={},
        accepted_constructs=[
            AcceptedConstructCheckpoint(
                submission_id=f"submission-{name}",
                construct_name=name,
                outcome="ADMITTED",
                feedback="accepted",
            )
            for name in ("A", "B")
        ],
        created_at="2026-07-11T00:00:00+00:00",
    )
    causal_design = {
        "estimation": {
            "state_order": ["A", "B", "X"],
            "edges": [{"cause": "A", "effect": "X"}],
        }
    }

    state = restore_construct_state(
        checkpoint,
        causal_design=causal_design,
        data_for_model=object(),
        workspace_id=None,
        target_construct="X",
    )

    assert state.order == ["A", "X"]
    assert set(state.admitted_contributions) == {"A"}


def test_rebase_retains_independent_branch_and_reopens_failed_descendants(monkeypatch):
    from nof1_causal_lab.machine.temporal import model_spec_checkpoints
    from nof1_causal_lab.machine.temporal.model_spec_checkpoints import ModelSpecCheckpoint
    from nof1_causal_lab.models.ssm import construct_admission

    class FakeState:
        def __init__(self, target):
            self.target = target
            self.search_queries = {}
            self.search_cache = {}
            self.last_report = None
            self.attempt = 0
            self.submission_made = False

        @property
        def current_construct(self):
            return self.target

        def submit_construct(self, *, construct, **_kwargs):
            if construct == "stress":
                return "stress no longer passes the scale check"
            self.target = None
            self.last_report = SimpleNamespace(
                name=construct,
                admitted=True,
                annotations=(),
                outcome="ADMITTED",
            )
            return "accepted"

    monkeypatch.setattr(
        construct_admission,
        "build_construct_order",
        lambda _causal_design: ["stress", "sleep", "mood"],
    )
    monkeypatch.setattr(
        construct_admission,
        "build_construct_units",
        lambda _causal_design: [
            SimpleNamespace(unit_id="stress", constructs=("stress",), predecessors=()),
            SimpleNamespace(unit_id="sleep", constructs=("sleep",), predecessors=()),
            SimpleNamespace(unit_id="mood", constructs=("mood",), predecessors=("stress",)),
        ],
    )
    monkeypatch.setattr(
        model_spec_checkpoints,
        "restore_construct_state",
        lambda _checkpoint, **kwargs: FakeState(kwargs.get("target_construct")),
    )
    accepted = [
        AcceptedConstructCheckpoint(
            submission_id=f"submission-{name}",
            construct_name=name,
            outcome="ADMITTED",
            feedback="accepted",
        )
        for name in ("stress", "sleep", "mood")
    ]
    source = ModelSpecCheckpoint(
        workspace_id="workspace",
        run_id="seq-000001",
        seq=1,
        checkpoint_index=3,
        input_pins={
            "question": 1,
            "causal_design": 1,
            "identification_report": 1,
            "panel": 1,
            "validation_report": 1,
        },
        accepted_constructs=accepted,
        created_at="2026-07-11T00:00:00+00:00",
    )

    _state, retained, reopened, reason = rebase_accepted_constructs(
        source,
        causal_design={},
        data_for_model=object(),
    )

    assert [item.construct_name for item in retained] == ["sleep"]
    assert reopened == "stress"
    assert reason == "stress no longer passes the scale check"
