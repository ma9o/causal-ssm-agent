from __future__ import annotations

from nof1_causal_lab.flows.runtime_events import (
    ModelSpecAdmissionEvent,
    TransitionRuntimeEvent,
    emit_model_spec_admission_event,
    emit_transition_event,
    read_events,
)


def test_runtime_events_roundtrip_as_discriminated_models(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    workspace_id = "runtime-event-contracts"

    emit_transition_event(
        workspace_id,
        "statistical_model_spec",
        "failed",
        error={"type": "ValidationError", "message": "invalid prior"},
    )
    emit_model_spec_admission_event(
        workspace_id,
        "barrier_report",
        {"round": 2, "status": "ready"},
    )

    events = read_events(workspace_id)
    assert len(events) == 2
    assert isinstance(events[0], TransitionRuntimeEvent)
    assert events[0].payload.error is not None
    assert events[0].payload.error.type == "ValidationError"
    assert isinstance(events[1], ModelSpecAdmissionEvent)
    assert events[1].payload["context_id"] == "statistical-model-spec"
    assert events[1].event == "nof1-causal-lab.model-spec.admission.barrier_report"
