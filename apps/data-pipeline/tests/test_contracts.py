"""Tests for stage payload contracts.

Covers: validate_stage_payload, STAGE_CONTRACTS, individual contract validation.
"""

import pytest
from pydantic import ValidationError

from causal_ssm_agent.flows.stages.contracts import (
    STAGE_CONTRACTS,
    DateRangeContract,
    GateOverrideContract,
    LiveMetadata,
    PartialStageResult,
    Stage0Contract,
    WorkerStatusContract,
    validate_stage_payload,
)

# =============================================================================
# STAGE_CONTRACTS registry
# =============================================================================


class TestStageContracts:
    def test_has_all_nine_stages(self):
        expected = {
            "stage-0", "stage-1a", "stage-1b", "stage-2",
            "stage-3", "stage-4", "stage-4b", "stage-5", "stage-6",
        }
        assert set(STAGE_CONTRACTS.keys()) == expected

    def test_all_values_are_basemodel_classes(self):
        from pydantic import BaseModel

        for stage_id, cls in STAGE_CONTRACTS.items():
            assert issubclass(cls, BaseModel), f"{stage_id} maps to {cls}"


# =============================================================================
# validate_stage_payload
# =============================================================================


def _stage0_data():
    """Minimal valid Stage0 payload."""
    return {
        "source_type": "google_takeout",
        "source_label": "My Takeout",
        "n_records": 100,
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        "sample": [{"col": "val"}],
    }


class TestValidateStagePayload:
    def test_valid_stage0(self):
        result = validate_stage_payload("stage-0", _stage0_data())
        assert result["source_type"] == "google_takeout"
        assert result["n_records"] == 100
        assert result["outcome"] == "success"

    def test_unknown_stage_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown stage_id"):
            validate_stage_payload("stage-99", {})

    def test_invalid_payload_raises_validation_error(self):
        with pytest.raises(ValidationError):
            validate_stage_payload("stage-0", {"bad": "data"})

    def test_extra_fields_forbidden(self):
        data = {**_stage0_data(), "extra_field": "not allowed"}
        with pytest.raises(ValidationError, match="extra"):
            validate_stage_payload("stage-0", data)

    def test_outcome_default(self):
        result = validate_stage_payload("stage-0", _stage0_data())
        assert result["outcome"] == "success"

    def test_outcome_warn(self):
        data = {**_stage0_data(), "outcome": "warn"}
        result = validate_stage_payload("stage-0", data)
        assert result["outcome"] == "warn"

    def test_outcome_fail(self):
        data = {**_stage0_data(), "outcome": "fail"}
        result = validate_stage_payload("stage-0", data)
        assert result["outcome"] == "fail"

    def test_outcome_invalid_value(self):
        data = {**_stage0_data(), "outcome": "invalid"}
        with pytest.raises(ValidationError):
            validate_stage_payload("stage-0", data)

    def test_returns_json_serializable_dict(self):
        """Result should be JSON-serializable (no Pydantic models, datetimes, etc.)."""
        import json

        result = validate_stage_payload("stage-0", _stage0_data())
        # Should not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


# =============================================================================
# DateRangeContract
# =============================================================================


class TestDateRangeContract:
    def test_valid(self):
        dr = DateRangeContract(start="2024-01-01", end="2024-12-31")
        assert dr.start == "2024-01-01"
        assert dr.end == "2024-12-31"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            DateRangeContract(start="2024-01-01")


# =============================================================================
# GateOverrideContract
# =============================================================================


class TestGateOverrideContract:
    def test_valid(self):
        go = GateOverrideContract(reason="testing override")
        assert go.reason == "testing override"

    def test_missing_reason(self):
        with pytest.raises(ValidationError):
            GateOverrideContract()


# =============================================================================
# Stage0Contract
# =============================================================================


class TestStage0Contract:
    def test_valid(self):
        c = Stage0Contract(**_stage0_data())
        assert c.n_records == 100

    def test_sample_with_none_values(self):
        data = {**_stage0_data(), "sample": [{"k": None}]}
        c = Stage0Contract(**data)
        assert c.sample[0]["k"] is None


# =============================================================================
# WorkerStatusContract
# =============================================================================


class TestWorkerStatusContract:
    def test_valid(self):
        ws = WorkerStatusContract(
            worker_id=1,
            status="completed",
            n_extractions=42,
            chunk_size=50,
        )
        assert ws.worker_id == 1
        assert ws.status == "completed"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            WorkerStatusContract(
                worker_id=1,
                status="unknown",
                n_extractions=0,
                chunk_size=50,
            )

    def test_error_field_optional(self):
        ws = WorkerStatusContract(
            worker_id=0,
            status="failed",
            n_extractions=0,
            chunk_size=50,
            error="Something went wrong",
        )
        assert ws.error == "Something went wrong"


# =============================================================================
# LiveMetadata & PartialStageResult
# =============================================================================


class TestLiveMetadata:
    def test_valid(self):
        lm = LiveMetadata(status="running", label="Analyzing", turn=2, elapsed_seconds=5.3)
        assert lm.status == "running"
        assert lm.turn == 2

    def test_status_must_be_running(self):
        with pytest.raises(ValidationError):
            LiveMetadata(status="completed", label="Done", turn=1, elapsed_seconds=1.0)


class TestPartialStageResult:
    def test_valid_with_alias(self):
        data = {
            "llm_trace": {"turns": [], "total_tokens": 100, "model": "test"},
            "_live": {"status": "running", "label": "Working", "turn": 1, "elapsed_seconds": 0.5},
        }
        psr = PartialStageResult.model_validate(data)
        assert psr.live.status == "running"
