"""Tests for individual contract models and STAGE_CONTRACTS registry.

Complements test_stage_contracts.py which covers validate_stage_payload
and end-to-end validation. This file focuses on individual model schemas
and the registry itself.
"""

import pytest
from pydantic import BaseModel, ValidationError

from causal_ssm_agent.flows.stages.contracts import (
    STAGE_CONTRACTS,
    DateRangeContract,
    GateOverrideContract,
    LiveMetadata,
    PartialStageResult,
    Stage0Contract,
    WorkerStatusContract,
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
        for stage_id, cls in STAGE_CONTRACTS.items():
            assert issubclass(cls, BaseModel), f"{stage_id} maps to {cls}"

    def test_all_contracts_forbid_extra(self):
        for stage_id, cls in STAGE_CONTRACTS.items():
            assert cls.model_config.get("extra") == "forbid", (
                f"{stage_id} contract does not forbid extra fields"
            )


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


def _stage0_data():
    """Minimal valid Stage0 payload."""
    return {
        "source_type": "google_takeout",
        "source_label": "My Takeout",
        "n_records": 100,
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        "sample": [{"col": "val"}],
    }


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
