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
    ExtractionContract,
    GateOverrideContract,
    IndicatorHealthContract,
    InferenceMetadataContract,
    LiveMetadata,
    PartialStageResult,
    PowerScalingResultContract,
    Stage0Contract,
    TreatmentEffectContract,
    ValidationIssueContract,
    ValidationRetryContract,
    WorkerStatusContract,
)

# =============================================================================
# STAGE_CONTRACTS registry
# =============================================================================


class TestStageContracts:
    def test_has_all_stages(self):
        expected = {
            "stage-0",
            "stage-1a",
            "stage-1b",
            "stage-2",
            "stage-3",
            "stage-4",
            "stage-4b",
            "stage-5a",
            "stage-5",
            "stage-6",
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
        "source_label": "My Takeout",
        "n_records": 100,
        "n_columns": 5,
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        "sample": [{"col": "val"}],
        "column_descriptions": [{"name": "col", "dtype": "str", "description": "A column"}],
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


# =============================================================================
# ExtractionContract
# =============================================================================


class TestExtractionContract:
    def test_string_value(self):
        e = ExtractionContract(indicator="mood", value="happy", timestamp="2024-01-01T10:00:00")
        assert e.value == "happy"

    def test_float_value(self):
        e = ExtractionContract(indicator="weight", value=72.5, timestamp="2024-01-01")
        assert e.value == 72.5

    def test_int_value(self):
        e = ExtractionContract(indicator="steps", value=5000, timestamp="2024-01-01")
        assert e.value == 5000

    def test_bool_value(self):
        e = ExtractionContract(indicator="exercised", value=True, timestamp="2024-01-01")
        assert e.value is True

    def test_none_value(self):
        e = ExtractionContract(indicator="mood", value=None, timestamp="2024-01-01")
        assert e.value is None

    def test_none_timestamp(self):
        e = ExtractionContract(indicator="mood", value="good", timestamp=None)
        assert e.timestamp is None

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionContract(indicator="mood", value="good", timestamp="2024-01-01", extra="bad")


# =============================================================================
# ValidationIssueContract
# =============================================================================


class TestValidationIssueContract:
    def test_valid_error(self):
        v = ValidationIssueContract(
            indicator="mood", issue_type="missing_data", severity="error", message="No data found"
        )
        assert v.severity == "error"

    def test_valid_warning(self):
        v = ValidationIssueContract(
            indicator="steps", issue_type="low_variance", severity="warning", message="Low var"
        )
        assert v.severity == "warning"

    def test_valid_info(self):
        v = ValidationIssueContract(
            indicator="hr", issue_type="note", severity="info", message="All good"
        )
        assert v.severity == "info"

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            ValidationIssueContract(
                indicator="mood", issue_type="test", severity="critical", message="test"
            )


# =============================================================================
# IndicatorHealthContract
# =============================================================================


class TestIndicatorHealthContract:
    def test_valid(self):
        h = IndicatorHealthContract(
            indicator="mood",
            n_obs=100,
            variance=2.5,
            time_coverage_ratio=0.9,
            max_gap_ratio=0.1,
            dtype_violations=0,
            duplicate_pct=0.05,
            arithmetic_sequence_detected=False,
            cell_statuses={"coverage": "ok", "variance": "warning"},
        )
        assert h.n_obs == 100
        assert h.cell_statuses["coverage"] == "ok"

    def test_optional_fields_none(self):
        h = IndicatorHealthContract(
            indicator="mood",
            n_obs=50,
            variance=None,
            time_coverage_ratio=None,
            max_gap_ratio=None,
            dtype_violations=0,
            duplicate_pct=0.0,
            arithmetic_sequence_detected=False,
            cell_statuses={},
        )
        assert h.variance is None
        assert h.cell_statuses == {}

    def test_invalid_cell_status(self):
        with pytest.raises(ValidationError):
            IndicatorHealthContract(
                indicator="mood",
                n_obs=50,
                variance=1.0,
                time_coverage_ratio=0.8,
                max_gap_ratio=0.2,
                dtype_violations=0,
                duplicate_pct=0.0,
                arithmetic_sequence_detected=False,
                cell_statuses={"coverage": "invalid_status"},
            )


# =============================================================================
# ValidationRetryContract
# =============================================================================


class TestValidationRetryContract:
    def test_valid(self):
        r = ValidationRetryContract(
            attempt=1, failed_params=["sigma", "beta"], feedback="Fix sigma prior"
        )
        assert r.attempt == 1
        assert len(r.failed_params) == 2

    def test_empty_failed_params(self):
        r = ValidationRetryContract(attempt=2, failed_params=[], feedback="Retry")
        assert r.failed_params == []


# =============================================================================
# PowerScalingResultContract
# =============================================================================


class TestPowerScalingResultContract:
    def test_well_identified(self):
        p = PowerScalingResultContract(
            parameter="sigma",
            diagnosis="well_identified",
            prior_sensitivity=0.1,
            likelihood_sensitivity=0.9,
        )
        assert p.diagnosis == "well_identified"
        assert p.psis_k_hat is None

    def test_prior_dominated_with_psis(self):
        p = PowerScalingResultContract(
            parameter="beta",
            diagnosis="prior_dominated",
            prior_sensitivity=0.8,
            likelihood_sensitivity=0.2,
            psis_k_hat=0.5,
        )
        assert p.psis_k_hat == 0.5

    def test_invalid_diagnosis(self):
        with pytest.raises(ValidationError):
            PowerScalingResultContract(
                parameter="x",
                diagnosis="unknown",
                prior_sensitivity=0.5,
                likelihood_sensitivity=0.5,
            )


# =============================================================================
# TreatmentEffectContract
# =============================================================================


class TestTreatmentEffectContract:
    def test_identifiable(self):
        t = TreatmentEffectContract(
            treatment="exercise",
            effect_size=0.35,
            identifiable=True,
            prob_positive=0.92,
        )
        assert t.treatment == "exercise"
        assert t.identifiable is True

    def test_not_identifiable(self):
        t = TreatmentEffectContract(
            treatment="unknown_tx",
            effect_size=None,
            identifiable=False,
            prior_sensitivity_warning="Not identifiable from data",
        )
        assert t.identifiable is False
        assert t.effect_size is None

    def test_with_optional_fields(self):
        t = TreatmentEffectContract(
            treatment="sleep",
            effect_size=0.2,
            identifiable=True,
            posterior_draws=[0.1, 0.2, 0.3],
            ppc_warnings=["Low coverage"],
            manifest_effects={"mood_rating": 0.15},
        )
        assert len(t.posterior_draws) == 3
        assert t.manifest_effects["mood_rating"] == 0.15


# =============================================================================
# InferenceMetadataContract
# =============================================================================


class TestInferenceMetadataContract:
    def test_valid(self):
        m = InferenceMetadataContract(method="svi", n_samples=1000, duration_seconds=45.2)
        assert m.method == "svi"
        assert m.n_samples == 1000

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            InferenceMetadataContract(
                method="mcmc", n_samples=500, duration_seconds=120.0, extra="bad"
            )
