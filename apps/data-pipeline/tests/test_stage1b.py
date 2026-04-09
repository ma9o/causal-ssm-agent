"""Test Stage 1b: Measurement Model with Identifiability.

Tests the unified flow:
1. Measurement model proposal via fat validation tool
2. Identifiability checking within the tool
3. Marginalization analysis (deterministic post-processing)
"""

import asyncio
import json

import pytest

from causal_ssm_agent.artifacts import CausalSpec
from causal_ssm_agent.flows.stages.stage1b.flow import build_causal_spec
from causal_ssm_agent.flows.stages.stage1b.run import (
    Stage1bResult,
    run_stage1b,
)
from causal_ssm_agent.models.ssm_compiler import trial_compile_measurement_model
from causal_ssm_agent.utils.causal_spec import get_outcome_name
from tests.helpers import make_mock_generate

# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Measurement Compiler
# ══════════════════════════════════════════════════════════════════════════════


class TestMeasurementCompiler:
    """Test compiler-level measurement validation used in Stage 1b."""

    def test_valid_measurement_returns_none(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """A valid measurement model compiles cleanly."""
        result = trial_compile_measurement_model(
            stage1b_measurement_all_observed, stage1b_simple_latent
        )
        assert result is None

    def test_missing_outcome_indicator_returns_error(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """Outcome coverage is enforced at compile time."""
        measurement = {
            "model_clock": "1d",
            "indicators": [
                indicator
                for indicator in stage1b_measurement_all_observed["indicators"]
                if indicator["construct_name"] != "Outcome"
            ],
        }

        result = trial_compile_measurement_model(measurement, stage1b_simple_latent)

        assert result is not None
        assert "Outcome construct 'Outcome'" in result

    def test_duplicate_operationalization_returns_error(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """Identical indicators for the same construct are rejected."""
        measurement = {
            "model_clock": "1d",
            "indicators": [
                stage1b_measurement_all_observed["indicators"][0],
                {
                    "name": "treatment_dose_copy",
                    "construct_name": "Treatment",
                    "construct_polarity": "positive",
                    "how_to_measure": "Extract the treatment dosage from the data",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                stage1b_measurement_all_observed["indicators"][1],
            ],
        }

        result = trial_compile_measurement_model(measurement, stage1b_simple_latent)

        assert result is not None
        assert "duplicate indicator operationalizations" in result

    def test_semantic_collision_returns_error(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """Compiler surfaces aggregation/measurement semantic mismatches."""
        measurement = {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "treatment_count",
                    "construct_name": "Treatment",
                    "construct_polarity": "positive",
                    "how_to_measure": "Count the number of treatments administered",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                stage1b_measurement_all_observed["indicators"][1],
            ],
        }

        result = trial_compile_measurement_model(measurement, stage1b_simple_latent)

        assert result is not None
        assert "Semantic collision" in result


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Compute functions
# ══════════════════════════════════════════════════════════════════════════════


class TestStage1bGrounding:
    """Test the grounding function directly."""

    def test_build_causal_spec_round_trips_schema(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """build_causal_spec.fn() returns a valid CausalSpec with the expected outcome."""
        spec = build_causal_spec.fn(
            stage1b_simple_latent,
            stage1b_measurement_all_observed,
            identifiability_status={
                "identifiable_treatments": {
                    "Treatment": {
                        "method": "do_calculus",
                        "estimand": "P(Outcome|do(Treatment))",
                        "marginalized_confounders": [],
                        "instruments": [],
                    }
                },
                "non_identifiable_treatments": {},
            },
        )

        validated = CausalSpec.model_validate(spec)
        assert len(validated.latent.constructs) == 2
        assert len(validated.measurement.indicators) == 2
        assert validated.estimation is not None
        assert set(validated.estimation.state_order) == {"Treatment", "Outcome"}
        assert get_outcome_name(spec["latent"]) == "Outcome"

    def test_valid_identifiable(self, stage1b_simple_latent, stage1b_measurement_all_observed):
        """Valid + identifiable returns VALID feedback and stage_output."""
        from causal_ssm_agent.flows.stages.stage1b.grounding import stage1b_grounding

        output, feedback = stage1b_grounding(
            stage1b_measurement_all_observed, stage1b_simple_latent
        )

        assert feedback == "VALID"
        assert output is not None
        assert "causal_spec" in output

    def test_valid_not_identifiable(
        self, stage1b_confounded_latent, stage1b_measurement_all_observed
    ):
        """Valid but not identifiable returns stage_output with identifiability feedback."""
        from causal_ssm_agent.flows.stages.stage1b.grounding import stage1b_grounding

        output, feedback = stage1b_grounding(
            stage1b_measurement_all_observed, stage1b_confounded_latent
        )

        assert output is not None  # stage_output set even when not identifiable
        assert "causal_spec" in output
        assert output["causal_spec"]["estimation"]["state_order"] == ["Treatment", "Outcome"]
        assert feedback != "VALID"
        assert "NOT fully identifiable" in feedback
        assert "proxy" in feedback.lower()

    def test_invalid_schema(self, stage1b_simple_latent):
        """Invalid schema returns None stage_output."""
        from causal_ssm_agent.flows.stages.stage1b.grounding import stage1b_grounding

        output, feedback = stage1b_grounding(
            {"indicators": [{"bad": "data"}]}, stage1b_simple_latent
        )

        assert output is None
        assert "VALIDATION ERRORS" in feedback

    def test_drops_unmeasured_constructs_from_estimation_projection(self, monkeypatch):
        """Latent-only constructs should not remain in the executable state vector."""
        from causal_ssm_agent.flows.stages.stage1b.grounding import stage1b_grounding

        latent_model = {
            "constructs": [
                {
                    "name": "Treatment",
                    "description": "Observed treatment",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "Mediator",
                    "description": "Unmeasured mediator",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "Outcome",
                    "description": "Observed outcome",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            "edges": [
                {
                    "cause": "Treatment",
                    "effect": "Mediator",
                    "description": "Treatment shifts mediator",
                },
                {
                    "cause": "Mediator",
                    "effect": "Outcome",
                    "description": "Mediator shifts outcome",
                },
            ],
        }
        measurement_model = {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "treatment_signal",
                    "construct_name": "Treatment",
                    "construct_polarity": "positive",
                    "how_to_measure": "Use the treatment column directly",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "outcome_signal",
                    "construct_name": "Outcome",
                    "construct_polarity": "positive",
                    "how_to_measure": "Use the outcome column directly",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
        }

        monkeypatch.setattr(
            "causal_ssm_agent.utils.identifiability.check_identifiability",
            lambda *_args, **_kwargs: {
                "identifiable_treatments": {},
                "non_identifiable_treatments": {},
            },
        )

        output, feedback = stage1b_grounding(measurement_model, latent_model)

        assert output is not None
        assert feedback == "VALID"
        assert output["causal_spec"]["estimation"]["state_order"] == ["Treatment", "Outcome"]
        assert output["causal_spec"]["estimation"]["edges"] == []


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS: Full Stage 1b Flow
# ══════════════════════════════════════════════════════════════════════════════


class TestStage1bFlow:
    """Integration tests for the full Stage 1b flow."""

    def test_all_identifiable(
        self,
        stage1b_simple_latent,
        stage1b_measurement_all_observed,
        stage1b_dummy_chunks,
    ):
        """When all effects are identifiable, result has empty non_identifiable_treatments."""
        mock_generate = make_mock_generate([json.dumps(stage1b_measurement_all_observed)])

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_simple_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        assert isinstance(result, Stage1bResult)
        status = result.identifiability_status
        assert isinstance(status["identifiable_treatments"], dict)
        assert isinstance(status["non_identifiable_treatments"], dict)
        assert len(status["non_identifiable_treatments"]) == 0

    def test_non_identifiable_still_produces_result(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_all_observed,
        stage1b_dummy_chunks,
    ):
        """Non-identifiable model still produces a result (fat tool captures on structural validity)."""
        mock_generate = make_mock_generate([json.dumps(stage1b_measurement_all_observed)])

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_confounded_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        assert isinstance(result, Stage1bResult)
        assert len(result.identifiability_status["non_identifiable_treatments"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
