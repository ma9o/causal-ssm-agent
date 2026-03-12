"""Test Stage 1b: Measurement Model with Identifiability.

Tests the unified flow:
1. Measurement model proposal via fat validation tool
2. Identifiability checking within the tool
3. Marginalization analysis (deterministic post-processing)
"""

import asyncio
import json

import pytest

from causal_ssm_agent.models.ssm_compiler import trial_compile_measurement_model
from causal_ssm_agent.orchestrator.stage1b import (
    Stage1bMessages,
    Stage1bResult,
    run_stage1b,
)
from tests.helpers import make_mock_generate

# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Stage1bMessages
# ══════════════════════════════════════════════════════════════════════════════


class TestStage1bMessages:
    """Test Stage1bMessages builder."""

    def test_proposal_messages(self, stage1b_simple_latent, stage1b_dummy_chunks):
        """Proposal messages include question, latent model, and chunks."""
        msgs = Stage1bMessages(
            question="Does treatment improve outcome?",
            latent_model=stage1b_simple_latent,
            chunks=stage1b_dummy_chunks,
        )

        messages = msgs.proposal_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "treatment improve outcome" in messages[1]["content"]
        assert "Treatment" in messages[1]["content"]


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
            "indicators": [
                indicator
                for indicator in stage1b_measurement_all_observed["indicators"]
                if indicator["construct_name"] != "Outcome"
            ]
        }

        result = trial_compile_measurement_model(measurement, stage1b_simple_latent)

        assert result is not None
        assert "Outcome construct 'Outcome'" in result

    def test_duplicate_operationalization_returns_error(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """Identical indicators for the same construct are rejected."""
        measurement = {
            "indicators": [
                stage1b_measurement_all_observed["indicators"][0],
                {
                    "name": "treatment_dose_copy",
                    "construct_name": "Treatment",
                    "how_to_measure": "Extract the treatment dosage from the data",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                stage1b_measurement_all_observed["indicators"][1],
            ]
        }

        result = trial_compile_measurement_model(measurement, stage1b_simple_latent)

        assert result is not None
        assert "duplicate indicator operationalizations" in result

    def test_semantic_collision_returns_error(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """Compiler surfaces aggregation/measurement semantic mismatches."""
        measurement = {
            "indicators": [
                {
                    "name": "treatment_count",
                    "construct_name": "Treatment",
                    "how_to_measure": "Count the number of treatments administered",
                    "measurement_dtype": "count",
                    "aggregation": "mean",
                },
                stage1b_measurement_all_observed["indicators"][1],
            ]
        }

        result = trial_compile_measurement_model(measurement, stage1b_simple_latent)

        assert result is not None
        assert "Semantic collision" in result


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Compute functions
# ══════════════════════════════════════════════════════════════════════════════


class TestStage1bGrounding:
    """Test the grounding function directly."""

    def test_valid_identifiable(self, stage1b_simple_latent, stage1b_measurement_all_observed):
        """Valid + identifiable returns VALID feedback and stage_output."""
        from causal_ssm_agent.flows.stages.stage_tools import stage1b_grounding

        output, feedback = stage1b_grounding(
            stage1b_measurement_all_observed, stage1b_simple_latent
        )

        assert feedback == "VALID"
        assert output is not None
        assert "causal_spec" in output

    def test_valid_not_identifiable(
        self, stage1b_confounded_latent, stage1b_measurement_missing_confounder
    ):
        """Valid but not identifiable returns stage_output with identifiability feedback."""
        from causal_ssm_agent.flows.stages.stage_tools import stage1b_grounding

        output, feedback = stage1b_grounding(
            stage1b_measurement_missing_confounder, stage1b_confounded_latent
        )

        assert output is not None  # stage_output set even when not identifiable
        assert "causal_spec" in output
        assert feedback != "VALID"
        assert "NOT fully identifiable" in feedback
        assert "proxy" in feedback.lower()

    def test_invalid_schema(self, stage1b_simple_latent):
        """Invalid schema returns None stage_output."""
        from causal_ssm_agent.flows.stages.stage_tools import stage1b_grounding

        output, feedback = stage1b_grounding(
            {"indicators": [{"bad": "data"}]}, stage1b_simple_latent
        )

        assert output is None
        assert "VALIDATION ERRORS" in feedback


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
        assert len(result.identifiability_status["non_identifiable_treatments"]) == 0

    def test_non_identifiable_still_produces_result(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_missing_confounder,
        stage1b_dummy_chunks,
    ):
        """Non-identifiable model still produces a result (fat tool captures on structural validity)."""
        mock_generate = make_mock_generate([json.dumps(stage1b_measurement_missing_confounder)])

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

    def test_identifiable_model_has_clean_status(
        self,
        stage1b_simple_latent,
        stage1b_measurement_all_observed,
        stage1b_dummy_chunks,
    ):
        """Identifiable model has proper identifiability_status structure."""
        mock_generate = make_mock_generate([json.dumps(stage1b_measurement_all_observed)])

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_simple_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        status = result.identifiability_status
        assert "identifiable_treatments" in status
        assert "non_identifiable_treatments" in status
        assert isinstance(status["identifiable_treatments"], dict)
        assert isinstance(status["non_identifiable_treatments"], dict)
        assert len(status["non_identifiable_treatments"]) == 0

    def test_marginalization_analysis_included(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_missing_confounder,
        stage1b_dummy_chunks,
    ):
        """Marginalization analysis is computed and accessible."""
        mock_generate = make_mock_generate([json.dumps(stage1b_measurement_missing_confounder)])

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_confounded_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        assert result.marginalization_analysis is not None
        assert "can_marginalize" in result.marginalization_analysis
        assert "blocking_details" in result.marginalization_analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
