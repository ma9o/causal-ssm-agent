"""Test Stage 1b measurement structure proposal and related assembly helpers."""

import asyncio
import json

import pytest

from nof1_causal_lab.artifacts import CausalDesign
from nof1_causal_lab.flows.stages.stage1b.flow import build_causal_design
from nof1_causal_lab.flows.stages.stage1b.run import (
    Stage1bResult,
    run_stage1b,
)
from nof1_causal_lab.models.ssm.compile.artifact import trial_compile_measurement_structure
from nof1_causal_lab.utils.causal_design import get_outcome_name
from tests.helpers import make_mock_session_factory


def _assert_same_declared_measurement(actual: dict, expected: dict) -> None:
    assert actual["model_clock"] == expected["model_clock"]
    assert [item["name"] for item in actual["indicators"]] == [
        item["name"] for item in expected["indicators"]
    ]
    assert [item["construct_name"] for item in actual["indicators"]] == [
        item["construct_name"] for item in expected["indicators"]
    ]


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Measurement Compiler
# ══════════════════════════════════════════════════════════════════════════════


class TestMeasurementCompiler:
    """Test compiler-level measurement validation used in Stage 1b."""

    def test_valid_measurement_returns_none(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """A valid measurement structure compiles cleanly."""
        result = trial_compile_measurement_structure(
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

        result = trial_compile_measurement_structure(measurement, stage1b_simple_latent)

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

        result = trial_compile_measurement_structure(measurement, stage1b_simple_latent)

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

        result = trial_compile_measurement_structure(measurement, stage1b_simple_latent)

        assert result is not None
        assert "Semantic collision" in result


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Compute functions
# ══════════════════════════════════════════════════════════════════════════════


class TestStage1bGrounding:
    """Test the grounding function directly."""

    def test_build_causal_design_round_trips_schema(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """build_causal_design() returns a valid CausalDesign with the expected outcome."""
        spec = build_causal_design(
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

        validated = CausalDesign.model_validate(spec)
        assert len(validated.latent.constructs) == 2
        assert len(validated.measurement.indicators) == 2
        assert validated.estimation is not None
        assert set(validated.estimation.state_order) == {"Treatment", "Outcome"}
        assert get_outcome_name(spec["latent"]) == "Outcome"

    def test_valid_identifiable(self, stage1b_simple_latent, stage1b_measurement_all_observed):
        """Valid measurement structure returns VALID feedback and stage_output."""
        from nof1_causal_lab.flows.stages.stage1b.grounding import stage1b_grounding

        output, feedback = stage1b_grounding(
            stage1b_measurement_all_observed, stage1b_simple_latent
        )

        assert feedback == "VALID"
        assert output is not None
        _assert_same_declared_measurement(
            output["measurement_structure"],
            stage1b_measurement_all_observed,
        )

    def test_valid_confounded_measurement_still_validates(
        self, stage1b_confounded_latent, stage1b_measurement_all_observed
    ):
        """Identifiability is not checked inside the measurement proposal."""
        from nof1_causal_lab.flows.stages.stage1b.grounding import stage1b_grounding

        output, feedback = stage1b_grounding(
            stage1b_measurement_all_observed, stage1b_confounded_latent
        )

        assert output is not None
        _assert_same_declared_measurement(
            output["measurement_structure"],
            stage1b_measurement_all_observed,
        )
        assert feedback == "VALID"

    def test_lagged_time_varying_query_uses_prior_timestep(self):
        """Lagged X->Y effects are checked as X_{t-1}->Y_t, not X_t->Y_t."""
        from nof1_causal_lab.utils.identifiability import check_identifiability

        latent_structure = {
            "constructs": [
                {
                    "name": "Sleep",
                    "description": "Sleep quality",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "Mood",
                    "description": "Mood state",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                },
                {
                    "name": "Chronotype",
                    "description": "Unobserved stable circadian preference",
                    "role": "exogenous",
                    "temporal_status": "time_invariant",
                },
            ],
            "edges": [
                {
                    "cause": "Sleep",
                    "effect": "Mood",
                    "description": "Better sleep improves later mood",
                    "lagged": True,
                },
                {
                    "cause": "Chronotype",
                    "effect": "Sleep",
                    "description": "Chronotype shifts sleep quality",
                    "lagged": False,
                },
                {
                    "cause": "Chronotype",
                    "effect": "Mood",
                    "description": "Chronotype shifts mood vulnerability",
                    "lagged": False,
                },
            ],
        }
        measurement_structure = {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "sleep_score",
                    "construct_name": "Sleep",
                    "construct_polarity": "positive",
                    "how_to_measure": "Extract the sleep score",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "source_columns": ["sleep_score"],
                },
                {
                    "name": "mood_score",
                    "construct_name": "Mood",
                    "construct_polarity": "positive",
                    "how_to_measure": "Extract the mood score",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "source_columns": ["mood_score"],
                },
            ],
        }

        result = check_identifiability(latent_structure, measurement_structure)
        non_identifiable = result["non_identifiable_treatments"]
        assert non_identifiable["Sleep"]["confounders"] == ["Chronotype"]

    def test_invalid_schema(self, stage1b_simple_latent):
        """Invalid schema returns None stage_output."""
        from nof1_causal_lab.flows.stages.stage1b.grounding import stage1b_grounding

        output, feedback = stage1b_grounding(
            {"indicators": [{"bad": "data"}]}, stage1b_simple_latent
        )

        assert output is None
        assert "VALIDATION ERRORS" in feedback

    def test_drops_unmeasured_constructs_from_estimation_projection(self):
        """Latent-only constructs should not remain in the executable state vector."""
        latent_structure = {
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
        measurement_structure = {
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

        causal_design = build_causal_design(
            latent_structure,
            measurement_structure,
            {
                "identifiable_treatments": {},
                "non_identifiable_treatments": {},
            },
        )

        assert causal_design["estimation"]["state_order"] == ["Treatment", "Outcome"]
        assert causal_design["estimation"]["edges"] == []


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
        factory = make_mock_session_factory([json.dumps(stage1b_measurement_all_observed)])

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_structure=stage1b_simple_latent,
                chunks=stage1b_dummy_chunks,
                session_factory=factory,
            )
        )

        assert isinstance(result, Stage1bResult)
        _assert_same_declared_measurement(
            result.measurement_structure,
            stage1b_measurement_all_observed,
        )

    def test_non_identifiable_still_produces_result(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_all_observed,
        stage1b_dummy_chunks,
    ):
        """Non-identifiable model still produces a result (fat tool captures on structural validity)."""
        factory = make_mock_session_factory([json.dumps(stage1b_measurement_all_observed)])

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_structure=stage1b_confounded_latent,
                chunks=stage1b_dummy_chunks,
                session_factory=factory,
            )
        )

        assert isinstance(result, Stage1bResult)
        _assert_same_declared_measurement(
            result.measurement_structure,
            stage1b_measurement_all_observed,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
