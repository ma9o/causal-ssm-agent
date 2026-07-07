"""Tests for schema validation functions that collect all errors.

Covers: validate_latent_structure, validate_measurement_structure, validate_causal_design.
"""

from nof1_causal_lab.artifacts import (
    LatentStructure,
    validate_causal_design,
    validate_latent_structure,
    validate_measurement_structure,
)
from tests.helpers import invalid_dict_payload


def _require_latent_structure(model: LatentStructure | None) -> LatentStructure:
    assert model is not None
    return model


def _valid_latent_data():
    """Minimal valid latent structure dict."""
    return {
        "constructs": [
            {
                "name": "stress",
                "description": "Perceived stress",
                "role": "exogenous",
                "temporal_status": "time_varying",
            },
            {
                "name": "sleep",
                "description": "Sleep quality",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
        ],
        "edges": [
            {"cause": "stress", "effect": "sleep", "description": "Stress disrupts sleep"},
        ],
    }


def _valid_measurement_data():
    """Minimal valid measurement structure dict."""
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "name": "pss_score",
                "construct_name": "stress",
                "construct_polarity": "positive",
                "how_to_measure": "Perceived Stress Scale score",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "sleep_hours",
                "construct_name": "sleep",
                "construct_polarity": "positive",
                "how_to_measure": "Hours of sleep reported",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ],
    }


# =============================================================================
# validate_latent_structure
# =============================================================================


class TestValidateLatentStructure:
    def test_valid_model_returns_model(self):
        model, errors = validate_latent_structure(_valid_latent_data())
        assert model is not None
        assert errors == []

    def test_not_dict_returns_error(self):
        model, errors = validate_latent_structure(invalid_dict_payload("not a dict"))
        assert model is None
        assert len(errors) == 1
        assert "dictionary" in errors[0].lower()

    def test_missing_constructs(self):
        model, errors = validate_latent_structure({"edges": []})
        assert model is None
        assert any("outcome" in e.lower() for e in errors)

    def test_constructs_not_list(self):
        model, errors = validate_latent_structure({"constructs": "not a list", "edges": []})
        assert model is None
        assert any("list" in e.lower() for e in errors)

    def test_duplicate_construct_name(self):
        data = _valid_latent_data()
        data["constructs"].append(data["constructs"][0].copy())
        model, errors = validate_latent_structure(data)
        assert model is None
        assert any("duplicate" in e.lower() for e in errors)

    def test_invalid_construct_schema(self):
        data = _valid_latent_data()
        data["constructs"][0] = {"name": "bad"}  # missing required fields
        model, errors = validate_latent_structure(data)
        assert model is None
        assert len(errors) > 0

    def test_multiple_errors_collected(self):
        """Should collect all errors, not just the first."""
        data = {
            "constructs": [
                {"name": "bad1"},  # invalid schema
                {"name": "bad2"},  # invalid schema
            ],
            "edges": [],
        }
        model, errors = validate_latent_structure(data)
        assert model is None
        assert len(errors) >= 2  # at least one per bad construct

    def test_edge_not_dict(self):
        data = _valid_latent_data()
        data["edges"].append("not a dict")
        model, errors = validate_latent_structure(data)
        assert model is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_construct_not_dict(self):
        data = _valid_latent_data()
        data["constructs"].append(42)
        model, errors = validate_latent_structure(data)
        assert model is None
        assert any("dictionary" in e.lower() for e in errors)


# =============================================================================
# validate_measurement_structure
# =============================================================================


class TestValidateMeasurementStructure:
    def test_valid_model_returns_model(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        model, errors = validate_measurement_structure(
            _valid_measurement_data(), _require_latent_structure(latent)
        )
        assert model is not None
        assert errors == []

    def test_not_dict_returns_error(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        model, errors = validate_measurement_structure(
            invalid_dict_payload("not a dict"), _require_latent_structure(latent)
        )
        assert model is None
        assert len(errors) == 1

    def test_indicators_not_list(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        model, errors = validate_measurement_structure(
            {"indicators": "bad"}, _require_latent_structure(latent)
        )
        assert model is None
        assert any("list" in e.lower() for e in errors)

    def test_duplicate_indicator_name(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        data = _valid_measurement_data()
        data["indicators"].append(data["indicators"][0].copy())
        model, errors = validate_measurement_structure(data, _require_latent_structure(latent))
        assert model is None
        assert any("duplicate" in e.lower() for e in errors)

    def test_indicator_not_dict(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        data = {"indicators": [42]}
        model, errors = validate_measurement_structure(data, _require_latent_structure(latent))
        assert model is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_invalid_indicator_schema(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        data = {"indicators": [{"name": "bad"}]}  # missing required fields
        model, errors = validate_measurement_structure(data, _require_latent_structure(latent))
        assert model is None
        assert len(errors) > 0


# =============================================================================
# validate_causal_design
# =============================================================================


class TestValidateCausalDesign:
    def test_valid_spec(self):
        spec, errors = validate_causal_design(_valid_latent_data(), _valid_measurement_data())
        assert spec is not None
        assert errors == []

    def test_invalid_latent_propagates(self):
        spec, errors = validate_causal_design({"bad": True}, _valid_measurement_data())
        assert spec is None
        assert any("latent" in e.lower() for e in errors)

    def test_invalid_measurement_propagates(self):
        """Measurement with unknown construct references should fail."""
        bad_measurement = {
            "indicators": [
                {
                    "name": "x",
                    "construct_name": "nonexistent",
                    "construct_polarity": "positive",
                    "how_to_measure": "test",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                }
            ]
        }
        spec, errors = validate_causal_design(_valid_latent_data(), bad_measurement)
        assert spec is None
        assert any("measurement" in e.lower() for e in errors)

    def test_both_invalid(self):
        """With invalid latent, measurement is never validated."""
        spec, errors = validate_causal_design({"bad": True}, {"also_bad": True})
        assert spec is None
        # Should have latent errors (measurement not reached)
        assert any("latent" in e.lower() for e in errors)
