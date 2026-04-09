"""Slow Stage 4 tests."""

import pytest

from tests.test_stage4 import _make_polars_data

pytestmark = pytest.mark.slow


class TestPriorPredictiveValidation:
    def test_build_validation_payload_from_assembly(self, simple_model_spec, simple_priors):
        from causal_ssm_agent.flows.stages.stage4.assembly import (
            build_validation_payload,
            validate_assembly,
        )

        data_for_model = _make_polars_data()
        validation = validate_assembly(simple_model_spec, simple_priors, data_for_model, None, None)
        result = build_validation_payload(validation, simple_model_spec)
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["results"], list)
        assert isinstance(result["issues"], list)
        assert isinstance(result["warnings"], list)
        for issue in result["issues"]:
            assert isinstance(issue, str)
        for warning in result["warnings"]:
            assert isinstance(warning, str)
