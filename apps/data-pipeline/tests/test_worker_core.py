"""Tests for worker core helper functions.

Covers: _format_indicators, _get_outcome_description, WorkerMessages.
"""

from causal_ssm_agent.workers.core import (
    WorkerMessages,
    _format_indicators,
    _get_outcome_description,
)


def _causal_spec():
    """Minimal CausalSpec for testing."""
    return {
        "latent": {
            "constructs": [
                {"name": "stress", "role": "exogenous", "description": "Perceived stress"},
                {"name": "sleep", "role": "endogenous", "description": "Sleep quality", "is_outcome": True},
            ],
            "edges": [{"cause": "stress", "effect": "sleep"}],
        },
        "measurement": {
            "indicators": [
                {
                    "name": "pss_score",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Perceived Stress Scale score",
                },
                {
                    "name": "sleep_hours",
                    "construct_name": "sleep",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Self-reported hours of sleep",
                },
            ]
        },
    }


# =============================================================================
# _format_indicators
# =============================================================================


class TestFormatIndicators:
    def test_basic_formatting(self):
        result = _format_indicators(_causal_spec())
        assert "pss_score" in result
        assert "sleep_hours" in result
        assert "continuous" in result
        assert "Perceived Stress Scale" in result

    def test_empty_indicators(self):
        result = _format_indicators({"measurement": {"indicators": []}})
        assert result == ""

    def test_missing_optional_fields(self):
        spec = {"measurement": {"indicators": [{"name": "x"}]}}
        result = _format_indicators(spec)
        assert "x" in result


# =============================================================================
# _get_outcome_description
# =============================================================================


class TestGetOutcomeDescription:
    def test_returns_description(self):
        result = _get_outcome_description(_causal_spec())
        assert "Sleep quality" in result

    def test_no_outcome(self):
        spec = {
            "latent": {
                "constructs": [{"name": "X", "role": "exogenous"}],
                "edges": [],
            },
        }
        result = _get_outcome_description(spec)
        assert result == "Not specified"

    def test_outcome_without_description(self):
        spec = {
            "latent": {
                "constructs": [{"name": "Y", "role": "endogenous", "is_outcome": True}],
                "edges": [],
            },
        }
        result = _get_outcome_description(spec)
        assert "Y" in result


# =============================================================================
# WorkerMessages
# =============================================================================


class TestWorkerMessages:
    def test_extraction_messages_structure(self):
        wm = WorkerMessages(
            question="Does stress affect sleep?",
            causal_spec=_causal_spec(),
            chunk="Day 1: patient reported high stress",
        )
        msgs = wm.extraction_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_context(self):
        wm = WorkerMessages(
            question="Does stress affect sleep?",
            causal_spec=_causal_spec(),
            chunk="Day 1: PSS score = 25",
        )
        msgs = wm.extraction_messages()
        user_msg = msgs[1]["content"]
        assert "stress" in user_msg.lower() or "sleep" in user_msg.lower()
        assert "PSS score = 25" in user_msg

    def test_indicators_in_prompt(self):
        wm = WorkerMessages(
            question="test",
            causal_spec=_causal_spec(),
            chunk="test chunk",
        )
        msgs = wm.extraction_messages()
        user_msg = msgs[1]["content"]
        assert "pss_score" in user_msg
        assert "sleep_hours" in user_msg
