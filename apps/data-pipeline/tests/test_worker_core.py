"""Tests for worker core helper functions.

Covers: _format_indicators, _get_outcome_description, WorkerMessages,
run_worker_extraction.
"""

import asyncio
import json
import logging

import polars as pl
import pytest

from causal_ssm_agent.workers.core import (
    WorkerMessages,
    _format_indicators,
    _get_outcome_description,
    run_worker_extraction,
)


def _causal_spec():
    """Minimal CausalSpec for testing."""
    return {
        "latent": {
            "constructs": [
                {"name": "stress", "role": "exogenous", "description": "Perceived stress"},
                {
                    "name": "sleep",
                    "role": "endogenous",
                    "description": "Sleep quality",
                    "is_outcome": True,
                },
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


def _run(coro):
    return asyncio.run(coro)


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
    def _sample_df(self):
        return pl.DataFrame(
            {
                "date": ["Day 1", "Day 2"],
                "pss_score": [25, 18],
                "sleep_hours": [6.5, 7.2],
            }
        )

    def test_extraction_messages_structure(self):
        wm = WorkerMessages(
            question="Does stress affect sleep?",
            causal_spec=_causal_spec(),
            chunk_df=self._sample_df(),
        )
        msgs = wm.extraction_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_context(self):
        wm = WorkerMessages(
            question="Does stress affect sleep?",
            causal_spec=_causal_spec(),
            chunk_df=self._sample_df(),
        )
        msgs = wm.extraction_messages()
        user_msg = msgs[1]["content"]
        assert "stress" in user_msg.lower() or "sleep" in user_msg.lower()
        # CSV format should contain the actual data values
        assert "25" in user_msg
        assert "6.5" in user_msg

    def test_indicators_in_prompt(self):
        wm = WorkerMessages(
            question="test",
            causal_spec=_causal_spec(),
            chunk_df=self._sample_df(),
        )
        msgs = wm.extraction_messages()
        user_msg = msgs[1]["content"]
        assert "pss_score" in user_msg
        assert "sleep_hours" in user_msg


# =============================================================================
# run_worker_extraction
# =============================================================================


class TestRunWorkerExtraction:
    def _sample_df(self):
        return pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00Z"],
                "activity_type": ["Search"],
                "full_title": ["Searched for sleep hygiene"],
                "url": ["https://example.com"],
            }
        )

    def test_empty_completion_raises_parse_error(self, caplog):
        async def fake_generate(messages, tools=None, follow_ups=None):
            return ""

        logger = logging.getLogger("test_worker_core")

        with (
            caplog.at_level(logging.INFO, logger=logger.name),
            pytest.raises(ValueError, match="Failed to parse model response as JSON"),
        ):
            _run(
                run_worker_extraction(
                    chunk_df=self._sample_df(),
                    question="How does screen time affect sleep?",
                    causal_spec=_causal_spec(),
                    generate=fake_generate,
                    logger=logger,
                )
            )

        assert "Model call returned 0 characters" in caplog.text
        assert "falling back to completion parsing" in caplog.text

    def test_call_label_passed_when_generate_supports_it(self, caplog):
        captured: dict[str, str | None] = {}

        async def fake_generate(messages, tools=None, follow_ups=None, label=None):
            captured["label"] = label
            return json.dumps(
                {
                    "extractions": [
                        {
                            "indicator": "pss_score",
                            "value": 12.0,
                            "timestamp": "2024-01-01T00:00:00Z",
                        }
                    ]
                }
            )

        logger = logging.getLogger("test_worker_core")

        with caplog.at_level(logging.INFO, logger=logger.name):
            result = _run(
                run_worker_extraction(
                    chunk_df=self._sample_df(),
                    question="How does screen time affect sleep?",
                    causal_spec=_causal_spec(),
                    generate=fake_generate,
                    logger=logger,
                    call_label="stage2 chunk=3 rows=1 cols=4",
                )
            )

        assert captured["label"] == "stage2 chunk=3 rows=1 cols=4"
        assert result.dataframe.height == 1
        assert "[stage2 chunk=3 rows=1 cols=4] Calling extraction model" in caplog.text

    def test_call_label_is_optional_for_legacy_generate_functions(self):
        async def fake_generate(messages, tools=None, follow_ups=None):
            return json.dumps(
                {
                    "extractions": [
                        {
                            "indicator": "sleep_hours",
                            "value": 7.5,
                            "timestamp": "2024-01-01T00:00:00Z",
                        }
                    ]
                }
            )

        result = _run(
            run_worker_extraction(
                chunk_df=self._sample_df(),
                question="How does screen time affect sleep?",
                causal_spec=_causal_spec(),
                generate=fake_generate,
                call_label="stage2 chunk=4 rows=1 cols=4",
            )
        )

        assert result.dataframe.height == 1
        assert result.dataframe["indicator"].to_list() == ["sleep_hours"]
