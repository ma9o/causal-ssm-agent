"""Tests for worker core helper functions.

Covers: _format_indicators, WorkerMessages, run_worker_extraction.
"""

import json
import logging

import pytest

from nof1_causal_lab.workers.core import (
    WorkerMessages,
    _format_indicators,
    run_worker_extraction,
)
from tests.helpers import make_mock_session_factory
from tests.helpers import run_async as _run


def _measurement_structure():
    """Minimal MeasurementStructure for testing."""
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "name": "pss_score",
                "construct_name": "stress",
                "measurement_dtype": "continuous",
                "how_to_measure": "Perceived Stress Scale score",
                "aggregation": "mean",
            },
            {
                "name": "sleep_hours",
                "construct_name": "sleep",
                "measurement_dtype": "continuous",
                "how_to_measure": "Self-reported hours of sleep",
                "aggregation": "mean",
            },
        ],
    }


# =============================================================================
# _format_indicators
# =============================================================================


class TestFormatIndicators:
    def test_basic_formatting(self):
        result = _format_indicators(_measurement_structure())
        assert "pss_score" in result
        assert "sleep_hours" in result
        assert "continuous" in result
        assert "Perceived Stress Scale" in result
        assert "support=interval" in result
        assert "operator=mean" in result
        assert "window=1d" in result

    def test_empty_indicators(self):
        result = _format_indicators({"model_clock": "1d", "indicators": []})
        assert result == ""

    def test_missing_optional_fields(self):
        spec = {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "x",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                }
            ],
        }
        result = _format_indicators(spec)
        assert "x" in result

    def test_indicator_specific_window_overrides_model_clock(self):
        spec = {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "monthly_pss_score",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Average perceived stress over the last month",
                    "aggregation": "mean",
                    "observation_window": "1mo",
                }
            ],
        }

        result = _format_indicators(spec)
        assert "window=1mo" in result


# =============================================================================
# WorkerMessages
# =============================================================================


class TestWorkerMessages:
    def _sample_window_text(self):
        return "## Window Start: 2024-01-01\n\n08:00  pss=25, sleep=6.5\n09:00  pss=18, sleep=7.2"

    def test_user_message_contains_context(self):
        wm = WorkerMessages(
            question="Does stress affect sleep?",
            measurement_structure=_measurement_structure(),
            window_text=self._sample_window_text(),
            n_windows=1,
        )
        msgs = wm.extraction_messages()
        user_msg = msgs[1]["content"]
        assert "stress" in user_msg.lower() or "sleep" in user_msg.lower()
        assert "25" in user_msg
        assert "6.5" in user_msg

    def test_indicators_in_prompt(self):
        wm = WorkerMessages(
            question="test",
            measurement_structure=_measurement_structure(),
            window_text=self._sample_window_text(),
            n_windows=1,
        )
        msgs = wm.extraction_messages()
        user_msg = msgs[1]["content"]
        assert "pss_score" in user_msg
        assert "sleep_hours" in user_msg


# =============================================================================
# run_worker_extraction
# =============================================================================


class TestRunWorkerExtraction:
    def _sample_window_text(self):
        return "## Window Start: 2024-01-01\n\n08:00  Searched for sleep hygiene"

    def _sample_window_starts(self):
        return ["2024-01-01"]

    def test_empty_completion_raises_parse_error(self, caplog):
        factory = make_mock_session_factory([""])
        logger = logging.getLogger("test_worker_core")

        with (
            caplog.at_level(logging.INFO, logger=logger.name),
            pytest.raises(
                RuntimeError,
                match="did not capture structured output",
            ),
        ):
            _run(
                run_worker_extraction(
                    window_text=self._sample_window_text(),
                    window_starts=self._sample_window_starts(),
                    question="How does screen time affect sleep?",
                    measurement_structure=_measurement_structure(),
                    session_factory=factory,
                    logger=logger,
                )
            )

        assert "Model call returned 0 characters" in caplog.text

    def test_call_label_passed_when_generate_supports_it(self, caplog):
        factory = make_mock_session_factory(
            [
                json.dumps(
                    {
                        "extractions": [
                            {
                                "window_start": "2024-01-01",
                                "indicator": "pss_score",
                                "value": 12.0,
                            }
                        ]
                    }
                )
            ]
        )

        logger = logging.getLogger("test_worker_core")

        with caplog.at_level(logging.INFO, logger=logger.name):
            result = _run(
                run_worker_extraction(
                    window_text=self._sample_window_text(),
                    window_starts=self._sample_window_starts(),
                    question="How does screen time affect sleep?",
                    measurement_structure=_measurement_structure(),
                    session_factory=factory,
                    logger=logger,
                    call_label="extraction chunk=3 windows=1 events=1",
                )
            )

        assert result.dataframe.height == 1
        assert "[extraction chunk=3 windows=1 events=1] Calling extraction model" in caplog.text
