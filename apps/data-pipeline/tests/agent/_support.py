"""Shared fixtures for agent-session utility tests."""

from __future__ import annotations

import json

from nof1_causal_lab.utils.llm import make_validation_tool
from nof1_causal_lab.workers.schemas import validate_worker_output


def worker_schema():
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "name": "sleep_hours",
                "construct_name": "sleep",
                "measurement_dtype": "continuous",
                "aggregation": "last",
                "how_to_measure": "Read sleep hours directly from the rows",
            }
        ],
    }


def valid_worker_output_json() -> str:
    return json.dumps(
        {
            "extractions": [
                {
                    "indicator": "sleep_hours",
                    "value": 7.5,
                    "window_start": "2024-01-01T00:00:00Z",
                }
            ]
        }
    )


def make_worker_tool(schema=None):
    if schema is None:
        schema = worker_schema()
    return make_validation_tool(
        name="validate_extractions",
        description="Validate worker extraction output JSON.",
        param_name="output_json",
        param_description="The JSON string containing the worker output.",
        validator=lambda data: validate_worker_output(data, schema),
        capture_key="output",
    )
