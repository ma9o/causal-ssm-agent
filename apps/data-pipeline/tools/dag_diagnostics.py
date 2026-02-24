"""Shared diagnostics core for DAG explorer and CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from commons import parse_dag_json

from causal_ssm_agent.utils.effects import get_outcome_from_latent_model
from causal_ssm_agent.utils.identifiability import (
    analyze_unobserved_constructs,
    check_identifiability,
    format_identifiability_report,
    format_marginalization_report,
    get_observed_constructs,
)


@dataclass(slots=True)
class DagDiagnostics:
    """Container for DAG diagnostics shared by Streamlit UI and CLI."""

    data: dict[str, Any]
    identifiability: dict[str, Any]
    marginalization: dict[str, Any]
    graph_summary: dict[str, Any]
    measurement_summary: dict[str, Any]
    identifiability_report: str
    marginalization_report: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable payload with diagnostics."""
        return {
            "graph_summary": self.graph_summary,
            "measurement_summary": self.measurement_summary,
            "identifiability": _jsonify(self.identifiability),
            "marginalization": _jsonify(self.marginalization),
            "identifiability_report": self.identifiability_report,
            "marginalization_report": self.marginalization_report,
        }


def load_model_file(path: str | Path) -> dict[str, Any]:
    """Load a causal model JSON file and normalize it for diagnostics."""
    with open(path) as f:
        raw = json.load(f)
    normalized, error = parse_dag_json(json.dumps(raw))
    if error:
        raise ValueError(f"{path}: {error}")
    return normalized


def run_diagnostics(data: dict[str, Any]) -> DagDiagnostics:
    """Run identifiability + marginalization diagnostics on normalized data."""
    latent_model = {
        "constructs": data["constructs"],
        "edges": data["edges"],
    }
    measurement_model = {
        "indicators": data["indicators"],
    }

    id_result = check_identifiability(latent_model, measurement_model)
    marginalization = analyze_unobserved_constructs(
        latent_model,
        measurement_model,
        id_result,
    )

    outcome = get_outcome_from_latent_model(latent_model) or "unknown"
    graph_summary = _build_graph_summary(data, measurement_model, outcome)
    measurement_summary = _summarize_measurement(data["indicators"])

    return DagDiagnostics(
        data=data,
        identifiability=id_result,
        marginalization=marginalization,
        graph_summary=graph_summary,
        measurement_summary=measurement_summary,
        identifiability_report=format_identifiability_report(id_result, outcome),
        marginalization_report=format_marginalization_report(marginalization),
    )


def _build_graph_summary(
    data: dict[str, Any],
    measurement_model: dict[str, Any],
    outcome: str,
) -> dict[str, Any]:
    observed = get_observed_constructs(measurement_model)
    constructs = {c["name"] for c in data["constructs"]}
    unobserved = constructs - observed

    return {
        "construct_count": len(data["constructs"]),
        "edge_count": len(data["edges"]),
        "indicator_count": len(data["indicators"]),
        "observed_constructs": sorted(observed),
        "unobserved_constructs": sorted(unobserved),
        "outcome": outcome,
    }


def _summarize_measurement(indicators: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, list[dict[str, Any]]] = {}
    for indicator in indicators:
        construct = indicator.get("construct_name", "unknown")
        summary.setdefault(construct, []).append(
            {
                "name": indicator.get("name"),
                "dtype": indicator.get("measurement_dtype"),
                "aggregation": indicator.get("aggregation"),
            }
        )
    return summary


def _jsonify(value: Any) -> Any:
    """Recursively convert sets to sorted lists for JSON serialization."""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value
