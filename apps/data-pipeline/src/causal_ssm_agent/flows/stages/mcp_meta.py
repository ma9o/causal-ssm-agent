"""MCP server metadata derived from pipeline configuration.

Lightweight module with no heavy dependencies so it can be safely imported by
export_schemas.py.

When adding new interactive stages or large-array fields, update here.
The codegen pipeline will propagate changes to the MCP server.
"""

from __future__ import annotations

# Stages supporting human/agent-in-the-loop refinement via ``stage_overrides``.
INTERACTIVE_STAGES: frozenset[str] = frozenset({"stage-1a", "stage-1b", "stage-4"})

# Top-level fields per stage that contain large numerical arrays.
# Stripped from MCP results by default to keep responses within context limits.
LARGE_ARRAY_FIELDS: dict[str, list[str]] = {
    "stage-4": ["prior_predictive_samples"],
    "stage-5": ["posterior_marginals", "posterior_pairs"],
}

# For stages with arrays of objects, nested fields within each object
# that contain large numerical arrays.
LARGE_NESTED_FIELDS: dict[str, dict[str, list[str]]] = {
    "stage-6": {"intervention_results": ["posterior_draws"]},
}
