"""Export Pydantic schemas as JSON Schema for TypeScript codegen.

Imports all contract/API models and their nested domain models, generates a
combined JSON Schema document, and writes it to the api-types package.

Usage:
    cd apps/data-pipeline
    uv run python scripts/export_schemas.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from artifact_contract_catalog import ARTIFACT_CONTRACTS

from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
from nof1_causal_lab.distributions import OBSERVATION_FAMILY_SPECS
from nof1_causal_lab.episode_api import (
    ArtifactEnvelope,
    CapabilitiesResponse,
    UploadResponse,
    WorkspaceEntry,
    WorkspaceList,
)

# Import all artifact contracts — this pulls in every nested domain model
from nof1_causal_lab.flows.artifact_contracts import (
    CONTEXT_TOOLS,
)
from nof1_causal_lab.flows.transitions.analysis.contracts import (
    BaselineReportVisualizationContract,
    EffectSummaryContract,
    EffectTrajectoryPointContract,
    ScenarioStartResultContract,
    SimulateScenarioResultContract,
    SimulateScenarioToolResultContract,
    ToolErrorContract,
)
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm.parameterization import SiteKind
from nof1_causal_lab.utils.llm import LLMTrace

if TYPE_CHECKING:
    from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "packages" / "api-types" / "schemas"

EXPORTED_API_MODELS: tuple[type[BaseModel], ...] = (
    CapabilitiesResponse,
    WorkspaceEntry,
    WorkspaceList,
    UploadResponse,
    ArtifactEnvelope,
    LLMTrace,
    StructuralPlan,
)

EXPORTED_TOOL_RESULT_MODELS: tuple[type[BaseModel], ...] = (
    ToolErrorContract,
    EffectSummaryContract,
    EffectTrajectoryPointContract,
    BaselineReportVisualizationContract,
    ScenarioStartResultContract,
    SimulateScenarioResultContract,
    SimulateScenarioToolResultContract,
)

INTERACTIVE_CONTEXTS = frozenset(
    {"latent-structure", "measurement-structure", "statistical-model-spec", "ranking"}
)


def _make_defaults_required(schema: UncheckedJsonObject) -> UncheckedJsonObject:
    """Make all properties with defaults required in serialization schema.

    Pydantic marks fields with defaults as optional in JSON Schema, but in
    serialization mode they're always present. This post-processes the schema
    to make them required so TypeScript types aren't overly permissive.

    Only applies to object schemas that have 'properties'.
    Does NOT touch fields where the default is None and the type includes null
    (those are genuinely optional/nullable).
    """
    if not isinstance(schema, dict):
        return schema

    # Recurse into $defs
    if "$defs" in schema:
        for name, defn in schema["$defs"].items():
            schema["$defs"][name] = _make_defaults_required(defn)

    # Recurse into properties
    if "properties" in schema:
        props = schema["properties"]
        current_required = set(schema.get("required", []))

        for prop_name, prop_schema in props.items():
            # Skip already-required fields
            if prop_name in current_required:
                continue

            # Skip fields that are nullable (anyOf with null) — these are
            # genuinely optional fields that default to None
            if _is_nullable(prop_schema):
                continue

            # This field has a default but is not nullable — make it required
            current_required.add(prop_name)

        if current_required:
            schema["required"] = sorted(current_required)

        # Recurse into nested properties
        for prop_schema in props.values():
            _make_defaults_required(prop_schema)

    # Recurse into items (arrays)
    if "items" in schema:
        _make_defaults_required(schema["items"])

    # Recurse into anyOf/oneOf
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for i, item in enumerate(schema[key]):
                schema[key][i] = _make_defaults_required(item)

    return schema


def _is_nullable(prop_schema: UncheckedJsonObject) -> bool:
    """Check if a property schema allows null (e.g., anyOf with null type)."""
    if not isinstance(prop_schema, dict):
        return False

    # Direct null type
    if prop_schema.get("type") == "null":
        return True

    # Default is None
    if prop_schema.get("default") is None and "default" in prop_schema:
        return True

    # anyOf contains null
    any_of = prop_schema.get("anyOf", [])
    return any(isinstance(item, dict) and item.get("type") == "null" for item in any_of)


def _collect_model_schema(
    model_cls: type[BaseModel], all_defs: UncheckedJsonObject
) -> dict[str, str]:
    schema = model_cls.model_json_schema(mode="serialization")
    defs = schema.pop("$defs", {})
    all_defs.update(defs)
    model_name = model_cls.__name__
    all_defs[model_name] = {k: v for k, v in schema.items() if k not in ("$defs",)}
    return {"$ref": f"#/$defs/{model_name}"}


def export_schemas() -> UncheckedJsonObject:
    """Build a combined JSON Schema with exported Python models in $defs."""
    all_defs: UncheckedJsonObject = {}
    artifact_refs: dict[str, dict[str, str]] = {}

    for artifact_id, model_cls in ARTIFACT_CONTRACTS.items():
        artifact_refs[artifact_id] = _collect_model_schema(model_cls, all_defs)

    for model_cls in EXPORTED_API_MODELS:
        _collect_model_schema(model_cls, all_defs)

    combined = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CausalSSMContracts",
        "description": "Combined JSON Schema for exported artifact contracts and facade API models. Generated from Python Pydantic models.",
        "type": "object",
        "properties": artifact_refs,
        "$defs": dict(sorted(all_defs.items())),
    }

    # Post-process: make non-nullable defaults required
    return _make_defaults_required(combined)


def export_tool_result_schemas() -> UncheckedJsonObject:
    """Build a JSON Schema dedicated to tool result contracts."""
    all_defs: UncheckedJsonObject = {}
    refs: list[dict[str, str]] = []

    for model_cls in EXPORTED_TOOL_RESULT_MODELS:
        schema = model_cls.model_json_schema(mode="serialization")
        defs = schema.pop("$defs", {})
        all_defs.update(defs)
        contract_name = model_cls.__name__
        all_defs[contract_name] = {k: v for k, v in schema.items() if k not in ("$defs",)}
        refs.append({"$ref": f"#/$defs/{contract_name}"})

    combined = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CausalSSMToolResults",
        "description": "Combined JSON Schema for declared tool result contracts.",
        "anyOf": refs,
        "$defs": dict(sorted(all_defs.items())),
    }
    return _make_defaults_required(combined)


def export_tool_schemas() -> UncheckedJsonObject:
    """Build a JSON document describing all context tools for TypeScript codegen.

    Output structure::

        {
          "latent-structure": [
            {"name": "validate_latent_structure", "description": "...", "parameters": {...}},
          ],
          ...
          "_interactive": ["latent-structure", "measurement-structure", ...]
        }
    """
    result: UncheckedJsonObject = {}
    for context_id, tools in CONTEXT_TOOLS.items():
        result[context_id] = [
            {
                "name": tc.name,
                "description": tc.description,
                "parameters": tc.parameters_json_schema(),
                "result": tc.result_json_schema(),
            }
            for tc in tools
        ]
    result["_interactive"] = sorted(INTERACTIVE_CONTEXTS)
    return result


def export_metadata() -> UncheckedJsonObject:
    """Export distribution catalog metadata for TypeScript type-safe rendering maps."""
    site_kind_values = {sk.value for sk in SiteKind if sk.name.startswith("OBS_")}
    catalog_hypers = {h for spec in OBSERVATION_FAMILY_SPECS for h in spec.hyperparameters}
    if catalog_hypers != site_kind_values:
        diff = catalog_hypers.symmetric_difference(site_kind_values)
        raise ValueError(
            f"ObservationFamilyCatalogEntry.hyperparameters out of sync with SiteKind: {diff}"
        )
    return {
        "observationHyperparametersByDistribution": {
            spec.family.value: list(spec.hyperparameters)
            for spec in OBSERVATION_FAMILY_SPECS
            if spec.hyperparameters
        },
    }


def _write_or_check_json(
    path: Path,
    payload: UncheckedJsonObject,
    *,
    check: bool,
    changed_paths: list[Path],
) -> None:
    rendered = json.dumps(payload, indent=2) + "\n"
    if check:
        if not path.exists() or path.read_text() != rendered:
            changed_paths.append(path.relative_to(REPO_ROOT))
        return

    path.write_text(rendered)


def main(*, check: bool = False) -> bool:
    if not check:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    changed_paths: list[Path] = []

    # Export JSON Schema for TypeScript type codegen
    output_path = OUTPUT_DIR / "contracts.json"
    schema = export_schemas()
    _write_or_check_json(output_path, schema, check=check, changed_paths=changed_paths)
    n_defs = len(schema.get("$defs", {}))
    if not check:
        print(f"Exported {n_defs} definitions to {output_path}")

    # Export tool definitions for TypeScript tool codegen
    tools_path = OUTPUT_DIR / "tools.json"
    tools = export_tool_schemas()
    _write_or_check_json(tools_path, tools, check=check, changed_paths=changed_paths)
    n_tools = sum(len(v) for k, v in tools.items() if k != "_interactive")
    if not check:
        print(f"Exported {n_tools} tool definitions to {tools_path}")

    tool_results_path = OUTPUT_DIR / "tool-results.json"
    tool_results = export_tool_result_schemas()
    _write_or_check_json(tool_results_path, tool_results, check=check, changed_paths=changed_paths)
    n_tool_defs = len(tool_results.get("$defs", {}))
    if not check:
        print(f"Exported {n_tool_defs} tool result definitions to {tool_results_path}")

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata = export_metadata()
    _write_or_check_json(metadata_path, metadata, check=check, changed_paths=changed_paths)
    if not check:
        print(f"Exported metadata to {metadata_path}")

    if check and changed_paths:
        print("Schema exports are out of date. Run `bun run codegen`.", file=sys.stderr)
        for path in changed_paths:
            print(f"  {path}", file=sys.stderr)
        return True

    if check:
        print("Schema exports checked.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Verify generated schemas without writing files."
    )
    args = parser.parse_args()
    if main(check=args.check):
        sys.exit(1)
