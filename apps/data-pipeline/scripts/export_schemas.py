"""Export Pydantic contract schemas as JSON Schema for TypeScript codegen.

Imports all contract models and their nested domain models, generates a
combined JSON Schema document, and writes it to the api-types package.

Usage:
    cd apps/data-pipeline
    uv run python scripts/export_schemas.py
"""

from __future__ import annotations

import json
from pathlib import Path

# Import all stage contracts — this pulls in every nested domain model
from causal_ssm_agent.flows.stages.contracts import (
    STAGE_CONTRACTS,
    PartialStageResult,
)

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "packages" / "api-types" / "schemas"


def _make_defaults_required(schema: dict) -> dict:
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


def _is_nullable(prop_schema: dict) -> bool:
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
    for item in any_of:
        if isinstance(item, dict) and item.get("type") == "null":
            return True

    return False


def export_schemas() -> dict:
    """Build a combined JSON Schema with all stage contracts in $defs."""
    all_defs: dict = {}
    stage_refs: dict[str, dict] = {}

    for stage_id, model_cls in STAGE_CONTRACTS.items():
        schema = model_cls.model_json_schema(mode="serialization")

        # Collect nested $defs
        defs = schema.pop("$defs", {})
        all_defs.update(defs)

        # Store the top-level contract under a clean name
        contract_name = model_cls.__name__
        all_defs[contract_name] = {k: v for k, v in schema.items() if k not in ("$defs",)}
        stage_refs[stage_id] = {"$ref": f"#/$defs/{contract_name}"}

    # Add PartialStageResult (live trace contract) alongside stage contracts
    partial_schema = PartialStageResult.model_json_schema(mode="serialization")
    partial_defs = partial_schema.pop("$defs", {})
    all_defs.update(partial_defs)
    all_defs["PartialStageResult"] = {
        k: v for k, v in partial_schema.items() if k not in ("$defs",)
    }
    stage_refs["_partial"] = {"$ref": "#/$defs/PartialStageResult"}

    combined = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CausalSSMContracts",
        "description": "Combined JSON Schema for all stage contracts. Generated from Python Pydantic models.",
        "type": "object",
        "properties": stage_refs,
        "$defs": dict(sorted(all_defs.items())),
    }

    # Post-process: make non-nullable defaults required
    combined = _make_defaults_required(combined)

    return combined


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "contracts.json"

    schema = export_schemas()
    output_path.write_text(json.dumps(schema, indent=2) + "\n")

    n_defs = len(schema.get("$defs", {}))
    print(f"Exported {n_defs} definitions to {output_path}")


if __name__ == "__main__":
    main()
