"""Shared contract primitives for persisted artifact payloads and tool schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001


def _inline_refs(schema: UncheckedJsonObject) -> UncheckedJsonObject:
    """Inline ``$ref`` pointers so tool schemas are self-contained."""
    defs = schema.get("$defs", {})
    if not defs:
        return schema

    def _resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref_path = node["$ref"]
            ref_name = ref_path.rsplit("/", 1)[-1]
            resolved = defs.get(ref_name, node)
            return _resolve(dict(resolved))
        return {key: _resolve(value) for key, value in node.items() if key != "$defs"}

    return _resolve(schema)


@dataclass(frozen=True)
class ToolContract:
    """Declarative tool definition shared between pipeline and codegen."""

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel] | None = None

    def parameters_json_schema(self) -> UncheckedJsonObject:
        schema = self.input_schema.model_json_schema()
        schema["additionalProperties"] = False
        return _inline_refs(schema)

    def result_json_schema(self) -> UncheckedJsonObject | None:
        if self.output_schema is None:
            return None
        schema = self.output_schema.model_json_schema(mode="serialization")
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        return _inline_refs(schema)


class BaseArtifactContract(BaseModel):
    """Shared base for persisted artifact payloads.

    Contracts are pure artifacts: execution failure is a typed exception on
    the transition (state unchanged, attempt journaled), and negative
    findings are report-present / enabling-artifact-absent — never an
    ``outcome`` enum on the payload.
    """

    model_config = ConfigDict(extra="forbid")
