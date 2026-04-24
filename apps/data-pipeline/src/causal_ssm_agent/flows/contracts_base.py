"""Shared contract primitives for persisted stage payloads and tool schemas."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from causal_ssm_agent.utils.llm import LLMTrace  # noqa: TC001

StageId = Literal[
    "stage-0",
    "stage-1a",
    "stage-1b",
    "stage-2",
    "stage-3",
    "stage-4",
    "stage-4b",
    "stage-5a",
    "stage-5b",
    "stage-6",
]


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
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

    def parameters_json_schema(self) -> dict[str, Any]:
        schema = self.input_schema.model_json_schema()
        schema["additionalProperties"] = False
        return _inline_refs(schema)

    def result_json_schema(self) -> dict[str, Any] | None:
        if self.output_schema is None:
            return None
        schema = self.output_schema.model_json_schema(mode="serialization")
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        return _inline_refs(schema)


class BaseStageContract(BaseModel):
    """Shared base for persisted stage payloads."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    fail_reason: str | None = None

    @model_validator(mode="after")
    def validate_fail_reason(self) -> BaseStageContract:
        if self.outcome == "fail" and self.fail_reason is None:
            raise ValueError("fail_reason is required when outcome='fail'")
        if self.outcome != "fail" and self.fail_reason is not None:
            raise ValueError("fail_reason is only allowed when outcome='fail'")
        return self

    def summary_level(self) -> int:
        return logging.WARNING if self.outcome in {"warn", "fail"} else logging.INFO

    def summary_message(self) -> str:
        raise NotImplementedError

    def summarize(self) -> tuple[int, str]:
        return self.summary_level(), self.summary_message()


class LLMStageContract(BaseStageContract):
    """Base contract for stages that surface an LLM trace."""

    llm_trace: LLMTrace | None = None
