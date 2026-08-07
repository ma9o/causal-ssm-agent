"""Helpers for projecting transition output onto persisted model contracts."""

from typing import Any, cast

from pydantic import BaseModel

type ModelContractData = dict[str, Any]


def filter_model_fields(
    model: type[BaseModel],
    data: ModelContractData,
) -> ModelContractData:
    """Keep only fields declared by ``model`` while preserving input order."""
    return {key: value for key, value in data.items() if key in model.model_fields}


def project_model_fields(
    model: type[BaseModel],
    data: ModelContractData,
) -> ModelContractData:
    """Drop transient fields, validate the persisted contract, and serialize it."""
    projected = filter_model_fields(model, data)
    validated = model.model_validate(projected)
    return cast(
        "ModelContractData",
        validated.model_dump(mode="json", exclude_unset=True),
    )
