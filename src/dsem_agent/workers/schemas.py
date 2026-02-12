"""Schemas for worker LLM outputs."""

from typing import Any

import polars as pl
from pydantic import BaseModel, Field


class Extraction(BaseModel):
    """A single extracted observation for an indicator."""

    indicator: str = Field(description="Name of the indicator")
    value: int | float | bool | str | None = Field(
        description="Extracted value of the correct datatype"
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp if identifiable",
    )


class ProposedIndicator(BaseModel):
    """A suggested new indicator found in local data."""

    name: str = Field(description="Variable name")
    description: str = Field(description="What this variable represents")
    evidence: str = Field(description="What was seen in this chunk")
    relevant_because: str = Field(description="How it connects to the causal question")
    not_already_in_indicators_because: str = Field(
        description="Why it needs to be added and why existing indicators don't capture it"
    )


class WorkerOutput(BaseModel):
    """Complete output from a worker processing a single chunk."""

    extractions: list[Extraction] = Field(
        default_factory=list,
        description="Extracted observations for indicators",
    )
    proposed_indicators: list[ProposedIndicator] | None = Field(
        default=None,
        description="Suggested new indicators if something important is missing",
    )

    def to_dataframe(self) -> pl.DataFrame:
        """Convert extractions to a Polars DataFrame.

        Returns:
            DataFrame with columns: indicator, value, timestamp
            Value column uses pl.Object to preserve mixed types.
        """
        if not self.extractions:
            return pl.DataFrame(
                schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8}
            )

        return pl.DataFrame(
            [
                {
                    "indicator": e.indicator,
                    "value": e.value,
                    "timestamp": e.timestamp,
                }
                for e in self.extractions
            ],
            schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
        )


def _check_dtype_match(value: Any, expected_dtype: str) -> bool:
    """Check if a value matches the expected measurement_dtype."""
    if value is None:
        return True  # None is always acceptable

    dtype_checks = {
        "continuous": lambda v: isinstance(v, (int, float)),
        "binary": lambda v: (
            isinstance(v, bool) or v in (0, 1, "0", "1", "true", "false", "True", "False")
        ),
        "count": lambda v: isinstance(v, int) or (isinstance(v, float) and v == int(v) and v >= 0),
        "ordinal": lambda v: isinstance(
            v, (int, float, str)
        ),  # Flexible - can be numeric or string
        "categorical": lambda v: isinstance(v, str),
    }

    check = dtype_checks.get(expected_dtype)
    if check is None:
        return True  # Unknown dtype, don't fail
    return check(value)


def _get_indicator_info(causal_spec: dict) -> dict[str, dict]:
    """Extract indicator info from a CausalSpec dict.

    Args:
        causal_spec: CausalSpec dict with latent.constructs and measurement.indicators

    Returns:
        Dict mapping indicator name to {dtype, construct_name}
    """
    indicators = causal_spec.get("measurement", {}).get("indicators", [])
    return {
        ind.get("name"): {
            "dtype": ind.get("measurement_dtype"),
            "construct_name": ind.get("construct") or ind.get("construct_name"),
        }
        for ind in indicators
    }


def _get_all_construct_names(causal_spec: dict) -> set[str]:
    """Get all construct names from a CausalSpec dict."""
    constructs = causal_spec.get("latent", {}).get("constructs", [])
    return {c.get("name") for c in constructs}


def validate_worker_output(
    data: dict,
    causal_spec: dict,
) -> tuple[WorkerOutput | None, list[str]]:
    """Validate worker output dict, collecting ALL errors instead of failing on first.

    Args:
        data: Dictionary to validate as WorkerOutput
        causal_spec: The CausalSpec dict to validate against

    Returns:
        Tuple of (validated output or None, list of error messages)
    """
    errors = []

    # Basic structure checks
    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    extractions = data.get("extractions", [])
    proposed_indicators = data.get("proposed_indicators")

    if not isinstance(extractions, list):
        errors.append("'extractions' must be a list")
        extractions = []

    if proposed_indicators is not None and not isinstance(proposed_indicators, list):
        errors.append("'proposed_indicators' must be a list or null")
        proposed_indicators = None

    # Build set of valid indicator names and their dtypes
    indicator_info = _get_indicator_info(causal_spec)
    all_construct_names = _get_all_construct_names(causal_spec)

    # Validate each extraction
    valid_extractions = []
    for i, ext_data in enumerate(extractions):
        if not isinstance(ext_data, dict):
            errors.append(f"extractions[{i}]: must be a dictionary")
            continue

        ind_name = ext_data.get("indicator", "<missing>")
        value = ext_data.get("value")

        # Check indicator exists
        if ind_name not in indicator_info:
            valid_ind_names = ", ".join(sorted(indicator_info.keys()))
            errors.append(
                f"extractions[{i}]: indicator '{ind_name}' not in indicators. "
                f"Valid indicators: {valid_ind_names}"
            )
            continue

        # Check dtype match
        expected_dtype = indicator_info[ind_name]["dtype"]
        if not _check_dtype_match(value, expected_dtype):
            errors.append(
                f"extractions[{i}]: value {value!r} for '{ind_name}' doesn't match "
                f"expected dtype '{expected_dtype}'"
            )
            continue

        # Normalize to "indicator" key
        normalized = {
            "indicator": ind_name,
            "value": value,
            "timestamp": ext_data.get("timestamp"),
        }

        # Validate via Pydantic
        try:
            ext = Extraction.model_validate(normalized)
            valid_extractions.append(ext)
        except Exception as e:
            errors.append(f"extractions[{i}] ({ind_name}): {e}")

    # Validate proposed indicators if present
    valid_proposed = None
    if proposed_indicators is not None:
        valid_proposed = []
        for i, prop_data in enumerate(proposed_indicators):
            if not isinstance(prop_data, dict):
                errors.append(f"proposed_indicators[{i}]: must be a dictionary")
                continue

            name = prop_data.get("name", "<missing>")

            # Check not already in schema (check both indicators and constructs)
            all_names = set(indicator_info.keys()) | all_construct_names
            if name in all_names:
                errors.append(f"proposed_indicators[{i}]: '{name}' already exists in schema")
                continue

            normalized_prop = {
                "name": name,
                "description": prop_data.get("description", ""),
                "evidence": prop_data.get("evidence", ""),
                "relevant_because": prop_data.get("relevant_because", ""),
                "not_already_in_indicators_because": prop_data.get(
                    "not_already_in_indicators_because", ""
                ),
            }

            try:
                prop = ProposedIndicator.model_validate(normalized_prop)
                valid_proposed.append(prop)
            except Exception as e:
                errors.append(f"proposed_indicators[{i}] ({name}): {e}")

    # If no errors, build and return the output
    if not errors:
        try:
            output = WorkerOutput(
                extractions=valid_extractions,
                proposed_indicators=valid_proposed if valid_proposed else None,
            )
            return output, []
        except Exception as e:
            errors.append(f"Final validation failed: {e}")

    return None, errors
