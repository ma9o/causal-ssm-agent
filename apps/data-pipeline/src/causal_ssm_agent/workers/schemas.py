"""Schemas for worker LLM outputs."""

from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from causal_ssm_agent.utils.causal_spec import get_indicator_info as _get_indicator_info


class TickExtraction(BaseModel):
    """A single extracted observation for an indicator within a tick."""

    tick: str = Field(description="The clock tick ID (e.g. '2024-01-15T00:00:00')")
    indicator: str = Field(description="Name of the indicator")
    value: int | float | bool | str | None = Field(
        description="Extracted value of the correct datatype"
    )


class WorkerOutput(BaseModel):
    """Complete output from a worker processing a chunk of ticks."""

    extractions: list[TickExtraction] = Field(
        default_factory=list,
        description="Extracted observations for indicators (one per tick per indicator)",
    )

    def to_dataframe(self) -> pl.DataFrame:
        """Convert extractions to a Polars DataFrame.

        Returns:
            DataFrame with columns: indicator (Utf8), value (Utf8), timestamp (Utf8).
            The timestamp column contains the tick ID (tick start time).
            Value column is stored as string for downstream encoding.
        """
        schema = {
            "indicator": pl.Utf8,
            "value": pl.Utf8,
            "timestamp": pl.Utf8,
        }
        if not self.extractions:
            return pl.DataFrame(schema=schema)

        rows = []
        for e in self.extractions:
            v = e.value
            if v is None:
                str_val = None
            elif isinstance(v, (bool, int, float)):
                str_val = str(v)
            elif isinstance(v, str):
                str_val = v
            else:
                str_val = None
            rows.append(
                {
                    "indicator": e.indicator,
                    "value": str_val,
                    "timestamp": e.tick,
                }
            )

        return pl.DataFrame(rows, schema=schema)


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


def validate_worker_output(
    data: dict,
    causal_spec: dict,
    expected_ticks: list[str] | None = None,
) -> tuple[WorkerOutput | None, list[str]]:
    """Validate worker output dict, collecting ALL errors instead of failing on first.

    Args:
        data: Dictionary to validate as WorkerOutput
        causal_spec: The CausalSpec dict to validate against
        expected_ticks: If provided, validate that extractions only reference
            these tick IDs and flag missing ticks.

    Returns:
        Tuple of (validated output or None, list of error messages)
    """
    errors = []

    # Basic structure checks
    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    extractions = data.get("extractions", [])

    if not isinstance(extractions, list):
        errors.append("'extractions' must be a list")
        extractions = []

    # Build set of valid indicator names and their dtypes
    indicator_info = _get_indicator_info(causal_spec)
    expected_tick_set = set(expected_ticks) if expected_ticks else None

    # Validate each extraction
    valid_extractions = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, ext_data in enumerate(extractions):
        if not isinstance(ext_data, dict):
            errors.append(f"extractions[{i}]: must be a dictionary")
            continue

        tick = ext_data.get("tick", "<missing>")
        ind_name = ext_data.get("indicator", "<missing>")
        value = ext_data.get("value")

        # Check tick is valid
        if expected_tick_set is not None and tick not in expected_tick_set:
            errors.append(f"extractions[{i}]: tick '{tick}' not in expected ticks")
            continue

        # Check indicator exists
        if ind_name not in indicator_info:
            valid_ind_names = ", ".join(sorted(indicator_info.keys()))
            errors.append(
                f"extractions[{i}]: indicator '{ind_name}' not in indicators. "
                f"Valid indicators: {valid_ind_names}"
            )
            continue

        # Check no duplicate (tick, indicator) pairs
        pair = (tick, ind_name)
        if pair in seen_pairs:
            errors.append(
                f"extractions[{i}]: duplicate (tick, indicator) pair: ({tick}, {ind_name})"
            )
            continue
        seen_pairs.add(pair)

        # Check dtype match
        expected_dtype = indicator_info[ind_name]["dtype"]
        if not _check_dtype_match(value, expected_dtype):
            errors.append(
                f"extractions[{i}]: value {value!r} for '{ind_name}' doesn't match "
                f"expected dtype '{expected_dtype}'"
            )
            continue

        normalized = {
            "tick": tick,
            "indicator": ind_name,
            "value": value,
        }

        # Validate via Pydantic
        try:
            ext = TickExtraction.model_validate(normalized)
            valid_extractions.append(ext)
        except Exception as e:
            errors.append(f"extractions[{i}] ({ind_name}): {e}")

    # If no errors, build and return the output
    if not errors:
        try:
            output = WorkerOutput(extractions=valid_extractions)
            return output, []
        except Exception as e:
            errors.append(f"Final validation failed: {e}")

    return None, errors
