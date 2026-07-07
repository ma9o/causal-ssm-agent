"""Stage 2 materialization helpers."""

from __future__ import annotations

from typing import Any, cast

import polars as pl


def materialize_stage2_outputs(
    stage2_result: dict,
    measurement_structure: dict,
) -> dict[str, Any]:
    """Materialize the Stage 2 observation table from a serialized extraction result."""
    from nof1_causal_lab.utils.aggregations import _encode_non_continuous
    from nof1_causal_lab.utils.data import ObservationRecord, observation_row_schema

    observation_dicts = cast("list[ObservationRecord]", stage2_result.get("observation_rows", []))
    if observation_dicts:
        data_for_model = pl.DataFrame(observation_dicts)
    else:
        data_for_model = pl.DataFrame(schema=observation_row_schema())

    if len(data_for_model) > 0:
        dtype_lookup = {
            indicator["name"]: indicator.get("measurement_dtype", "continuous")
            for indicator in measurement_structure.get("indicators", [])
            if indicator.get("name")
        }
        ordinal_levels_lookup: dict[str, list[str]] = {
            ind["name"]: ind["ordinal_levels"]
            for ind in measurement_structure.get("indicators", [])
            if ind.get("ordinal_levels")
        }
        data_for_model = _encode_non_continuous(data_for_model, dtype_lookup, ordinal_levels_lookup)
        data_for_model = data_for_model.with_columns(
            pl.col("value").cast(pl.Float64, strict=False).alias("value"),
            pl.col("anchor_time")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("anchor_time"),
            pl.col("support_start")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("support_start"),
            pl.col("support_end")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("support_end"),
        ).drop_nulls(subset=["anchor_time"])
        data_for_model = data_for_model.sort("indicator", "anchor_time")

    return {
        "data_for_model": data_for_model,
        "worker_statuses": stage2_result.get("worker_statuses", []),
        "llm_trace": stage2_result.get("llm_trace"),
    }
