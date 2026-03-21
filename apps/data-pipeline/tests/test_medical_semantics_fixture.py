"""Contract checks for the tracked MEDICAL_SEMANTICS Stage 2 fixture."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from causal_ssm_agent.utils.causal_spec import get_indicators
from causal_ssm_agent.utils.data import bucket_by_clock, detect_time_column

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "MEDICAL_SEMANTICS"
RUN_DIR = FIXTURE_DIR / "run"
EXPECTED_SHAPE = json.loads((FIXTURE_DIR / "expected-stage2-shape.json").read_text())
EXPECTED_SUPPORT_STARTS = EXPECTED_SHAPE["expected_support_starts"]
EXPECTED_SEMANTICS = {
    indicator: (
        semantics["support_kind"],
        semantics["summary_operator"],
        semantics["anchor_policy"],
    )
    for indicator, semantics in EXPECTED_SHAPE["indicators"].items()
}
EXPECTED_STAGE2_COLUMNS = EXPECTED_SHAPE["stage2_columns"]


def _load_stage1b() -> dict:
    return json.loads((RUN_DIR / "stage-1b.json").read_text())["causal_spec"]


def _expected_support_ends(window: str) -> list[str]:
    return (
        pl.DataFrame({"support_start": EXPECTED_SUPPORT_STARTS})
        .with_columns(
            pl.col("support_start")
            .str.to_datetime(strict=False)
            .dt.offset_by(window)
            .dt.to_string("%Y-%m-%dT%H:%M:%S")
            .alias("support_end")
        )["support_end"]
        .to_list()
    )


def _model_as_strings(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("anchor_time").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("anchor_time"),
        pl.col("support_start").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("support_start"),
        pl.col("support_end").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("support_end"),
    )


def test_medical_semantics_stage2_fixture_contract() -> None:
    causal_spec = _load_stage1b()
    raw = pl.read_parquet(RUN_DIR / "stage2-raw-data.parquet")
    model = pl.read_parquet(RUN_DIR / "stage2-model-data.parquet")
    stage0 = pl.read_parquet(RUN_DIR / "stage0-raw-input.parquet")
    stage2 = json.loads((RUN_DIR / "stage-2.json").read_text())

    assert raw.columns == EXPECTED_STAGE2_COLUMNS
    assert model.columns == EXPECTED_STAGE2_COLUMNS
    assert raw.schema["value"] == getattr(pl, EXPECTED_SHAPE["raw_schema"]["value"])
    assert model.schema["value"] == getattr(pl, EXPECTED_SHAPE["model_schema"]["value"])
    assert str(raw.schema["anchor_time"]) == EXPECTED_SHAPE["raw_schema"]["anchor_time"]
    assert str(raw.schema["support_start"]) == EXPECTED_SHAPE["raw_schema"]["support_start"]
    assert str(raw.schema["support_end"]) == EXPECTED_SHAPE["raw_schema"]["support_end"]
    assert str(model.schema["anchor_time"]).startswith(EXPECTED_SHAPE["model_schema"]["anchor_time"])
    assert str(model.schema["support_start"]).startswith(EXPECTED_SHAPE["model_schema"]["support_start"])
    assert str(model.schema["support_end"]).startswith(EXPECTED_SHAPE["model_schema"]["support_end"])

    model_clock = causal_spec["measurement"]["model_clock"]
    assert model_clock == EXPECTED_SHAPE["model_clock"]
    assert len(EXPECTED_SUPPORT_STARTS) == EXPECTED_SHAPE["support_window_count"]
    assert len(EXPECTED_SEMANTICS) == EXPECTED_SHAPE["indicator_count"]

    time_col = detect_time_column(stage0)
    observed_support_starts = [
        start.replace("+00:00", "") for start, _ in bucket_by_clock(stage0, model_clock, time_col)
    ]
    assert observed_support_starts == EXPECTED_SUPPORT_STARTS

    indicators = get_indicators(causal_spec)
    assert [ind["name"] for ind in indicators] == list(EXPECTED_SEMANTICS)

    expected_rows = EXPECTED_SHAPE["row_count"]
    assert raw.height == expected_rows
    assert model.height == expected_rows
    assert stage2["per_indicator_counts"] == {
        indicator: len(EXPECTED_SUPPORT_STARTS) for indicator in EXPECTED_SEMANTICS
    }

    expected_support_ends = _expected_support_ends(model_clock)
    model_as_strings = _model_as_strings(model)

    raw_pairs = raw.select("indicator", "support_start").sort("indicator", "support_start")
    model_pairs = model_as_strings.select("indicator", "support_start").sort(
        "indicator", "support_start"
    )
    assert raw_pairs.equals(model_pairs)
    assert raw_pairs.height == raw_pairs.unique().height

    stage1b_lookup = {ind["name"]: ind for ind in indicators}
    for indicator, (support_kind, summary_operator, anchor_policy) in EXPECTED_SEMANTICS.items():
        stage1b_indicator = stage1b_lookup[indicator]
        assert stage1b_indicator["support_kind"] == support_kind
        assert stage1b_indicator["summary_operator"] == summary_operator
        assert stage1b_indicator["anchor_policy"] == anchor_policy
        assert stage1b_indicator["observation_window"] is None

        raw_subset = raw.filter(pl.col("indicator") == indicator).sort("support_start")
        model_subset = model_as_strings.filter(pl.col("indicator") == indicator).sort(
            "support_start"
        )

        assert raw_subset.height == len(EXPECTED_SUPPORT_STARTS)
        assert model_subset.height == len(EXPECTED_SUPPORT_STARTS)
        assert raw_subset["support_start"].to_list() == EXPECTED_SUPPORT_STARTS
        assert raw_subset["support_end"].to_list() == expected_support_ends
        assert raw_subset["anchor_time"].to_list() == expected_support_ends
        assert raw_subset["observation_window"].to_list() == ["1d"] * len(EXPECTED_SUPPORT_STARTS)
        assert raw_subset["support_kind"].to_list() == [support_kind] * len(EXPECTED_SUPPORT_STARTS)
        assert raw_subset["summary_operator"].to_list() == [summary_operator] * len(
            EXPECTED_SUPPORT_STARTS
        )
        assert raw_subset["anchor_policy"].to_list() == [anchor_policy] * len(
            EXPECTED_SUPPORT_STARTS
        )

        assert model_subset["support_start"].to_list() == EXPECTED_SUPPORT_STARTS
        assert model_subset["support_end"].to_list() == expected_support_ends
        assert model_subset["anchor_time"].to_list() == expected_support_ends
        assert model_subset["support_kind"].to_list() == [support_kind] * len(
            EXPECTED_SUPPORT_STARTS
        )
        assert model_subset["summary_operator"].to_list() == [summary_operator] * len(
            EXPECTED_SUPPORT_STARTS
        )
        assert model_subset["anchor_policy"].to_list() == [anchor_policy] * len(
            EXPECTED_SUPPORT_STARTS
        )
        assert model_subset["observation_window"].to_list() == ["1d"] * len(EXPECTED_SUPPORT_STARTS)
