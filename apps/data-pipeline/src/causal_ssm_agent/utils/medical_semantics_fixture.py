"""Shared MEDICAL_SEMANTICS fixture utilities for tests and evals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from causal_ssm_agent.utils.causal_spec import get_indicators
from causal_ssm_agent.utils.data import bucket_by_clock, detect_time_column

FIXTURE_USER_ID = "MEDICAL_SEMANTICS"
EXPECTED_STAGE2_COLUMNS = [
    "indicator",
    "value",
    "anchor_time",
    "support_kind",
    "summary_operator",
    "anchor_policy",
    "observation_window",
    "support_start",
    "support_end",
]
EXPECTED_SCHEMA_OVERRIDES = {
    "indicator": pl.String,
    "value": pl.Float64,
    "anchor_time": pl.String,
    "support_kind": pl.String,
    "summary_operator": pl.String,
    "anchor_policy": pl.String,
    "observation_window": pl.String,
    "support_start": pl.String,
    "support_end": pl.String,
}


@dataclass(frozen=True)
class MedicalSemanticsFixture:
    """Tracked MEDICAL_SEMANTICS fixture inputs."""

    fixture_dir: Path
    run_dir: Path
    question: str
    stage0: pl.DataFrame
    column_descriptions: dict[str, str]
    expected_model: pl.DataFrame


@dataclass(frozen=True)
class ComparisonLevel:
    """One ordered comparison level in the fixture contract."""

    name: str
    description: str
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def issue_count(self) -> int:
        return len(self.issues)


@dataclass(frozen=True)
class MedicalSemanticsComparison:
    """Ordered comparison result for the MEDICAL_SEMANTICS fixture."""

    levels: tuple[ComparisonLevel, ...]
    summary: dict[str, Any]
    stage1b_indicators: tuple[str, ...]

    def all_issues(self) -> list[str]:
        return [issue for level in self.levels for issue in level.issues]

    def rank_key(self) -> tuple[int, ...]:
        return tuple(0 if level.passed else 1 for level in self.levels) + tuple(
            level.issue_count for level in self.levels
        )

    def format_report(self, *, max_issues_per_level: int = 8) -> str:
        lines = [
            "MEDICAL_SEMANTICS comparison summary",
            f"- rows: {self.summary['actual_rows']} actual vs {self.summary['expected_rows']} expected",
            f"- stage1b indicators: {self.summary['actual_indicator_count']} actual vs {self.summary['expected_indicator_count']} expected",
            f"- rank key: {self.rank_key()} (lower is better)",
            "",
            "Ordered comparison levels:",
        ]
        for idx, level in enumerate(self.levels, start=1):
            status = "PASS" if level.passed else "FAIL"
            lines.append(f"{idx}. {level.name}: {status} ({level.issue_count} issues)")
            lines.append(f"   {level.description}")
            if level.issues:
                for issue in level.issues[:max_issues_per_level]:
                    lines.append(f"   - {issue}")
                remaining = level.issue_count - max_issues_per_level
                if remaining > 0:
                    lines.append(f"   - ... {remaining} more issues")
        return "\n".join(lines)


def _find_fixture_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "data" / FIXTURE_USER_ID
        if candidate.exists():
            return candidate
    return Path.cwd() / "data" / FIXTURE_USER_ID


def _load_expected_table(fixture_dir: Path) -> pl.DataFrame:
    return pl.read_csv(
        fixture_dir / "expected-stage2-model-data.csv",
        null_values="",
        schema_overrides=EXPECTED_SCHEMA_OVERRIDES,
    )


def _column_descriptions_from_stage0_payload(stage0_payload: dict[str, Any]) -> dict[str, str]:
    descriptions = stage0_payload.get("column_descriptions", [])
    if not isinstance(descriptions, list):
        return {}
    return {
        str(item.get("name")): str(item.get("description", ""))
        for item in descriptions
        if isinstance(item, dict) and item.get("name")
    }


def load_medical_semantics_fixture() -> MedicalSemanticsFixture:
    """Load the tracked MEDICAL_SEMANTICS fixture inputs."""

    fixture_dir = _find_fixture_dir()
    run_dir = fixture_dir / "run"
    stage0_payload = json.loads((run_dir / "stage-0.json").read_text())
    expected_model = _load_expected_table(fixture_dir)
    return MedicalSemanticsFixture(
        fixture_dir=fixture_dir,
        run_dir=run_dir,
        question=(fixture_dir / "query.txt").read_text().strip(),
        stage0=pl.read_parquet(run_dir / "stage0-raw-input.parquet"),
        column_descriptions=_column_descriptions_from_stage0_payload(stage0_payload),
        expected_model=expected_model,
    )


def _add_issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def _sorted_rows(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(EXPECTED_STAGE2_COLUMNS)


def _model_as_strings(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("anchor_time").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("anchor_time"),
        pl.col("support_start").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("support_start"),
        pl.col("support_end").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("support_end"),
    )


def _diff_rows(actual: pl.DataFrame, expected: pl.DataFrame) -> str | None:
    actual_sorted = _sorted_rows(actual)
    expected_sorted = _sorted_rows(expected)
    if actual_sorted.equals(expected_sorted):
        return None

    only_actual = actual_sorted.join(expected_sorted, on=EXPECTED_STAGE2_COLUMNS, how="anti").head(
        3
    )
    only_expected = expected_sorted.join(
        actual_sorted, on=EXPECTED_STAGE2_COLUMNS, how="anti"
    ).head(3)
    return (
        f"unexpected rows sample={only_actual.to_dicts()} "
        f"missing rows sample={only_expected.to_dicts()}"
    )


def _expected_support_starts(expected: pl.DataFrame) -> list[str]:
    return (
        expected.select("support_start").unique().sort("support_start")["support_start"].to_list()
    )


def _expected_support_ends(expected_support_starts: list[str], window: str) -> list[str]:
    return (
        pl.DataFrame({"support_start": expected_support_starts})
        .with_columns(
            pl.col("support_start")
            .str.to_datetime(strict=False)
            .dt.offset_by(window)
            .dt.to_string("%Y-%m-%dT%H:%M:%S")
            .alias("support_end")
        )["support_end"]
        .to_list()
    )


def _expected_semantics(expected: pl.DataFrame) -> dict[str, tuple[str, str, str]]:
    return {
        indicator: (
            subset["support_kind"][0],
            subset["summary_operator"][0],
            subset["anchor_policy"][0],
        )
        for indicator, subset in {
            indicator: expected.filter(pl.col("indicator") == indicator)
            for indicator in expected["indicator"].unique().sort()
        }.items()
    }


def compare_medical_semantics_outputs(
    *,
    causal_spec: dict,
    stage0: pl.DataFrame,
    data_for_model: pl.DataFrame,
    expected_model: pl.DataFrame,
) -> MedicalSemanticsComparison:
    """Compare candidate Stage 2 outputs against the tracked fixture."""

    stage1b_surface_issues: list[str] = []
    stage2_structure_issues: list[str] = []
    stage2_value_issues: list[str] = []

    expected_support_starts = _expected_support_starts(expected_model)
    expected_semantics = _expected_semantics(expected_model)
    expected_support_ends = _expected_support_ends(
        expected_support_starts,
        causal_spec["measurement"]["model_clock"],
    )
    model_as_strings = _model_as_strings(data_for_model)

    indicators = get_indicators(causal_spec)
    stage1b_lookup = {ind["name"]: ind for ind in indicators}

    _add_issue(
        stage1b_surface_issues,
        causal_spec["measurement"]["model_clock"] == "1d",
        f"model_clock mismatch: {causal_spec['measurement']['model_clock']}",
    )
    _add_issue(
        stage1b_surface_issues,
        {ind["name"] for ind in indicators} == set(expected_semantics),
        "indicator set mismatch between stage-1b and expected Stage 2 tables",
    )

    for indicator, (support_kind, summary_operator, anchor_policy) in expected_semantics.items():
        stage1b_indicator = stage1b_lookup.get(indicator)
        if stage1b_indicator is None:
            stage1b_surface_issues.append(f"missing stage-1b indicator: {indicator}")
            continue

        _add_issue(
            stage1b_surface_issues,
            stage1b_indicator["support_kind"] == support_kind,
            f"{indicator} support_kind mismatch: actual={stage1b_indicator['support_kind']} expected={support_kind}",
        )
        _add_issue(
            stage1b_surface_issues,
            stage1b_indicator["summary_operator"] == summary_operator,
            f"{indicator} summary_operator mismatch: actual={stage1b_indicator['summary_operator']} expected={summary_operator}",
        )
        _add_issue(
            stage1b_surface_issues,
            stage1b_indicator["anchor_policy"] == anchor_policy,
            f"{indicator} anchor_policy mismatch: actual={stage1b_indicator['anchor_policy']} expected={anchor_policy}",
        )
        _add_issue(
            stage1b_surface_issues,
            stage1b_indicator["observation_window"] is None,
            f"{indicator} observation_window expected None in stage-1b but got {stage1b_indicator['observation_window']}",
        )

    _add_issue(
        stage2_structure_issues,
        data_for_model.columns == EXPECTED_STAGE2_COLUMNS,
        f"columns mismatch: {data_for_model.columns}",
    )
    _add_issue(
        stage2_structure_issues,
        data_for_model.schema["value"] == pl.Float64,
        f"value dtype mismatch: {data_for_model.schema['value']}",
    )
    _add_issue(
        stage2_structure_issues,
        str(data_for_model.schema["anchor_time"]).startswith("Datetime("),
        f"anchor_time dtype mismatch: {data_for_model.schema['anchor_time']}",
    )

    time_col = detect_time_column(stage0)
    observed_support_starts = [
        start.replace("+00:00", "") for start, _ in bucket_by_clock(stage0, "1d", time_col)
    ]
    _add_issue(
        stage2_structure_issues,
        observed_support_starts == expected_support_starts,
        f"support starts mismatch: actual={observed_support_starts} expected={expected_support_starts}",
    )
    _add_issue(
        stage2_structure_issues,
        data_for_model.height == expected_model.height,
        f"row count mismatch: actual={data_for_model.height} expected={expected_model.height}",
    )
    _add_issue(
        stage2_structure_issues,
        dict(data_for_model.group_by("indicator").len().iter_rows())
        == {
            indicator: expected_model.filter(pl.col("indicator") == indicator).height
            for indicator in expected_semantics
        },
        "stage-2 per_indicator_counts mismatch",
    )

    model_pairs = model_as_strings.select("indicator", "support_start").sort(
        "indicator", "support_start"
    )
    _add_issue(
        stage2_structure_issues,
        model_pairs.height == model_pairs.unique().height,
        "duplicate (indicator, support_start) pairs in Stage 2 output",
    )

    model_diff = _diff_rows(model_as_strings, expected_model)
    if model_diff:
        stage2_value_issues.append(f"rows differ from expected fixture: {model_diff}")

    for indicator in expected_semantics:
        model_subset = model_as_strings.filter(pl.col("indicator") == indicator).sort(
            "support_start"
        )
        expected_model_subset = expected_model.filter(pl.col("indicator") == indicator)

        _add_issue(
            stage2_structure_issues,
            model_subset.height == len(expected_support_starts),
            f"{indicator} row count mismatch: actual={model_subset.height} expected={len(expected_support_starts)}",
        )
        _add_issue(
            stage2_structure_issues,
            model_subset["support_start"].to_list() == expected_support_starts,
            f"{indicator} support_start coverage mismatch",
        )
        _add_issue(
            stage2_structure_issues,
            model_subset["support_end"].to_list() == expected_support_ends,
            f"{indicator} support_end coverage mismatch",
        )
        _add_issue(
            stage2_structure_issues,
            model_subset["anchor_time"].to_list() == expected_support_ends,
            f"{indicator} anchor_time coverage mismatch",
        )

        model_subset_diff = _diff_rows(model_subset, expected_model_subset)
        if model_subset_diff:
            stage2_value_issues.append(f"{indicator} subset differs: {model_subset_diff}")

    return MedicalSemanticsComparison(
        levels=(
            ComparisonLevel(
                name="stage1b_surface",
                description="Indicator identity, support semantics, and observation-window choices.",
                issues=tuple(stage1b_surface_issues),
            ),
            ComparisonLevel(
                name="stage2_structure",
                description="Row shape, dtypes, support coverage, and per-indicator counts.",
                issues=tuple(stage2_structure_issues),
            ),
            ComparisonLevel(
                name="stage2_values",
                description="Row-for-row value agreement after structure is aligned.",
                issues=tuple(stage2_value_issues),
            ),
        ),
        summary={
            "actual_rows": data_for_model.height,
            "expected_rows": expected_model.height,
            "actual_indicator_count": len(indicators),
            "expected_indicator_count": len(expected_semantics),
        },
        stage1b_indicators=tuple(sorted(ind["name"] for ind in indicators)),
    )
