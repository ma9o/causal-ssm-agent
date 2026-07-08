"""Shared DEMO fixture utilities for tests and evals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from nof1_causal_lab.utils.causal_design import get_indicators

FIXTURE_USER_ID = "DEMO"
EXPECTED_EXTRACTION_COLUMNS = ["indicator", "value", "anchor_time"]


@dataclass(frozen=True)
class DemoHealthFixture:
    """Tracked DEMO fixture inputs (current versions in the artifact store)."""

    question: str
    raw_data_df: pl.DataFrame
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
class DemoHealthComparison:
    """Ordered comparison result for the DEMO fixture."""

    levels: tuple[ComparisonLevel, ...]
    summary: dict[str, Any]
    measurement_payload_indicators: tuple[str, ...]

    def all_issues(self) -> list[str]:
        return [issue for level in self.levels for issue in level.issues]

    def rank_key(self) -> tuple[int, ...]:
        return tuple(0 if level.passed else 1 for level in self.levels) + tuple(
            level.issue_count for level in self.levels
        )

    def format_report(self, *, max_issues_per_level: int = 8) -> str:
        lines = [
            "DEMO comparison summary",
            f"- rows: {self.summary['actual_rows']} actual vs {self.summary['expected_rows']} expected",
            f"- measurement_payload indicators: {self.summary['actual_indicator_count']} actual vs {self.summary['expected_indicator_count']} expected",
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


def _column_descriptions_from_raw_data_df_payload(
    raw_data_df_payload: dict[str, Any],
) -> dict[str, str]:
    descriptions = raw_data_df_payload.get("column_descriptions", [])
    if not isinstance(descriptions, list):
        return {}
    return {
        str(item.get("name")): str(item.get("description", ""))
        for item in descriptions
        if isinstance(item, dict) and item.get("name")
    }


def load_demo_health_fixture() -> DemoHealthFixture:
    """Load the tracked DEMO fixture from its artifact store."""
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal

    store = ArtifactStore(FIXTURE_USER_ID)
    state = EpisodeJournal(FIXTURE_USER_ID).latest_state()

    def _version(artifact_id: str) -> int:
        info = state.get(artifact_id)
        if info is None:
            raise FileNotFoundError(
                f"No current '{artifact_id}' artifact for workspace '{FIXTURE_USER_ID}'"
            )
        return info.version

    raw_version = _version("raw_data")
    profile = store.read_json_file("raw_data", raw_version, "profile.json")
    return DemoHealthFixture(
        question=store.read_json_file("question", _version("question"), "question.json")["text"],
        raw_data_df=store.read_parquet_file("raw_data", raw_version, "raw.parquet"),
        column_descriptions=_column_descriptions_from_raw_data_df_payload(profile),
        expected_model=store.read_parquet_file("panel", _version("panel"), "panel.parquet").select(
            EXPECTED_EXTRACTION_COLUMNS
        ),
    )


def _add_issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def _normalized_extraction(df: pl.DataFrame) -> pl.DataFrame:
    available = [column for column in EXPECTED_EXTRACTION_COLUMNS if column in df.columns]
    normalized = df.select(available)
    missing = [column for column in EXPECTED_EXTRACTION_COLUMNS if column not in normalized.columns]
    for column in missing:
        normalized = normalized.with_columns(pl.lit(None).cast(pl.String).alias(column))

    return normalized.select(EXPECTED_EXTRACTION_COLUMNS).with_columns(
        pl.col("indicator").cast(pl.String),
        pl.col("value").cast(pl.Float64),
        pl.col("anchor_time")
        .cast(pl.String)
        .str.replace("T", " ")
        .str.replace(r"[Zz]$", "")
        .str.replace(r"[+-]\d{2}:\d{2}$", ""),
    )


def _sorted_rows(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(EXPECTED_EXTRACTION_COLUMNS)


def _diff_rows(actual: pl.DataFrame, expected: pl.DataFrame) -> str | None:
    actual_sorted = _sorted_rows(actual)
    expected_sorted = _sorted_rows(expected)
    if actual_sorted.equals(expected_sorted):
        return None

    only_actual = actual_sorted.join(
        expected_sorted, on=EXPECTED_EXTRACTION_COLUMNS, how="anti"
    ).head(3)
    only_expected = expected_sorted.join(
        actual_sorted, on=EXPECTED_EXTRACTION_COLUMNS, how="anti"
    ).head(3)
    return (
        f"unexpected rows sample={only_actual.to_dicts()} "
        f"missing rows sample={only_expected.to_dicts()}"
    )


def _per_indicator_counts(df: pl.DataFrame) -> dict[str, int]:
    return dict(df.group_by("indicator").len().sort("indicator").iter_rows())


def compare_demo_health_outputs(
    *,
    causal_design: dict,
    raw_data_df: pl.DataFrame,
    data_for_model: pl.DataFrame,
    expected_model: pl.DataFrame,
) -> DemoHealthComparison:
    """Compare candidate extraction outputs against the tracked fixture."""

    del raw_data_df
    measurement_payload_surface_issues: list[str] = []
    extraction_structure_issues: list[str] = []
    extraction_value_issues: list[str] = []

    actual = _normalized_extraction(data_for_model)
    expected = _normalized_extraction(expected_model)

    indicators = get_indicators(causal_design)
    actual_indicator_names = {ind["name"] for ind in indicators}
    expected_indicator_names = set(expected["indicator"].unique())

    _add_issue(
        measurement_payload_surface_issues,
        actual_indicator_names == expected_indicator_names,
        "indicator set mismatch between measurement-structure and expected extraction tables",
    )
    _add_issue(
        extraction_structure_issues,
        data_for_model.columns[: len(EXPECTED_EXTRACTION_COLUMNS)] == EXPECTED_EXTRACTION_COLUMNS,
        f"leading columns mismatch: {data_for_model.columns}",
    )
    _add_issue(
        extraction_structure_issues,
        actual.schema["value"] == pl.Float64,
        f"value dtype mismatch: {actual.schema['value']}",
    )
    _add_issue(
        extraction_structure_issues,
        actual.height == expected.height,
        f"row count mismatch: actual={actual.height} expected={expected.height}",
    )
    _add_issue(
        extraction_structure_issues,
        _per_indicator_counts(actual) == _per_indicator_counts(expected),
        "extraction per_indicator_counts mismatch",
    )

    model_diff = _diff_rows(actual, expected)
    if model_diff:
        extraction_value_issues.append(f"rows differ from expected fixture: {model_diff}")

    return DemoHealthComparison(
        levels=(
            ComparisonLevel(
                name="measurement_payload_surface",
                description="Indicator identity expected by the DEMO fixture.",
                issues=tuple(measurement_payload_surface_issues),
            ),
            ComparisonLevel(
                name="extraction_structure",
                description="Row shape, dtypes, and per-indicator counts.",
                issues=tuple(extraction_structure_issues),
            ),
            ComparisonLevel(
                name="extraction_values",
                description="Row-for-row value agreement after structure is aligned.",
                issues=tuple(extraction_value_issues),
            ),
        ),
        summary={
            "actual_rows": actual.height,
            "expected_rows": expected.height,
            "actual_indicator_count": len(indicators),
            "expected_indicator_count": len(expected_indicator_names),
        },
        measurement_payload_indicators=tuple(sorted(ind["name"] for ind in indicators)),
    )
