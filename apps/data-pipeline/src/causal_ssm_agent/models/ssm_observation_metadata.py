"""Shared observation metadata helpers for SSM model preparation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

NON_MANIFEST_COLUMNS = {"time"}
SECONDS_PER_DAY = 86400.0


@dataclass
class ObservationSupportRuntime:
    """Structured support metadata aligned to the prepared wide observation matrix."""

    anchor_times: np.ndarray  # shape (T,)
    manifest_names: list[str]
    support_kinds: list[str | None]
    summary_operators: list[str | None]
    anchor_policies: list[str | None]
    observation_windows: list[str | None]
    support_start_times: np.ndarray  # shape (T, n_manifest), NaN when missing
    support_end_times: np.ndarray  # shape (T, n_manifest), NaN when missing
    interval_prev_coeffs: np.ndarray  # shape (T, n_manifest, n_slots)
    interval_curr_coeffs: np.ndarray  # shape (T, n_manifest, n_slots)
    interval_weights: np.ndarray  # shape (T, n_manifest, n_slots)
    emission_slot_indices: np.ndarray  # shape (T, n_manifest), -1 when not emitted

    @property
    def requires_interval_summary_handling(self) -> bool:
        """Whether any manifest requires interval-summary measurement handling."""
        return any(kind == "interval" for kind in self.support_kinds)

    @property
    def interval_summary_manifest_names(self) -> list[str]:
        """Manifest names that require interval-summary measurement handling."""
        return [
            name
            for name, kind in zip(self.manifest_names, self.support_kinds, strict=False)
            if kind == "interval"
        ]

    @property
    def max_active_windows(self) -> int:
        """Maximum number of concurrent interval-summary windows per manifest."""
        return int(self.interval_prev_coeffs.shape[2]) if self.interval_prev_coeffs.ndim == 3 else 0


def default_manifest_columns(X: Any) -> list[str]:
    """Infer manifest columns from a wide dataframe-like object."""
    return [c for c in X.columns if c not in NON_MANIFEST_COLUMNS and not str(c).endswith("_lag1")]


def _datetime_expr(df: pl.DataFrame, column: str) -> pl.Expr:
    """Parse a datetime-like column to a consistent expression."""
    if column not in df.columns:
        return pl.lit(None, dtype=pl.Datetime(time_zone="UTC"))
    if df.schema.get(column) == pl.Utf8:
        return pl.col(column).str.to_datetime(strict=False, time_zone="UTC")
    return pl.col(column).cast(pl.Datetime, strict=False)


def _pivot_support_matrix(
    support_df: pl.DataFrame,
    *,
    value_col: str,
    base_times: pl.DataFrame,
    manifest_names: list[str],
) -> np.ndarray:
    """Pivot one support-time column to a dense matrix aligned with wide_data rows."""
    pivoted = (
        support_df.select("time", "indicator", value_col)
        .pivot(on="indicator", index="time", values=value_col, aggregate_function="first")
        .sort("time")
    )
    aligned = base_times.join(pivoted, on="time", how="left")
    for manifest in manifest_names:
        if manifest not in aligned.columns:
            aligned = aligned.with_columns(pl.lit(None, dtype=pl.Float64).alias(manifest))
    return aligned.select(manifest_names).to_numpy()


def _requires_interval_summary_support(support_kind: str | None) -> bool:
    return support_kind == "interval"


def _assign_support_slots(
    anchor_times: np.ndarray,
    support_start_times: np.ndarray,
    support_end_times: np.ndarray,
    support_kinds: list[str | None],
    manifest_names: list[str],
) -> tuple[list[list[tuple[float, float, int, int]]], int]:
    """Assign concurrent interval windows to reusable slots per manifest."""
    tol = 1e-8
    manifest_windows: list[list[tuple[float, float, int, int]]] = []
    max_slots = 0

    for manifest_idx, manifest_name in enumerate(manifest_names):
        support_kind = support_kinds[manifest_idx]
        if not _requires_interval_summary_support(support_kind):
            manifest_windows.append([])
            continue

        starts = support_start_times[:, manifest_idx]
        ends = support_end_times[:, manifest_idx]
        valid_rows = np.where(np.isfinite(starts) & np.isfinite(ends))[0]
        if valid_rows.size == 0:
            manifest_windows.append([])
            continue

        windows: list[tuple[float, float, int]] = []
        for row_idx in valid_rows:
            start = float(starts[row_idx])
            end = float(ends[row_idx])
            anchor = float(anchor_times[row_idx])
            if end + tol < start:
                raise ValueError(
                    f"Indicator '{manifest_name}' has support_end before support_start "
                    f"at row {row_idx}: {start} -> {end}"
                )
            if abs(anchor - end) > tol:
                raise ValueError(
                    f"Indicator '{manifest_name}' has support_end={end} that does not match "
                    f"its anchored observation time {anchor} at row {row_idx}."
                )
            windows.append((start, end, int(row_idx)))

        windows.sort(key=lambda item: (item[0], item[1], item[2]))
        if windows[0][0] < anchor_times[0] - tol:
            raise ValueError(
                f"Indicator '{manifest_name}' has support starting before the first model time. "
                "Add earlier model-clock rows or shift the observation anchor."
            )

        assigned: list[tuple[float, float, int, int]] = []
        active_slots: list[tuple[float, int]] = []
        free_slots: list[int] = []
        n_slots = 0
        for start, end, row_idx in windows:
            while active_slots and active_slots[0][0] <= start + tol:
                _finished_end, finished_slot = heapq.heappop(active_slots)
                heapq.heappush(free_slots, finished_slot)

            if free_slots:
                slot_idx = heapq.heappop(free_slots)
            else:
                slot_idx = n_slots
                n_slots += 1

            assigned.append((start, end, row_idx, slot_idx))
            heapq.heappush(active_slots, (end, slot_idx))

        manifest_windows.append(assigned)
        max_slots = max(max_slots, n_slots)

    return manifest_windows, max_slots


def _compile_interval_support_coefficients(
    anchor_times: np.ndarray,
    support_start_times: np.ndarray,
    support_end_times: np.ndarray,
    support_kinds: list[str | None],
    manifest_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compile per-interval trapezoidal coefficients for concurrent support windows."""
    T = anchor_times.shape[0]
    n_manifest = support_start_times.shape[1]
    assigned_windows, max_slots = _assign_support_slots(
        anchor_times,
        support_start_times,
        support_end_times,
        support_kinds,
        manifest_names,
    )
    n_slots = max(max_slots, 1)
    prev_coeffs = np.zeros((T, n_manifest, n_slots), dtype=np.float64)
    curr_coeffs = np.zeros((T, n_manifest, n_slots), dtype=np.float64)
    weights = np.zeros((T, n_manifest, n_slots), dtype=np.float64)
    emission_slots = np.full((T, n_manifest), -1, dtype=np.int32)
    tol = 1e-8

    for manifest_idx, _manifest_name in enumerate(manifest_names):
        support_kind = support_kinds[manifest_idx]
        if not _requires_interval_summary_support(support_kind):
            continue
        windows = assigned_windows[manifest_idx]
        if not windows:
            continue

        for start, end, row_idx, slot_idx in windows:
            emission_slots[row_idx, manifest_idx] = slot_idx
            for step_idx in range(1, T):
                interval_start = float(anchor_times[step_idx - 1])
                interval_end = float(anchor_times[step_idx])
                dt = interval_end - interval_start
                if dt <= tol:
                    continue

                overlap_start = max(interval_start, start)
                overlap_end = min(interval_end, end)
                if overlap_end <= overlap_start + tol:
                    continue

                overlap = overlap_end - overlap_start
                alpha_start = (overlap_start - interval_start) / dt
                alpha_end = (overlap_end - interval_start) / dt
                prev_coeffs[step_idx, manifest_idx, slot_idx] = overlap * (
                    1.0 - 0.5 * (alpha_start + alpha_end)
                )
                curr_coeffs[step_idx, manifest_idx, slot_idx] = (
                    overlap * 0.5 * (alpha_start + alpha_end)
                )
                weights[step_idx, manifest_idx, slot_idx] = overlap

    return prev_coeffs, curr_coeffs, weights, emission_slots


def compile_observation_support_runtime(
    observation_data: pl.DataFrame | None,
    wide_data: pl.DataFrame,
    manifest_names: list[str],
) -> ObservationSupportRuntime | None:
    """Compile long-format observation support metadata into wide aligned arrays."""
    if (
        observation_data is None
        or observation_data.is_empty()
        or not manifest_names
        or "anchor_time" not in observation_data.columns
        or "support_start" not in observation_data.columns
        or "support_end" not in observation_data.columns
    ):
        return None

    if "indicator" not in observation_data.columns:
        return None

    df = observation_data
    required_semantics = ("support_kind", "summary_operator", "anchor_policy", "observation_window")
    missing_semantics = [col_name for col_name in required_semantics if col_name not in df.columns]
    if missing_semantics:
        missing_display = ", ".join(missing_semantics)
        raise ValueError(
            f"Observation data is missing canonical support semantics columns: {missing_display}."
        )

    df = df.with_columns(
        _datetime_expr(df, "anchor_time").alias("__anchor_dt"),
        _datetime_expr(df, "support_start").alias("__support_start_dt"),
        _datetime_expr(df, "support_end").alias("__support_end_dt"),
    ).drop_nulls(subset=["__anchor_dt"])

    if df.is_empty():
        return None

    t0 = df.select(pl.col("__anchor_dt").min()).item()
    df = df.with_columns(
        ((pl.col("__anchor_dt") - pl.lit(t0)).dt.total_seconds() / SECONDS_PER_DAY)
        .cast(pl.Float64)
        .alias("time"),
        ((pl.col("__support_start_dt") - pl.lit(t0)).dt.total_seconds() / SECONDS_PER_DAY)
        .cast(pl.Float64)
        .alias("__support_start_time"),
        ((pl.col("__support_end_dt") - pl.lit(t0)).dt.total_seconds() / SECONDS_PER_DAY)
        .cast(pl.Float64)
        .alias("__support_end_time"),
    )

    if "time" not in wide_data.columns:
        return None
    base_times = wide_data.select(pl.col("time").cast(pl.Float64).alias("time"))
    anchor_times = base_times["time"].to_numpy()

    support_start_times = _pivot_support_matrix(
        df,
        value_col="__support_start_time",
        base_times=base_times,
        manifest_names=manifest_names,
    )
    support_end_times = _pivot_support_matrix(
        df,
        value_col="__support_end_time",
        base_times=base_times,
        manifest_names=manifest_names,
    )

    kind_window_rows = (
        df.group_by("indicator")
        .agg(
            pl.col("support_kind").drop_nulls().first().alias("support_kind"),
            pl.col("summary_operator").drop_nulls().first().alias("summary_operator"),
            pl.col("anchor_policy").drop_nulls().first().alias("anchor_policy"),
            pl.col("observation_window").drop_nulls().first().alias("observation_window"),
        )
        .iter_rows(named=True)
    )
    kind_window_lookup = {row["indicator"]: row for row in kind_window_rows}
    support_kinds = [
        kind_window_lookup.get(name, {}).get("support_kind") for name in manifest_names
    ]
    summary_operators = [
        kind_window_lookup.get(name, {}).get("summary_operator") for name in manifest_names
    ]
    anchor_policies = [
        kind_window_lookup.get(name, {}).get("anchor_policy") for name in manifest_names
    ]
    observation_windows = [
        kind_window_lookup.get(name, {}).get("observation_window") for name in manifest_names
    ]
    interval_prev_coeffs, interval_curr_coeffs, interval_weights, emission_slot_indices = (
        _compile_interval_support_coefficients(
            anchor_times,
            support_start_times,
            support_end_times,
            support_kinds,
            manifest_names,
        )
    )

    return ObservationSupportRuntime(
        anchor_times=anchor_times,
        manifest_names=manifest_names,
        support_kinds=support_kinds,
        summary_operators=summary_operators,
        anchor_policies=anchor_policies,
        observation_windows=observation_windows,
        support_start_times=support_start_times,
        support_end_times=support_end_times,
        interval_prev_coeffs=interval_prev_coeffs,
        interval_curr_coeffs=interval_curr_coeffs,
        interval_weights=interval_weights,
        emission_slot_indices=emission_slot_indices,
    )


def augment_wide_data_with_support_boundaries(
    observation_data: pl.DataFrame | None,
    wide_data: pl.DataFrame,
    manifest_names: list[str],
) -> pl.DataFrame:
    """Add missing support-boundary rows to the wide matrix.

    Interval-summary observations may begin before the earliest anchored
    observation time. The support-aware likelihood needs those boundary times
    on the latent path, even when all manifests are missing there.
    """
    if (
        observation_data is None
        or observation_data.is_empty()
        or wide_data.is_empty()
        or "time" not in wide_data.columns
        or not manifest_names
        or "anchor_time" not in observation_data.columns
        or "support_start" not in observation_data.columns
        or "support_end" not in observation_data.columns
    ):
        return wide_data

    df = observation_data.with_columns(
        _datetime_expr(observation_data, "anchor_time").alias("__anchor_dt"),
        _datetime_expr(observation_data, "support_start").alias("__support_start_dt"),
        _datetime_expr(observation_data, "support_end").alias("__support_end_dt"),
        pl.col("support_kind").alias("__support_kind"),
    ).drop_nulls(subset=["__anchor_dt"])
    if df.is_empty():
        return wide_data

    interval_df = df.filter(pl.col("__support_kind") == "interval")
    if interval_df.is_empty():
        return wide_data

    t0 = df.select(pl.col("__anchor_dt").min()).item()
    boundary_times = (
        pl.concat(
            [
                interval_df.select(pl.col("__anchor_dt").alias("__boundary_dt")),
                interval_df.select(pl.col("__support_start_dt").alias("__boundary_dt")),
                interval_df.select(pl.col("__support_end_dt").alias("__boundary_dt")),
            ],
            how="vertical_relaxed",
        )
        .drop_nulls(subset=["__boundary_dt"])
        .unique()
        .sort("__boundary_dt")
        .with_columns(
            ((pl.col("__boundary_dt") - pl.lit(t0)).dt.total_seconds() / SECONDS_PER_DAY)
            .cast(pl.Float64)
            .alias("time")
        )
        .select("time")
    )

    if boundary_times.is_empty():
        return wide_data

    current_times = wide_data.select(pl.col("time").cast(pl.Float64).alias("time"))
    missing_times = boundary_times.join(current_times, on="time", how="anti")
    if missing_times.is_empty():
        return wide_data.sort("time")

    filler_columns = [
        pl.lit(None, dtype=wide_data.schema.get(name, pl.Float64)).alias(name)
        for name in manifest_names
        if name in wide_data.columns
    ]
    missing_rows = missing_times.with_columns(filler_columns).select(wide_data.columns)
    return pl.concat([wide_data, missing_rows], how="vertical_relaxed").sort("time")


def resolve_manifest_metadata(
    spec: SSMSpec,
    X: Any,
) -> tuple[list[str], list[DistributionFamily]]:
    """Resolve manifest column names and per-channel distribution families."""
    manifest_dists = spec.manifest_dists or [spec.manifest_dist] * spec.n_manifest
    manifest_cols = spec.manifest_names or default_manifest_columns(X)
    if len(manifest_cols) != spec.n_manifest:
        raise ValueError(
            "Wide data columns do not match SSMSpec manifest dimensionality: "
            f"{len(manifest_cols)} vs {spec.n_manifest}"
        )
    return manifest_cols, manifest_dists


def extract_numeric_column_values(X: Any, column: str) -> np.ndarray:
    """Extract one manifest column as float64, dropping nulls but not infinities."""
    if isinstance(X, pl.DataFrame):
        values = X.select(pl.col(column).cast(pl.Float64, strict=False)).to_series().to_numpy()
    else:
        series = X[column]
        if hasattr(series, "to_numpy"):
            try:
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
            except TypeError:
                values = series.to_numpy()
        else:
            values = np.asarray(series)
        values = np.asarray(values, dtype=np.float64)

    return values[~np.isnan(values)]


def hydrate_discrete_manifest_metadata(spec: SSMSpec, X: pl.DataFrame) -> SSMSpec:
    """Infer per-channel discrete level counts from encoded wide data."""
    from causal_ssm_agent.models.likelihoods.observation_families import (
        any_family_needs_level_metadata,
        get_family_spec,
    )

    manifest_cols, manifest_dists = resolve_manifest_metadata(spec, X)
    needs_levels = any_family_needs_level_metadata(manifest_dists)
    if not needs_levels:
        return spec

    inferred_counts = [0] * spec.n_manifest
    for idx, (column, dist) in enumerate(zip(manifest_cols, manifest_dists, strict=False)):
        family_spec = get_family_spec(dist)
        if family_spec is None or not family_spec.needs_level_metadata:
            continue

        values = (
            X.select(column)
            .drop_nulls()
            .to_series()
            .cast(pl.Float64, strict=False)
            .drop_nulls()
            .to_numpy()
        )
        if values.size == 0:
            raise ValueError(
                f"Indicator '{column}' uses discrete emission '{dist.value}' but has no data"
            )

        try:
            inferred_count = family_spec.hydrate_levels(values)
        except ValueError as exc:
            raise ValueError(
                f"Indicator '{column}' uses discrete emission '{dist.value}' but {exc}"
            ) from exc
        if inferred_count is not None:
            inferred_counts[idx] = inferred_count

    if spec.manifest_level_counts is None:
        return replace(spec, manifest_level_counts=inferred_counts)

    if len(spec.manifest_level_counts) != spec.n_manifest:
        raise ValueError(
            "SSMSpec manifest_level_counts length does not match n_manifest: "
            f"{len(spec.manifest_level_counts)} vs {spec.n_manifest}"
        )

    resolved_counts = list(spec.manifest_level_counts)
    for idx, inferred_count in enumerate(inferred_counts):
        if inferred_count == 0:
            resolved_counts[idx] = 0
            continue
        existing_count = resolved_counts[idx]
        if existing_count in (0, inferred_count):
            resolved_counts[idx] = inferred_count
            continue
        raise ValueError(
            "Discrete level metadata mismatch for "
            f"'{manifest_cols[idx]}': spec={existing_count}, data={inferred_count}"
        )

    return replace(spec, manifest_level_counts=resolved_counts)


def validate_observation_support(spec: SSMSpec, X: Any) -> None:
    """Reject likelihoods whose support is incompatible with observed data."""
    from causal_ssm_agent.models.likelihoods.observation_families import get_family_spec

    manifest_cols, manifest_dists = resolve_manifest_metadata(spec, X)

    issues: list[str] = []
    for column, dist in zip(manifest_cols, manifest_dists, strict=False):
        values = extract_numeric_column_values(X, column)
        if values.size == 0:
            continue
        if np.any(~np.isfinite(values)):
            issues.append(
                f"- '{column}' uses {dist.value} emission but observed data contain non-finite values"
            )
            continue

        family_spec = get_family_spec(dist)
        if family_spec is None:
            continue
        invalid = family_spec.validate_support(values)
        if not np.any(invalid):
            continue

        bad_values = values[invalid]
        issues.append(
            f"- '{column}' uses {dist.value} emission but {bad_values.size}/{values.size} "
            f"observations are outside support ({family_spec.support_description}; "
            f"min={float(values.min()):.3g}, max={float(values.max()):.3g})"
        )

    if issues:
        raise ValueError("Observation support check failed:\n" + "\n".join(issues))
