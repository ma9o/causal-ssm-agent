"""Shared observation metadata helpers for SSM model preparation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

NON_MANIFEST_COLUMNS = {"time", "time_bucket"}


def default_manifest_columns(X: Any) -> list[str]:
    """Infer manifest columns from a wide dataframe-like object."""
    return [c for c in X.columns if c not in NON_MANIFEST_COLUMNS and not str(c).endswith("_lag1")]


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
    from causal_ssm_agent.models.likelihoods.observation_families import FAMILY_REGISTRY

    manifest_cols, manifest_dists = resolve_manifest_metadata(spec, X)
    needs_levels = any(
        (family_spec := FAMILY_REGISTRY.get(dist)) is not None
        and family_spec.requires_integer_encoding
        for dist in manifest_dists
    )
    if not needs_levels:
        return spec

    inferred_counts = [0] * spec.n_manifest
    for idx, (column, dist) in enumerate(zip(manifest_cols, manifest_dists, strict=False)):
        family_spec = FAMILY_REGISTRY.get(dist)
        if family_spec is None or not family_spec.requires_integer_encoding:
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
    from causal_ssm_agent.models.likelihoods.observation_families import FAMILY_REGISTRY

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

        family_spec = FAMILY_REGISTRY.get(dist)
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
