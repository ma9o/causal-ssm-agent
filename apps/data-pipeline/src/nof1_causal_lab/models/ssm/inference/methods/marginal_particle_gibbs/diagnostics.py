"""Optional diagnostic metric groups for marginalized Particle Gibbs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class MPGibbsDiagnosticMetric(StrEnum):
    """Optional MPGibbs diagnostic metric groups that allocate per-sample traces."""

    PARTICLE_FILTER = "particle_filter"
    BACKWARD_SELECTION = "backward_selection"
    PARTICLE_IDENTITY = "particle_identity"
    PARAMETER_MOVEMENT = "parameter_movement"


MPGIBBS_DIAGNOSTIC_METRIC_VALUES = tuple(metric.value for metric in MPGibbsDiagnosticMetric)


@dataclass(frozen=True)
class MPGibbsDiagnosticFlags:
    """Resolved static switches for optional MPGibbs diagnostic traces."""

    particle_filter: bool
    backward_selection: bool
    particle_identity: bool
    parameter_movement: bool


def resolve_mpgibbs_diagnostic_metrics(
    *,
    diagnostic_metrics_all: bool,
    diagnostic_metrics: Sequence[str] | None,
) -> frozenset[str]:
    """Resolve optional MPGibbs diagnostic metric groups from config/CLI strings."""
    if diagnostic_metrics_all:
        return frozenset(MPGIBBS_DIAGNOSTIC_METRIC_VALUES)
    requested = tuple(diagnostic_metrics or ())
    allowed = set(MPGIBBS_DIAGNOSTIC_METRIC_VALUES)
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(
            "marginal_particle_gibbs diagnostic_metrics contains unknown values "
            f"{unknown}; allowed: {sorted(allowed)}."
        )
    return frozenset(requested)


def build_mpgibbs_diagnostic_flags(
    *,
    latent_smoother: str,
    diagnostic_metrics: frozenset[str],
) -> MPGibbsDiagnosticFlags:
    """Build static booleans for optional diagnostic groups."""
    # Only the sequential blocked-backward csmc ("plain") smoother runs a forward filter
    # and a backward-sampling pass to instrument; dsmc builds the smoothing path by a
    # divide-and-conquer tree.
    is_sequential_particle_smoother = latent_smoother == "plain"
    return MPGibbsDiagnosticFlags(
        particle_filter=(
            is_sequential_particle_smoother
            and MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in diagnostic_metrics
        ),
        backward_selection=(
            is_sequential_particle_smoother
            and MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in diagnostic_metrics
        ),
        particle_identity=(MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in diagnostic_metrics),
        parameter_movement=(MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in diagnostic_metrics),
    )
