"""Cheap pre-fit parametric counting diagnostics."""

from __future__ import annotations

import numpy as np

from causal_ssm_agent.models.ssm.parameterization import build_site_registry

from .results import TRuleResult


def count_free_params(spec) -> dict[str, int]:
    """Count free parameters using the canonical site registry as authority."""
    counts: dict[str, int] = {}
    for site in build_site_registry(spec):
        counts[site.name] = int(np.prod(site.shape)) if site.shape else 1
    return counts


def check_t_rule(spec, T: int | None = None) -> TRuleResult:
    """Check the t-rule (necessary counting condition) for identification."""
    param_counts = count_free_params(spec)
    n_free = sum(param_counts.values())
    n_manifest = spec.n_manifest

    n_mean = n_manifest
    n_cov = n_manifest * (n_manifest + 1) // 2
    n_autocov = (T - 1) * n_manifest if T is not None and T > 1 else 0
    n_moments = n_mean + n_cov + n_autocov

    return TRuleResult(
        n_free_params=n_free,
        n_manifest=n_manifest,
        n_timepoints=T,
        n_moments=n_moments,
        satisfies=n_free <= n_moments,
        param_counts=param_counts,
    )
