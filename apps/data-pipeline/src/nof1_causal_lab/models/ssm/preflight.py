"""Fit-time consistency checks between observed data, spec, and priors.

The semantic pipeline layers (model-spec auto-standardization, prior-predictive
scale gates) only protect models that pass through them. Callers that drive
``inference.fit()`` directly — benchmarks, notebooks, manual runs — can hand
the sampler a configuration whose ground truth has essentially zero prior
density (e.g. a raw-scale Gaussian indicator mean of 87 under the canonical
``Normal(0, 2)`` manifest-mean prior). The posterior then concentrates on the
best compromise reachable within the priors and every coupled coordinate
distorts to absorb the misfit, which is indistinguishable from a sampler
failure in recovery summaries.

These checks run unconditionally at the ``fit()`` boundary and fail loudly:

- A channel marked ``manifest_standardized`` must actually arrive standardized
  (column mean near 0 and column sd near 1).
- A free manifest mean on an unstandardized identity-link location channel must
  have the observed column mean within reach of its prior.

Scope is deliberately limited to checks that cannot false-positive on a
legitimately authored model: fixed manifest means, non-identity links, and
non-location families are not judged here — scale plausibility for those
configurations belongs to the prior-predictive checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.priors import (
    PriorDistributionFamily,
    PriorSpec,
    default_prior_for_descriptor,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMModel

LOCATION_REACH_SIGMAS = 6.0
STANDARDIZED_MEAN_SD_RATIO = 0.5
STANDARDIZED_SD_BAND = (0.5, 2.0)

_LOCATION_FAMILIES = (DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T)
_REACH_PRIOR_FAMILIES = (
    PriorDistributionFamily.NORMAL,
    PriorDistributionFamily.TRUNCATED_NORMAL,
)


class ObservationPreflightError(ValueError):
    """Observed data is inconsistent with the spec/prior configuration."""


def _resolve_site_prior(model: SSMModel, site_name: str) -> PriorSpec | None:
    if model.priors is not None:
        prior = model.priors.get(site_name)
        if prior is not None:
            return prior
    for site in model.spec.iter_sample_sites():
        if site.name == site_name:
            return default_prior_for_descriptor(site)
    return None


def _prior_loc_scale(prior: PriorSpec, n_free: int, free_idx: int) -> tuple[float, float]:
    mu = np.broadcast_to(np.asarray(prior.params.get("mu", 0.0), dtype=np.float64), (n_free,))
    sigma = np.broadcast_to(np.asarray(prior.params.get("sigma", 1.0), dtype=np.float64), (n_free,))
    return float(mu[free_idx]), float(sigma[free_idx])


def validate_observations_for_fit(model: SSMModel, observations: Any) -> None:
    """Validate (spec, priors, observations) consistency before fitting.

    Raises:
        ObservationPreflightError: listing every violating channel.
    """
    spec = model.spec
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != spec.n_manifest:
        raise ObservationPreflightError(
            f"observations must have shape (N, {spec.n_manifest}), got {obs.shape}"
        )

    dists = list(spec.manifest_dists or [])
    if not dists:
        return

    links = (
        list(spec.manifest_links)
        if spec.manifest_links is not None
        else [LinkFunction.IDENTITY] * spec.n_manifest
    )
    standardized = list(spec.manifest_standardized or [False] * spec.n_manifest)
    names = (
        list(spec.manifest_names)
        if spec.manifest_names is not None
        else [f"manifest[{idx}]" for idx in range(spec.n_manifest)]
    )

    means_block = spec.manifest_means_block
    free_support = np.asarray(means_block.free_support, dtype=bool)
    n_free = int(free_support.sum())
    free_prior = _resolve_site_prior(model, means_block.free_site_name) if n_free else None

    problems: list[str] = []
    for j in range(spec.n_manifest):
        finite = obs[:, j][np.isfinite(obs[:, j])]
        if finite.size == 0:
            continue
        mean_j = float(finite.mean())
        sd_j = float(finite.std())

        if bool(standardized[j]):
            if abs(mean_j) > STANDARDIZED_MEAN_SD_RATIO * sd_j + 1e-4:
                problems.append(
                    f"{names[j]}: marked standardized but the observed column mean is "
                    f"{mean_j:.4g} (sd {sd_j:.4g}); apply standardization to the data before "
                    "fit (production applies it in prepare_model_runtime)"
                )
            sd_lo, sd_hi = STANDARDIZED_SD_BAND
            if finite.size >= 2 and sd_j > 0.0 and not (sd_lo <= sd_j <= sd_hi):
                problems.append(
                    f"{names[j]}: marked standardized but the observed column sd is "
                    f"{sd_j:.4g} (expected ~1); apply standardization to the data before "
                    "fit (production applies it in prepare_model_runtime)"
                )
            continue

        if DistributionFamily(dists[j]) not in _LOCATION_FAMILIES:
            continue
        if links[j] is not None and LinkFunction(links[j]) != LinkFunction.IDENTITY:
            continue
        if not bool(free_support[j]):
            continue
        if free_prior is None or free_prior.family not in _REACH_PRIOR_FAMILIES:
            continue

        free_idx = int(free_support[:j].sum())
        mu_j, sigma_j = _prior_loc_scale(free_prior, n_free, free_idx)
        if sigma_j <= 0.0:
            continue
        z = abs(mean_j - mu_j) / sigma_j
        if z > LOCATION_REACH_SIGMAS:
            problems.append(
                f"{names[j]}: observed mean {mean_j:.4g} lies {z:.1f} prior sd from its free "
                f"manifest-mean prior {free_prior.family.name}(mu={mu_j:.4g}, sigma={sigma_j:.4g}); "
                "the posterior cannot reach the data location — mark the indicator standardized "
                "(and standardize the data) or author the prior on the data scale"
            )

    if problems:
        raise ObservationPreflightError(
            "Observation/prior preflight failed:\n- " + "\n- ".join(problems)
        )
