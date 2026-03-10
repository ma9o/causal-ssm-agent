"""Automatic reparameterization strategies for NumPyro models.

Ported from Pyro's pyro.infer.reparam.strategies module.

Reference: Gorinova, Moore & Hoffman (2019) "Automatic Reparameterisation
of Probabilistic Programs" https://arxiv.org/abs/1906.03028

Provides Strategy base class and AutoReparam, which automatically selects
reparameterizations for each sample site based on distribution type:

- TransformReparam for TransformedDistribution
- LocScaleReparam (learnable centering) for loc-scale families with real support
- ProjectedNormalReparam for ProjectedNormal (fallback)

Usage::

    from causal_ssm_agent.models.ssm.autoreparam import AutoReparam

    # As a decorator (learnable centering for SVI):
    reparam_model = AutoReparam()(model_fn)

    # As a config for handlers.reparam:
    with numpyro.handlers.reparam(config=AutoReparam(centered=0.0)):
        ...  # fully decentered for MCMC

    # Inspect cached config after first model execution:
    strategy = AutoReparam()
    reparam_model = strategy(model_fn)
    # ... run model once ...
    print(strategy.config)  # {site_name: Reparam or None, ...}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer.reparam import (
    LocScaleReparam,
    ProjectedNormalReparam,
    Reparam,
    TransformReparam,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class Strategy(ABC):
    """Abstract base class for reparameterizer configuration strategies.

    Derived classes must implement the :meth:`configure` method.

    Can be used as:
    1. A config callable for ``numpyro.handlers.reparam(model, config=strategy)``
    2. A decorator: ``strategy(model)``

    The config dict is populated on first model execution. Subsequent
    executions reuse cached reparameterizers.
    """

    def __init__(self):
        self.config: dict[str, Reparam | None] = {}

    @abstractmethod
    def configure(self, msg: dict) -> Reparam | None:
        """Input a sample site and return a Reparam or None.

        Called only on first model execution per site; subsequent
        executions use the cached result from self.config.

        Args:
            msg: A sample site message dict with keys like 'name', 'fn',
                'value', 'is_observed', etc.

        Returns:
            A Reparam instance or None.
        """
        raise NotImplementedError

    def __call__(self, msg_or_fn: dict | Callable):
        """Use as config callable or model decorator.

        When called with a dict (by handlers.reparam internally),
        configures and caches the reparameterizer for that site.

        When called with a callable (as decorator), wraps the model
        with handlers.reparam using this strategy.
        """
        if isinstance(msg_or_fn, dict):
            msg = msg_or_fn
            name = msg["name"]
            if name in self.config:
                return self.config[name]
            result = self.configure(msg)
            self.config[name] = result
            return result
        else:
            fn = msg_or_fn
            return numpyro.handlers.reparam(fn, config=self)


class MinimalReparam(Strategy):
    """Minimal strategy that only reparameterizes sites that would
    otherwise lead to errors (e.g., ProjectedNormal).

    Example::

        reparam_model = MinimalReparam()(model_fn)
    """

    def configure(self, msg: dict) -> Reparam | None:
        return _minimal_reparam(msg["fn"], msg.get("is_observed", False))


class AutoReparam(Strategy):
    """Automatic reparameterization strategy.

    Applies a cascade of reparameterizers to each sample site:

    1. TransformReparam for TransformedDistribution
    2. LocScaleReparam for loc-scale families with unconstrained (real) support
    3. ProjectedNormalReparam for ProjectedNormal (fallback)

    For loc-scale families (Normal, Laplace, StudentT, etc.), the centering
    parameter controls the interpolation between centered (original) and
    decentered (non-centered) parameterizations:

    - ``centered=None`` (default): Learn per-site per-element centering via
      ``numpyro.param``. Requires an optimization loop (SVI).
    - ``centered=0.0``: Fully decentered (standard NCP). Best for MCMC.
    - ``centered=1.0``: Fully centered (no-op). Original parameterization.
    - ``centered=0.5``: Halfway between centered and decentered.

    .. warning:: This strategy may change behavior across releases.
        To inspect or save a given behavior, extract the ``.config`` dict
        after running the model at least once.

    Args:
        centered: Optional centering parameter for LocScaleReparam.
            If None (default), centering will be learned. If a float in
            [0.0, 1.0], uses fixed centering.
    """

    def __init__(self, *, centered: float | None = None):
        super().__init__()
        if centered is not None and not (0.0 <= centered <= 1.0):
            raise ValueError(f"centered must be in [0, 1], got {centered}")
        self.centered = centered

    def configure(self, msg: dict) -> Reparam | None:
        fn = msg["fn"]
        if not msg.get("is_observed", False):
            # Unwrap through known wrapper types only (Independent,
            # ExpandedDistribution, MaskedDistribution).  Substantive
            # distributions that happen to have base_dist (e.g.
            # TruncatedDistribution) are NOT unwrapped — they have
            # their own support/constraints that must be respected.
            inner = fn
            while True:
                if isinstance(inner, dist.TransformedDistribution):
                    return TransformReparam()
                if isinstance(
                    inner,
                    (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution),
                ):
                    inner = inner.base_dist
                    continue
                break

            # Apply LocScaleReparam for loc-scale families with real support.
            result = _loc_scale_reparam(msg["name"], inner, self.centered)
            if result is not None:
                return result

        # Fallback to minimal reparameterization.
        return _minimal_reparam(fn, msg.get("is_observed", False))


def _loc_scale_reparam(
    name: str, fn: dist.Distribution, centered: float | None
) -> LocScaleReparam | None:
    """Return LocScaleReparam if fn is a loc-scale family with real support."""
    if "_decentered" in name:
        return None  # Avoid infinite recursion from LocScaleReparam aux sites.

    # Check for location-scale family (must have both loc and scale).
    params = set(fn.arg_constraints)
    if not {"loc", "scale"}.issubset(params):
        return None

    # Check for unconstrained (real) support.
    if not _is_unconstrained(fn.support):
        return None

    # Extra shape params (everything except loc and scale).
    shape_params = sorted(params - {"loc", "scale"})
    return LocScaleReparam(centered=centered, shape_params=shape_params)


def _minimal_reparam(fn: dist.Distribution, is_observed: bool) -> Reparam | None:
    """Apply minimal reparameterization for distributions that need it."""
    if is_observed:
        return None

    # Unwrap through known wrapper types only.
    inner = fn
    while True:
        if isinstance(inner, dist.TransformedDistribution):
            if _minimal_reparam(inner.base_dist, is_observed) is None:
                return None
            else:
                return TransformReparam()
        if isinstance(
            inner, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)
        ):
            inner = inner.base_dist
            continue
        break

    if isinstance(inner, dist.ProjectedNormal):
        return ProjectedNormalReparam()

    return None


def _is_unconstrained(constraint) -> bool:
    """Check if a constraint is unconstrained (real-valued)."""
    while hasattr(constraint, "base_constraint"):
        constraint = constraint.base_constraint
    return constraint is constraints.real
