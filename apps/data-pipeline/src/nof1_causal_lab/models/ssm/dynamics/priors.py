"""Default NumPyro prior distributions for non-linear component parameters.

These factories produce NumPyro ``Distribution`` objects directly
plug-in-able into ``ComponentSpec``s. The defaults are PK/PD-informed
but deliberately weakly informative: the LLM is expected to override
them when domain knowledge is available (e.g., setting ``EC50`` to a
scale tied to the observed dose range).

The factories are *callables*, not module-level constants, because
NumPyro distributions cache JAX arrays internally and constructing them
at import time can interact badly with JAX initialisation order.

Structure blocks and dynamics components both bind priors through the
site-keyed registry in ``ssm.priors``; this module only provides
distribution factories for hand-built composite specs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpyro.distributions as ndist

if TYPE_CHECKING:
    from jax import Array


# ---------------------------------------------------------------------------
# Hill saturation primitive
# ---------------------------------------------------------------------------


def hill_emax_prior(loc: float = 0.0, scale: float = 1.0) -> ndist.Distribution:
    """LogNormal prior over Hill ``Emax`` (maximum effect magnitude).

    Defaults to ``LogNormal(0, 1)`` — median 1.0, 95% interval roughly
    ``[0.14, 7.4]``. Override ``loc`` to match the expected effect scale
    on the outcome latent (e.g., ``loc=ln(0.5)`` if you expect a ~0.5
    standardised effect at saturation).
    """
    return ndist.LogNormal(loc=loc, scale=scale)


def hill_ec50_prior(loc: float = 0.0, scale: float = 1.0) -> ndist.Distribution:
    """LogNormal prior over Hill ``EC50`` (half-maximal concentration).

    Defaults to ``LogNormal(0, 1)``. For pharmacological use the user
    should override ``loc`` to match the typical concentration scale —
    e.g., ``loc=ln(EC50_typical)`` where ``EC50_typical`` is the
    half-maximal dose in the patient's exposure units.
    """
    return ndist.LogNormal(loc=loc, scale=scale)


def hill_n_prior(
    low: float = 1.0,
    high: float = 4.0,
    loc: float = 2.0,
    scale: float = 0.5,
) -> ndist.Distribution:
    """Truncated Normal prior over Hill coefficient ``n``.

    Hill coefficients are biologically constrained to be ``≥ 1`` (the
    classical Hill–Langmuir saturation form) and almost never exceed
    ``4`` in physiological systems. Default mode ``n = 2`` captures the
    common slightly-sigmoidal regime; tighten ``scale`` to enforce
    near-Michaelis–Menten (``n = 1``) behaviour.
    """
    return ndist.TruncatedNormal(low=low, high=high, loc=loc, scale=scale)


# ---------------------------------------------------------------------------
# Multiplicative coupling primitive
# ---------------------------------------------------------------------------


def multiplicative_weight_prior(
    loc: float = 0.0, scale: float = 1.0
) -> ndist.Distribution:
    """Normal prior over multiplicative coupling weight ``w``.

    For ``w · η_a · η_b`` the right scale depends on the observed scale
    of ``η_a`` and ``η_b``. Default ``Normal(0, 1)`` is weakly
    informative; tighten ``scale`` when the latents are standardised.
    """
    return ndist.Normal(loc=loc, scale=scale)


# ---------------------------------------------------------------------------
# Effect-compartment / LinearEdge weight
# ---------------------------------------------------------------------------


def linear_edge_weight_prior(
    loc: float = 0.0, scale: float = 0.5
) -> ndist.Distribution:
    """Normal prior over a generic LinearEdge weight ``β``.

    Tighter default than the multiplicative prior because per-edge
    linear coupling is the most common "background" causal term and we
    want regularisation toward zero.
    """
    return ndist.Normal(loc=loc, scale=scale)


def effect_compartment_rate_prior(
    loc: float = 0.0, scale: float = 0.7
) -> ndist.Distribution:
    """LogNormal prior over the effect-compartment rate ``k_e0``.

    For SSRI-like delayed-onset dynamics the effective time constant is
    ``1 / k_e0``. ``LogNormal(0, 0.7)`` puts ~95% of mass on ``k_e0 ∈
    [0.25, 4.0]``, i.e., time constants of 4–0.25 days. Override the
    median to encode known pharmacokinetics (e.g., ``loc = ln(1/14)``
    for a 2-week onset).
    """
    return ndist.LogNormal(loc=loc, scale=scale)


# ---------------------------------------------------------------------------
# DiagonalDecay / per-latent decay rates
# ---------------------------------------------------------------------------


def diagonal_decay_prior(
    rate_concentration: Array | float = 2.0,
    rate_rate: Array | float = 4.0,
) -> ndist.Distribution:
    """Gamma prior over per-latent decay rates ``ρ`` (must be positive).

    Defaults match the structural affine drift defaults
    (``Gamma(concentration=2, rate=4)`` → mean 0.5, mode 0.25), so the
    composite primitive case retains the existing stability prior
    behaviour for the diagonal-decay component.
    """
    return ndist.Gamma(concentration=rate_concentration, rate=rate_rate)
