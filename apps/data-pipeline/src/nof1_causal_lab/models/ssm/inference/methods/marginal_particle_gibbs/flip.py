"""Joint (latent coordinate, loading column) sign-flip MH move.

Factor measurement structures carry a sign ambiguity: the likelihood sees only the
product of a loading column and its latent coordinate path, so (lambda_d, x_{:,d})
and (-lambda_d, -x_{:,d}) fit the data identically. The default lambda prior is
deliberately sign-asymmetric (soft anchoring), which identifies the posterior — but
a Gibbs-style sampler whose conditionals are each one-sided can still start in, or
wander into, the prior-disfavored mirror basin and never cross back: given a
negative loading the trajectory conditional props up the negative path, and vice
versa. This move proposes the simultaneous flip and accepts by the exact joint
posterior ratio (which prices the prior asymmetry, drift coupling, and everything
else), giving chains a direct route between the mirror basins. Composed after the
smoother sweep, it is a standard involutive MH kernel on the same target: the flip
is self-inverse and the coordinate choice is state-independent, so the acceptance
is just the posterior ratio.

The move is only meaningful when negating the unconstrained ``lambda_free`` entries
negates the constrained loadings, i.e. the site bijection is the identity (real
support). Positive-support loading priors pin the sign outright — there is no
mirror basin to escape — so requesting flips there is a configuration error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from numpyro.distributions.transforms import IdentityTransform


@dataclass(frozen=True)
class SignFlipSpec:
    """Flat-position masks for the flippable loading columns."""

    coords: jnp.ndarray  # (n_flippable,) latent coordinate index per move
    masks: jnp.ndarray  # (n_flippable, dim) bool: lambda_free entries of that column


def build_sign_flip_spec(model: Any, bundle: dict[str, Any]) -> SignFlipSpec:
    """Map each latent coordinate's free loading entries to flat-position indices."""
    site_info = bundle["site_info"]
    if "lambda_free" not in site_info:
        raise ValueError(
            "latent_sign_flip_moves requires free factor loadings (a 'lambda_free' "
            "site); this model has none, so there is no sign ambiguity to move across."
        )
    transform = site_info["lambda_free"]["transform"]
    if not isinstance(transform, IdentityTransform):
        raise ValueError(
            "latent_sign_flip_moves requires an unconstrained (identity-transform) "
            "lambda_free site: negation in unconstrained space must negate the "
            f"loadings. Got transform {type(transform).__name__}; a positive-support "
            "loading prior pins the sign, so flips are meaningless there."
        )

    positions = model.parameter_layout.lambda_free_positions  # [(manifest_row, latent_col)]
    n_latent = int(model.spec.n_latent)
    unravel_fn = bundle["unravel_fn"]
    flat_example = bundle["flat_example"]
    base = dict(unravel_fn(jnp.zeros_like(flat_example)))

    coords: list[int] = []
    masks: list[jnp.ndarray] = []
    for coord in range(n_latent):
        entry_indices = [entry for entry, (_row, col) in enumerate(positions) if col == coord]
        if not entry_indices:
            continue
        marked = dict(base)
        marked["lambda_free"] = (
            jnp.zeros_like(base["lambda_free"])
            .at[jnp.asarray(entry_indices, dtype=jnp.int32)]
            .set(1.0)
        )
        flat_marker, _ = ravel_pytree(marked)
        coords.append(coord)
        masks.append(flat_marker > 0.5)
    if not coords:
        raise ValueError("latent_sign_flip_moves found no latent coordinate with free loadings.")
    return SignFlipSpec(
        coords=jnp.asarray(coords, dtype=jnp.int32),
        masks=jnp.stack(masks, axis=0),
    )
