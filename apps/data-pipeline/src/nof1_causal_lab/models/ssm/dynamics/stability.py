"""Stability check for vector-field dynamics.

For dense-linear systems the existing parameter_layout enforces stability
via diagonal dominance: every diagonal of ``A`` exceeds the sum of its row's
off-diagonal magnitudes plus a margin, so all eigenvalues of ``A`` have
strictly negative real parts.

For vector-field systems the analogous check is local: at a
representative state (typically the prior predictive mean), the *Jacobian*
``∂f/∂x`` should have all eigenvalues with strictly negative real parts.
This guarantees local stability around that point — sufficient for the
EKF-style auxiliary samplers to converge (Corenflos §2.3 uses this same
linearisation).

Global stability for trajectory-dependent systems requires Lyapunov analysis
or checking the Jacobian over an invariant set; the local check here is a
necessary condition and matches the dense-matrix stability condition.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.shapes import Array, Complex, Float

from .intervention import Intervention
from .vector_field import VectorField, VectorFieldArgs


@dataclass(frozen=True)
class StabilityReport:
    """Result of the Jacobian-eigenvalue stability check at one point."""

    is_stable: bool
    """True if all eigenvalues of the Jacobian have real parts ≤ ``threshold``."""

    max_real_part: float
    """Maximum real part of the eigenvalues. Negative ⇒ stable."""

    eigenvalues: Complex[Array, " D"]
    """Complex eigenvalues ``(n_latent,)``."""

    linearization_point: Float[Array, " D"]
    """Where the Jacobian was evaluated."""


def check_jacobian_stability(
    vector_field: VectorField,
    vf_params: tuple[dict[str, Array], ...],
    x_lin: Float[Array, " D"],
    *,
    intervention: Intervention | None = None,
    threshold: float = 0.0,
) -> StabilityReport:
    """Check local stability of ``vector_field`` at ``x_lin``.

    The Jacobian ``A = ∂f/∂x |_{x_lin}`` is computed via the same
    ``vector_field.linearize`` used by the discretisation machinery. The
    system is locally stable if all eigenvalues of ``A`` have real part
    strictly below ``threshold``. The default ``threshold = 0`` flags
    any non-decaying mode (real-part ≥ 0).

    Args:
        vector_field: The vector-field drift.
        vf_params: Parameter tuple matching ``vector_field.components``.
        x_lin: ``(n_latent,)`` linearisation point. Typical choices:
            the prior predictive mean state, the steady-state under
            no intervention, or the initial state.
        intervention: Optional intervention; defaults to none. Pass an
            intervention to check stability *under* that intervention
            (the Jacobian uses the intervened drift).
        threshold: Margin below which the real part must lie for the
            system to be classified stable. Use a small negative number
            (e.g., ``-1e-3``) to require a finite margin.

    Returns:
        ``StabilityReport`` with ``is_stable``, the worst real-part
        eigenvalue, all eigenvalues, and the linearisation point.
    """
    if intervention is None:
        intervention = Intervention.none()
    args = VectorFieldArgs(params=vf_params, intervention=intervention)

    jacobian, _intercept = vector_field.linearize(x_lin, args)
    eigenvalues = jnp.linalg.eigvals(jacobian)
    max_real_part = float(jnp.max(jnp.real(eigenvalues)))
    is_stable = max_real_part < threshold

    return StabilityReport(
        is_stable=is_stable,
        max_real_part=max_real_part,
        eigenvalues=eigenvalues,
        linearization_point=x_lin,
    )
