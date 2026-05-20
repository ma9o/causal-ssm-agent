"""MAP estimation for composite SSMs.

Fast point-estimate fitting via L-BFGS on the marginal log-posterior
(Gaussian observations) or the joint-at-fixed-trajectory log-posterior
(non-Gaussian fallback). Useful for model exploration, smoke checks, and
warm-starting MCMC — same role as ``methods/map.py`` plays for the
linear path.

Reuses the existing ``pathfinder_init_z_unc`` optimisation machinery:
multi-start L-BFGS finds the mode in unconstrained space; we constrain
the result back into the canonical per-component parameter tuple and
wrap in the standard :class:`InferenceResult` envelope with
``method="map"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.methods.composite_aux_kalman import (
    _composite_log_post_unc,
    _composite_marginal_log_post_unc,
    _flatten_params_to_sites,
    build_composite_aux_kalman_bundle,
    build_unconstrained_transform,
    pathfinder_init_z_unc,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult

if TYPE_CHECKING:
    from jax import Array

    from nof1_causal_lab.models.ssm import SSMModel
    from nof1_causal_lab.models.ssm.inference.targets.kernels import ObservationKernel


def fit_composite_map(
    model: SSMModel,
    observations: Array,
    runtime_times: Array,
    *,
    obs_kernel: ObservationKernel,
    rng_seed: int = 0,
    n_starts: int = 4,
    maxiter: int = 50,
    elbo_samples: int = 5,
    obs_extra_params: dict | None = None,
) -> InferenceResult:
    """L-BFGS MAP estimate for a composite SSM.

    Args:
        model: SSMModel carrying the declarative block-based spec.
        observations: ``(T, n_m)`` observation matrix.
        runtime_times: ``(T,)`` observation times.
        obs_kernel: Pre-bound observation kernel.
        rng_seed: PRNG seed for multi-start initialisation.
        n_starts: Number of L-BFGS starts (best ELBO wins).
        maxiter: L-BFGS iteration budget per start.
        elbo_samples: ELBO MC samples for start ranking.
        obs_extra_params: Optional fixed observation hyperparams.

    Returns:
        :class:`InferenceResult` with ``method="map"`` (matching the
        linear MAP shape). The ``_samples`` dict contains a single
        draw per site (the MAP estimate). ``diagnostics`` carries
        ``params_map`` (the per-component tuple), ``log_post_at_map``,
        and ``canonical_model``.
    """
    from nof1_causal_lab.models.ssm.dynamics import runtime_from_ssm_model

    canonical = runtime_from_ssm_model(
        model, obs_kernel=obs_kernel, obs_extra_params=obs_extra_params
    )
    bundle = build_composite_aux_kalman_bundle(
        model,
        observations,
        runtime_times,
        obs_kernel=obs_kernel,
        obs_extra_params=obs_extra_params,
    )
    transform = build_unconstrained_transform(bundle.compiled)

    n_latent = canonical.init_mean.shape[0]
    T = observations.shape[0]
    x_lin = jnp.broadcast_to(canonical.init_mean, (T, n_latent))

    z_map, pf_diag = pathfinder_init_z_unc(
        bundle,
        transform,
        x_lin,
        n_starts=n_starts,
        maxiter=maxiter,
        elbo_samples=elbo_samples,
        rng_seed=rng_seed,
    )
    params_map = transform.constrain_to_tuple(z_map)

    log_post_fn = (
        _composite_marginal_log_post_unc
        if canonical.obs_kernel.is_gaussian
        else _composite_log_post_unc
    )
    log_post_at_map = float(log_post_fn(z_map, x_lin, bundle, transform))

    # Laplace marginal likelihood: log Z ≈ log p(θ*, y) + (d/2) log(2π)
    # + Σ log diag(chol). The pathfinder's L-BFGS-derived chol is the
    # Cholesky of the inverse-Hessian estimate at the mode; for the
    # marginal-Kalman objective this is a real Laplace approximation
    # of the parameter marginal likelihood. For the joint-at-fixed-x
    # fallback, the Laplace value is approximate w.r.t. the joint
    # rather than the marginal — interpret accordingly.
    laplace_log_evidence: float | None = None
    chol = pf_diag.get("chol")
    if chol is not None:
        import math

        import numpy as np

        chol_np = np.asarray(chol)
        d = int(chol_np.shape[0])
        log_det_inv_hess = float(np.sum(np.log(np.abs(np.diag(chol_np))))) * 2.0
        laplace_log_evidence = (
            log_post_at_map
            + 0.5 * d * math.log(2.0 * math.pi)
            + 0.5 * log_det_inv_hess
        )

    # Pack as a single-iteration "samples" dict so downstream code that
    # expects (n_iter, *shape) shape works without special-casing MAP.
    flat_sites = _flatten_params_to_sites(params_map, prefix="vf")
    samples_flat: dict[str, Array] = {
        name: jnp.expand_dims(value, axis=0) for name, value in flat_sites.items()
    }

    diagnostics: dict[str, Any] = {
        "params_map": params_map,
        "log_post_at_map": log_post_at_map,
        "laplace_log_evidence": laplace_log_evidence,
        "z_unc_map": z_map,
        "pathfinder_diagnostics": pf_diag,
        "vector_field": bundle.compiled.vector_field,
        "canonical_model": canonical,
        "objective": "marginal" if canonical.obs_kernel.is_gaussian else "joint_at_fixed_trajectory",
    }
    return InferenceResult(
        _samples=samples_flat,
        method="map",
        diagnostics=diagnostics,
    )
