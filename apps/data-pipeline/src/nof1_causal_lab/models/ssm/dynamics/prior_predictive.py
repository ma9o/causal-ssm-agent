"""Prior-predictive sampling for composite vector fields.

Closes the integration gap between the linear-path
``prior_predictive_runtime.py`` (which validates ``SSMPriors`` dict-config
through the Stage 4 prior-predictive pipeline) and the composite path
introduced for non-linear dynamics. The two cannot share an
implementation cleanly — they consume different prior representations
and the linear path is tightly coupled to the dense-linear discretizer
— but the *surface* matches: a single function that takes a compiled
spec, draws ``n`` parameter sets, simulates trajectories under each,
and reports stability + finite-output diagnostics.

This is the validation hook Stage 4 (or the agentic repair flow) calls
when it has a composite spec instead of an ``SSMPriors`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.random as jr
from numpyro.handlers import seed

from .config import compile_composite_from_dict
from .intervention import Intervention
from .simulator import simulate
from .stability import check_jacobian_stability

if TYPE_CHECKING:
    from jax import Array

    from .canonical import RuntimeSSM
    from .compilation import CompiledComposite


@dataclass(frozen=True)
class CompositePriorPredictive:
    """One draw of trajectories + per-draw stability verdicts.

    ``trajectories`` has shape ``(n_draws, T, n_latent)``. ``param_draws``
    is the per-draw parameter tuple suitable for replaying inference
    deterministically. ``stable`` is a boolean per-draw mask; ``False``
    means the Jacobian at the linearisation point had at least one
    non-negative-real eigenvalue. ``observations`` is populated by
    :func:`sample_composite_prior_predictive_full` and is ``None`` for
    the latents-only :func:`sample_composite_prior_predictive`.
    """

    trajectories: Array
    param_draws: list[tuple[dict[str, Array], ...]]
    stable: Array
    max_real_eigenvalue: Array
    finite: Array
    observations: Array | None = None


def sample_composite_prior_predictive(
    compiled: CompiledComposite,
    init_mean: Array,
    times: Array,
    *,
    n_draws: int = 100,
    rng_seed: int = 0,
    x_lin: Array | None = None,
    stability_threshold: float = 0.0,
) -> CompositePriorPredictive:
    """Sample ``n_draws`` prior-predictive trajectories for a composite spec.

    For each draw:
    - Draw a fresh parameter tuple from ``compiled.sample_params`` under a
      seeded NumPyro context.
    - Linearise the vector field at ``x_lin`` (default: ``init_mean``)
      and run :func:`check_jacobian_stability` to flag dynamics with
      non-decaying modes.
    - Simulate the deterministic ODE trajectory via :func:`simulate`.

    The verdicts feed Stage 4 / repair: any draw with ``stable=False`` or
    ``finite=False`` is a candidate failure the repair flow should
    address.

    Args:
        compiled: Output of ``compile_composite``.
        init_mean: ``(n_latent,)`` starting state.
        times: ``(T,)`` time grid for trajectory output.
        n_draws: Number of prior draws to take.
        rng_seed: Seed for the per-draw NumPyro contexts.
        x_lin: Linearisation point for the stability check. Defaults to
            ``init_mean`` (the trajectory's starting point).
        stability_threshold: Real-part threshold for ``check_jacobian_stability``.
    """
    if x_lin is None:
        x_lin = init_mean

    trajectories: list[Array] = []
    param_draws: list[tuple[dict[str, Array], ...]] = []
    stable_flags: list[bool] = []
    max_real_parts: list[float] = []
    finite_flags: list[bool] = []

    base_key = jr.PRNGKey(rng_seed)
    for draw_idx in range(n_draws):
        draw_key = jr.fold_in(base_key, draw_idx)
        with seed(rng_seed=int(draw_key[0])):
            params = compiled.sample_params()

        report = check_jacobian_stability(
            compiled.vector_field,
            params,
            x_lin=x_lin,
            threshold=stability_threshold,
        )

        traj = simulate(
            compiled.vector_field,
            params,
            Intervention.none(),
            init_mean,
            times,
        )

        param_draws.append(params)
        trajectories.append(traj)
        stable_flags.append(bool(report.is_stable))
        max_real_parts.append(float(report.max_real_part))
        finite_flags.append(bool(jnp.all(jnp.isfinite(traj))))

    return CompositePriorPredictive(
        trajectories=jnp.stack(trajectories, axis=0),
        param_draws=param_draws,
        stable=jnp.asarray(stable_flags),
        max_real_eigenvalue=jnp.asarray(max_real_parts),
        finite=jnp.asarray(finite_flags),
    )


def validate_composite_dynamics(
    compiled: CompiledComposite,
    init_mean: Array,
    times: Array,
    *,
    n_draws: int = 100,
    rng_seed: int = 0,
    stable_fraction_threshold: float = 0.5,
) -> dict[str, object]:
    """Summary verdict for a composite spec, in the shape Stage 4
    validation consumes (``code``, ``is_valid``, ``failing_draws``,
    ``primary_score``).

    Used by the composite-aware Stage 4 repair branch, mirroring how
    ``prior_predictive.py`` produces ``dynamics_stability`` results for
    the linear path. Composite path piggy-backs on the same code/
    repair-scope vocabulary so downstream consumers don't fork.
    """
    pp = sample_composite_prior_predictive(
        compiled, init_mean, times, n_draws=n_draws, rng_seed=rng_seed
    )
    n_unstable = int(jnp.sum(~pp.stable))
    n_nonfinite = int(jnp.sum(~pp.finite))
    n_total = int(pp.stable.shape[0])
    is_valid = (
        n_unstable <= n_total * stable_fraction_threshold
        and n_nonfinite <= n_total * stable_fraction_threshold
    )
    return {
        "parameter": "dynamics_stability",
        "code": "dynamics_stability",
        "is_valid": bool(is_valid),
        "n_draws": n_total,
        "n_unstable": n_unstable,
        "n_nonfinite": n_nonfinite,
        "failing_draw_indices": [
            int(i)
            for i in jnp.where(~pp.stable | ~pp.finite)[0].tolist()
        ],
        "primary_score": float((n_unstable + n_nonfinite) / max(1, n_total)),
        "max_real_eigenvalue_per_draw": pp.max_real_eigenvalue,
    }


def sample_composite_prior_predictive_full(
    canonical: RuntimeSSM,
    times: Array,
    *,
    n_draws: int = 100,
    rng_seed: int = 0,
    x_lin: Array | None = None,
    stability_threshold: float = 0.0,
) -> CompositePriorPredictive:
    """Full composite prior-predictive: params + latents + observations.

    Canonical-keyed wrapper that composes
    :func:`sample_composite_prior_predictive` and
    :func:`sample_observations_from_latents` in one call, mirroring the
    return shape of the linear-path
    ``prior_predictive_runtime.sample_prior_predictive_from_priors`` —
    with the addition that ``observations`` populates the returned
    :class:`CompositePriorPredictive`.

    Use this when validating a composite spec end-to-end (latent
    stability **and** observation plausibility). The latents-only
    :func:`sample_composite_prior_predictive` is still appropriate when
    the observation operator is unknown or not the focus of validation.
    """
    from .compilation import CompiledComposite

    compiled = CompiledComposite(
        vector_field=canonical.vector_field,
        sample_params=canonical.sample_params,
    )
    pp = sample_composite_prior_predictive(
        compiled,
        canonical.init_mean,
        times,
        n_draws=n_draws,
        rng_seed=rng_seed,
        x_lin=x_lin,
        stability_threshold=stability_threshold,
    )
    obs_key = jr.fold_in(jr.PRNGKey(rng_seed), n_draws + 1)
    observations = sample_observations_from_latents(
        canonical, pp.trajectories, obs_key
    )
    return CompositePriorPredictive(
        trajectories=pp.trajectories,
        param_draws=pp.param_draws,
        stable=pp.stable,
        max_real_eigenvalue=pp.max_real_eigenvalue,
        finite=pp.finite,
        observations=observations,
    )


def composite_posterior_predictive_check(
    canonical: RuntimeSSM,
    fit_result: Any,
    observations: Array,
    *,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Posterior-predictive check (PPC) diagnostics for a composite fit.

    Samples posterior predictive observations on top of the MCMC
    trajectory samples and compares them to the actual observations,
    returning the standard PPC summary statistics.

    Returns a dict with:

    - ``pp_mean``: posterior mean of predicted observations,
      shape ``(T, n_m)``.
    - ``pp_std``: posterior std of predicted observations,
      shape ``(T, n_m)``.
    - ``residuals``: ``y_actual − pp_mean``, shape ``(T, n_m)``.
    - ``z_scores``: ``residuals / pp_std``, shape ``(T, n_m)``.
    - ``coverage_95``: per-channel fraction of actual observations
      that fall within the 2.5/97.5 percentile of the posterior
      predictive distribution, shape ``(n_m,)``. Well-calibrated
      models should give values near 0.95.
    - ``rmse``: per-channel root-mean-square error between actual
      observations and ``pp_mean``, shape ``(n_m,)``.

    Useful for model-fit assessment — large residuals or low coverage
    indicate model misspecification.
    """
    pp_observations = sample_composite_posterior_predictive_observations(
        canonical, fit_result, rng_seed=rng_seed
    )
    pp_mean = jnp.mean(pp_observations, axis=0)
    pp_std = jnp.std(pp_observations, axis=0)
    residuals = observations - pp_mean
    safe_std = jnp.where(pp_std > 1e-12, pp_std, 1.0)
    z_scores = residuals / safe_std

    lower = jnp.quantile(pp_observations, 0.025, axis=0)
    upper = jnp.quantile(pp_observations, 0.975, axis=0)
    in_band = (observations >= lower) & (observations <= upper)
    coverage_95 = jnp.mean(in_band.astype(observations.dtype), axis=0)
    rmse = jnp.sqrt(jnp.mean(residuals * residuals, axis=0))

    return {
        "pp_mean": pp_mean,
        "pp_std": pp_std,
        "residuals": residuals,
        "z_scores": z_scores,
        "coverage_95": coverage_95,
        "rmse": rmse,
    }


def composite_per_t_log_likelihood(
    canonical: RuntimeSSM,
    fit_result: Any,
    observations: Array,
    *,
    chain_grouped: bool = False,
) -> Array:
    """Per-timestep log-likelihood across the composite posterior.

    For each posterior draw ``i`` (over both parameters and the latent
    trajectory), compute ``log p(y_t | x_t^{(i)}, θ^{(i)})`` for every
    timestep ``t``.

    Shape:
    - ``chain_grouped=False`` (default): ``(n_chains·n_iter, T)`` —
      flat across chains, matching the leading axis of
      ``trajectory_samples``.
    - ``chain_grouped=True``: ``(n_chains, n_iter, T)`` — the shape
      ArviZ's ``az.from_dict(log_likelihood=...)`` consumes directly
      for PSIS-LOO.

    Uses the kernel-driven trajectory log-prob path so all observation
    families (Gaussian / Beta / Binomial / Poisson / ...) work
    uniformly — no per-family special-case here.
    """
    from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
        trajectory_observation_log_probs,
    )

    trajectory_samples = fit_result.diagnostics.get("trajectory_samples")
    if trajectory_samples is None:
        raise ValueError(
            "composite_per_t_log_likelihood requires fit_result.diagnostics"
            "['trajectory_samples'] (populated by fit_composite_aux_kalman)."
        )

    obs_kernel = canonical.obs_kernel
    H = canonical.H
    d_meas = canonical.d_meas
    R = canonical.R

    def _per_draw(x_traj: Array) -> Array:
        return trajectory_observation_log_probs(
            x_traj, observations, None, H, d_meas, R, obs_kernel
        )

    ll_flat = jax.vmap(_per_draw)(trajectory_samples)
    if not chain_grouped:
        return ll_flat
    n_chains = int(fit_result.diagnostics.get("num_chains", 1))
    n_iter = int(
        fit_result.diagnostics.get("num_samples_per_chain", ll_flat.shape[0])
    )
    return ll_flat.reshape(n_chains, n_iter, -1)


def sample_composite_posterior_predictive_observations(
    canonical: RuntimeSSM,
    fit_result: Any,
    *,
    rng_seed: int = 0,
) -> Array:
    """Posterior-predictive observation samples for a composite fit.

    Composite MCMC's ``diagnostics["trajectory_samples"]`` already
    represents draws from the smoothing posterior over latent
    trajectories ``p(x_{1:T} | y, θ)``. Emitting observations on top of
    these via the canonical's observation kernel gives posterior-
    predictive draws ``p(y* | y)`` — useful for residual analysis, PPC
    plots, and Stage 5 model-fit diagnostics. Closes the parity gap
    with the linear path's ``FittedArtifact.ppc_result``.

    Args:
        canonical: Canonical model envelope used during fitting (must
            carry the observation operator and kernel).
        fit_result: ``InferenceResult`` from
            ``fit_composite_aux_kalman``. Required to have
            ``diagnostics["trajectory_samples"]``.
        rng_seed: PRNG seed for observation sampling.

    Returns:
        ``(n_draws, T, n_manifest)`` array of posterior-predictive
        observations. Same family handling as
        :func:`sample_observations_from_latents` — Gaussian via
        Cholesky noise, non-Gaussian via the canonical's predictive
        sampler when populated.
    """
    trajectory_samples = fit_result.diagnostics.get("trajectory_samples")
    if trajectory_samples is None:
        raise ValueError(
            "sample_composite_posterior_predictive_observations expects "
            "fit_result.diagnostics['trajectory_samples'] (populated by "
            "fit_composite_aux_kalman)."
        )
    return sample_observations_from_latents(
        canonical, trajectory_samples, jr.PRNGKey(rng_seed)
    )


def sample_observations_from_latents(
    canonical: RuntimeSSM,
    latents: Array,
    rng_key: Array,
) -> Array:
    """Emit observations ``y = obs_model(H · x + d)`` from latent trajectories.

    Closes the parity gap with the linear-path ``prior_predictive_runtime``:
    composite prior-predictive sampling can now produce observations,
    not just latents.

    Two paths:

    1. **``canonical.predictive_sampler`` populated** — the caller passed
       ``manifest_dists`` / ``manifest_links`` / ``obs_extra_params``
       through ``runtime_from_composite``. The existing
       ``build_predictive_observation_sampler`` factory handles
       Gaussian / Student-t / Beta / Binomial / Poisson /
       NegativeBinomial / Ordered-logistic / Categorical families
       uniformly; this function dispatches to its
       ``sample_point_trajectory``.
    2. **Sampler not populated, Gaussian kernel** — fall back to a
       Cholesky-noise Gaussian sample. This is the convenience path so
       callers who only have an ``ObservationKernel`` (no explicit dists)
       can still emit observations when the family happens to be Gaussian.

    Non-Gaussian without ``predictive_sampler`` raises
    ``NotImplementedError`` with a clear message naming the missing
    inputs.

    Args:
        canonical: Carries the observation operator (``H``, ``d_meas``,
            ``R``) and either an obs kernel + Gaussian assumption, or a
            full predictive sampler.
        latents: ``(..., T, n_latent)`` latent trajectories. A leading
            batch axis (e.g., ``n_draws``) is broadcasted automatically.
        rng_key: JAX PRNG key for sampling.
    """
    linear_pred = jnp.einsum("ij,...tj->...ti", canonical.H, latents) + canonical.d_meas

    if canonical.predictive_sampler is not None:
        sampler = canonical.predictive_sampler.sample_point_trajectory
        if linear_pred.ndim == 2:
            return sampler(rng_key, linear_pred)
        # Batched leading axes — vmap over the first.
        keys = jr.split(rng_key, linear_pred.shape[0])
        flat_lin = linear_pred.reshape(linear_pred.shape[0], *linear_pred.shape[-2:])
        return jax.vmap(sampler)(keys, flat_lin)

    if not canonical.obs_kernel.is_gaussian:
        raise NotImplementedError(
            "sample_observations_from_latents needs either a Gaussian obs kernel "
            "or a canonical with predictive_sampler populated. Pass manifest_dists, "
            "manifest_links and obs_extra_params to runtime_from_composite to "
            "build the predictive sampler for non-Gaussian families."
        )

    chol_R = jnp.linalg.cholesky(canonical.R)
    noise = jr.normal(rng_key, linear_pred.shape, dtype=linear_pred.dtype)
    return linear_pred + noise @ chol_R.T


@dataclass
class CompositeAssemblyValidation:
    """Composite analogue of Stage-4 ``AssemblyValidation``.

    Mirrors the linear-path shape (``compile_ok`` / ``pp_valid`` /
    ``diagnostics``) so a Stage 4 caller can layer composite validation
    on top of the existing assembly check by ANDing the two ``is_valid``
    flags and concatenating diagnostic lists.
    """

    compile_ok: bool = True
    compile_error: str | None = None
    pp_checked: bool = False
    pp_valid: bool = True
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    compiled: CompiledComposite | None = None

    @property
    def is_valid(self) -> bool:
        return self.compile_ok and self.pp_valid


def validate_composite_assembly(
    composite_spec_config: dict[str, Any],
    init_mean: Array,
    times: Array,
    *,
    n_draws: int = 100,
    rng_seed: int = 0,
    stable_fraction_threshold: float = 0.5,
) -> CompositeAssemblyValidation:
    """One-shot Stage-4-style validation for a composite spec config.

    Steps:

    1. Compile the spec via ``compile_composite_from_dict``. Compile
       errors are surfaced as ``compile_ok=False`` with the message in
       ``compile_error`` (matching the linear path's failure shape).
    2. Run ``validate_composite_dynamics`` against the compiled spec.
    3. Pack the diagnostic dict into ``diagnostics``.

    Callers (a future Stage 4 LLM tool, the agentic repair flow, the
    notebook validator) can compose this with the linear assembly
    validation by ANDing ``is_valid`` flags and concatenating
    diagnostics.
    """
    try:
        compiled = compile_composite_from_dict(composite_spec_config)
    except (ValueError, KeyError) as exc:
        return CompositeAssemblyValidation(
            compile_ok=False, compile_error=str(exc)
        )

    diagnostic = validate_composite_dynamics(
        compiled,
        init_mean,
        times,
        n_draws=n_draws,
        rng_seed=rng_seed,
        stable_fraction_threshold=stable_fraction_threshold,
    )
    return CompositeAssemblyValidation(
        compile_ok=True,
        pp_checked=True,
        pp_valid=bool(diagnostic["is_valid"]),
        diagnostics=[diagnostic],
        compiled=compiled,
    )
