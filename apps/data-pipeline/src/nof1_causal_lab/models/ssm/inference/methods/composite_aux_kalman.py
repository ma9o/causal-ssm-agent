"""Composite-spec auxiliary Kalman MCMC driver.

End-to-end Gibbs sampler for the non-linear vector field framework:

- Parameters ``θ`` are sampled via **either** Random Walk Metropolis on
  the constrained space (cheap, no autodiff) **or** blackjax NUTS on
  the unconstrained reparameterisation (better mixing on correlated
  parameter spaces; requires gradient flow through the composite
  context builder).
- Latent trajectory ``x_{0:T}`` is updated via the two-context
  ``composite_latent_mh_step_eq10_11`` from
  ``inference/trajectory_mcmc/composite_kalman.py``.
- The two steps alternate; ``θ`` is fixed during the trajectory update,
  ``x`` is fixed during the parameter update.

Production features built across the Phase A–D-3 work:

- ``num_warmup`` + ``num_chains`` with independent per-chain RNGs.
- ``adapt_step_size`` — Robbins-Monro adaptation of trajectory MH
  ``latent_delta`` and parameter step (RWM binary accept or NUTS
  ``acceptance_rate``) during the warmup window; step sizes freeze for
  sampling to preserve detailed balance.
- NUTS-only diagonal mass-matrix adaptation from warmup ``z_unc``
  sample variance.
- ``init_method="pathfinder"`` — runs ``scipy_pathfinder`` with the
  marginal log-likelihood objective (vanilla Kalman filter on the
  linearised observation model) when the obs kernel is Gaussian, and
  the joint-at-fixed-trajectory objective otherwise.

Returns an :class:`InferenceResult` with composite-shaped param
samples flattened to ``{site_name: (n_chains·n_iter, *param_shape)}``
and chain-grouped samples on ``diagnostics["chain_samples"]`` for r̂ /
ESS via :func:`InferenceResult.get_mcmc_diagnostics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import blackjax.mcmc.nuts as bjx_nuts
import jax
import jax.numpy as jnp
import jax.random as random
from numpyro.distributions.transforms import biject_to
from numpyro.handlers import seed, trace
from numpyro.infer.util import log_density

from nof1_causal_lab.models.ssm.inference.methods.scipy_pathfinder import (
    scipy_pathfinder,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    build_gaussian_trajectory_prior_terms,
    trajectory_prior_log_prob_from_terms,
)
from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
    trajectory_observation_log_prob,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman import (
    CompositeLatentMHState,
    composite_latent_context_at_trajectory,
    composite_latent_mh_step_eq10_11,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult
from nof1_causal_lab.models.ssm.inference.utils import (
    build_unconstrained_site_transform,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from nof1_causal_lab.models.ssm import SSMModel
    from nof1_causal_lab.models.ssm.dynamics import CompiledComposite, RuntimeSSM
    from nof1_causal_lab.models.ssm.inference.targets.kernels import ObservationKernel
    from nof1_causal_lab.models.ssm_observation_metadata import ObservationSupportRuntime


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeAuxKalmanBundle:
    """Closure-free bundle exposing the callables the composite MCMC needs.

    Created by ``build_composite_aux_kalman_bundle``. Methods on this
    object are not pytrees — the dataclass is the *configuration* the
    driver reads at construction time.
    """

    compiled: CompiledComposite
    observations: Array
    runtime_times: Array
    init_mean: Array
    init_cov: Array
    diffusion_cov: Array
    H: Array
    d_meas: Array
    R: Array
    obs_kernel: ObservationKernel
    log_prior_fn: Callable[[tuple[dict[str, Array], ...]], Array]
    observation_log_prob_and_grad_fn: Callable[..., tuple[Array, Array]]
    mean_log_prob_fn: Callable | None = None
    observation_support: ObservationSupportRuntime | None = None

    def context_builder(self, vf_params: tuple[dict[str, Array], ...]):
        """Return ``x_traj → LatentContext`` for a given parameter draw."""

        def _builder(x_traj):
            return composite_latent_context_at_trajectory(
                vector_field=self.compiled.vector_field,
                vf_params=vf_params,
                x_traj=x_traj,
                init_mean=self.init_mean,
                init_cov=self.init_cov,
                diffusion_cov=self.diffusion_cov,
                runtime_times=self.runtime_times,
                H=self.H,
                d_meas=self.d_meas,
                R=self.R,
            )

        return _builder


def _make_kernel_obs_log_prob_and_grad(
    obs_kernel: ObservationKernel,
    *,
    mean_log_prob_fn: Callable | None = None,
    observation_support: ObservationSupportRuntime | None = None,
) -> Callable[..., tuple[Array, Array]]:
    """Build an observation log-prob/grad callback driven by an ``ObservationKernel``.

    Replaces the hardcoded Gaussian path with the full dispatch used by the
    dense auxiliary Kalman driver. Handles per-channel heterogeneous families
    (Gaussian, Student-t, Beta, Binomial, Poisson, …) and interval-summary
    semantics when ``observation_support`` is provided.
    """

    def _fn(context, x_traj: Array, observations: Array) -> tuple[Array, Array]:
        def _logp(x: Array) -> Array:
            return trajectory_observation_log_prob(
                x,
                observations,
                None,
                context.H,
                context.d_meas,
                context.R,
                obs_kernel,
                mean_log_prob_fn,
                observation_support,
            )

        return jax.value_and_grad(_logp)(x_traj)

    return _fn


def _flatten_params_to_sites(
    params_tuple: tuple[dict[str, Array], ...], prefix: str = "vf"
) -> dict[str, Array]:
    """``(component_index, param_name) → NumPyro site name`` bridge."""
    sites: dict[str, Array] = {}
    for i, slice_params in enumerate(params_tuple):
        for key, value in slice_params.items():
            sites[f"{prefix}_{i}_{key}"] = value
    return sites


def _stack_param_samples(
    param_samples: list[tuple[dict[str, Array], ...]], prefix: str = "vf"
) -> dict[str, Array]:
    """Convert a per-iteration list of component-keyed param tuples into the
    flat ``dict[site_name, (n_iter, *param_shape)]`` shape the
    :class:`InferenceResult` envelope and all the existing diagnostics
    machinery (ArviZ summaries, posterior marginals, …) consume.
    """
    if not param_samples:
        return {}
    flat_first = _flatten_params_to_sites(param_samples[0], prefix=prefix)
    return {
        name: jnp.stack([_flatten_params_to_sites(p, prefix=prefix)[name] for p in param_samples])
        for name in flat_first
    }


def build_composite_aux_kalman_bundle(
    model: SSMModel,
    observations: Array,
    runtime_times: Array,
    *,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: Callable | None = None,
    observation_support: ObservationSupportRuntime | None = None,
    obs_extra_params: dict | None = None,
) -> CompositeAuxKalmanBundle:
    """Construct the closure-heavy bundle for composite MCMC.

    Takes a declarative :class:`SSMModel` carrying a block-based
    ``SSMSpec`` plus a runtime ``obs_kernel``; the SSM hyperparams
    (init moments, diffusion, measurement operator) are pulled from the
    owning block templates via :func:`runtime_from_ssm_model`.

    Args:
        model: SSMModel carrying the declarative spec.
        observations: ``(T, n_m)``.
        runtime_times: ``(T,)`` observation times.
        obs_kernel: Observation kernel evaluated at the fixed measurement
            hyperparams. For Gaussian channels this is constructed via
            ``build_observation_kernel(GAUSSIAN, IDENTITY, manifest_cov=R)``.
        mean_log_prob_fn: Optional per-mean log-prob for interval-summary
            channels (mean/sum aggregators); required when
            ``observation_support.requires_interval_summary_handling``.
        observation_support: Optional support runtime carrying the
            interval-summary plan.
        obs_extra_params: Optional dict of fixed observation hyperparams
            (df, shape, …) for non-Gaussian families; threaded into the
            predictive sampler built on the runtime envelope.
    """
    from nof1_causal_lab.models.ssm.dynamics import runtime_from_ssm_model

    runtime = runtime_from_ssm_model(
        model, obs_kernel=obs_kernel, obs_extra_params=obs_extra_params
    )

    sample_params = runtime.sample_params

    def _numpyro_model() -> tuple[dict[str, Array], ...]:
        return sample_params()

    site_prefix = runtime.site_prefix

    def _log_prior(params_tuple: tuple[dict[str, Array], ...]) -> Array:
        sites = _flatten_params_to_sites(params_tuple, prefix=site_prefix)
        log_dens, _ = log_density(_numpyro_model, (), {}, sites)
        return log_dens

    obs_callback = _make_kernel_obs_log_prob_and_grad(
        runtime.obs_kernel,
        mean_log_prob_fn=mean_log_prob_fn,
        observation_support=observation_support,
    )

    from nof1_causal_lab.models.ssm.dynamics import CompiledComposite as _CompiledComposite

    compiled = _CompiledComposite(
        vector_field=runtime.vector_field, sample_params=sample_params
    )

    return CompositeAuxKalmanBundle(
        compiled=compiled,
        observations=observations,
        runtime_times=runtime_times,
        init_mean=runtime.init_mean,
        init_cov=runtime.init_cov,
        diffusion_cov=runtime.diffusion_cov,
        H=runtime.H,
        d_meas=runtime.d_meas,
        R=runtime.R,
        obs_kernel=runtime.obs_kernel,
        log_prior_fn=_log_prior,
        observation_log_prob_and_grad_fn=obs_callback,
        mean_log_prob_fn=mean_log_prob_fn,
        observation_support=observation_support,
    )


# ---------------------------------------------------------------------------
# Parameter RWM step
# ---------------------------------------------------------------------------


def _param_rwm_step(
    params: tuple[dict[str, Array], ...],
    x_traj: Array,
    key: Array,
    bundle: CompositeAuxKalmanBundle,
    step_size: float,
) -> tuple[tuple[dict[str, Array], ...], dict[str, Any]]:
    """One random-walk MH step on the parameters with ``x_traj`` fixed.

    Proposes each leaf perturbation in the *constrained* space:
    infeasible values (e.g., negative ``EC50``) cause the log-prior to
    return ``-inf`` and the proposal is automatically rejected.

    Acceptance ratio combines:
    - parameter prior ratio (in constrained space; the proposal is
      symmetric so this is just ``log_prior(θ*) − log_prior(θ)``)
    - latent-trajectory log-prob ratio under the new vs. old linearisation
      of ``x_traj``
    - observation log-prob ratio under the new vs. old linearisation
      (the observation model itself doesn't depend on ``θ`` for the
      Gaussian case here, but the per-step transition density does).
    """
    propose_key, accept_key = random.split(key, 2)

    # Generate per-leaf Gaussian perturbations
    flat_params, treedef = jax.tree.flatten(params)
    perturb_keys = random.split(propose_key, len(flat_params))
    proposed_flat = [
        leaf + step_size * random.normal(k, leaf.shape, dtype=leaf.dtype)
        for k, leaf in zip(perturb_keys, flat_params, strict=True)
    ]
    proposed_params = jax.tree.unflatten(treedef, proposed_flat)

    # Compute log posterior at current and proposed params (conditioning on x_traj)
    log_prior_curr = bundle.log_prior_fn(params)
    log_prior_prop = bundle.log_prior_fn(proposed_params)

    def _joint_log_lik(p):
        ctx = bundle.context_builder(p)(x_traj)
        from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
            build_gaussian_trajectory_prior_terms,
            trajectory_prior_log_prob_from_terms,
        )

        prior_terms = build_gaussian_trajectory_prior_terms(
            ctx.Ad, ctx.Qd, ctx.cd, ctx.init_mean, ctx.init_cov, jitter=1e-6
        )
        prior_x = trajectory_prior_log_prob_from_terms(
            x_traj, ctx.Ad, ctx.cd, prior_terms
        )
        obs_lp, _ = bundle.observation_log_prob_and_grad_fn(
            ctx, x_traj, bundle.observations
        )
        return prior_x + obs_lp

    lik_curr = _joint_log_lik(params)
    lik_prop = _joint_log_lik(proposed_params)

    log_alpha = (log_prior_prop + lik_prop) - (log_prior_curr + lik_curr)
    accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
    # Handle -inf log_alpha → accept_prob = 0
    accept_prob = jnp.where(jnp.isnan(accept_prob) | jnp.isinf(log_alpha), 0.0, accept_prob)
    accept = random.bernoulli(accept_key, accept_prob)

    new_params = jax.tree.map(
        lambda old, new: jnp.where(accept, new, old), params, proposed_params
    )
    return new_params, {"accepted": accept.astype(jnp.float32), "log_alpha": log_alpha}


# ---------------------------------------------------------------------------
# Unconstrained transform + NUTS step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _UnconstrainedTransform:
    """Composite-spec wrapper around :class:`UnconstrainedSiteTransform`.

    The shared transform operates on flat dicts keyed by NumPyro site
    name; this wrapper layers the composite-component tuple reshape on
    top so blackjax NUTS sees a single flat vector while the vector
    field consumes the canonical ``(params_dict, ...)`` tuple.
    """

    flat_init: Array
    dim: int
    constrain_to_tuple: Callable[[Array], tuple[dict[str, Array], ...]]
    log_abs_det_jacobian: Callable[[Array], Array]


def build_unconstrained_transform(
    compiled: CompiledComposite,
    *,
    init_seed: int = 0,
    site_prefix: str = "vf",
) -> _UnconstrainedTransform:
    """Build the unconstrained ↔ constrained transform for a compiled spec.

    Delegates to ``build_unconstrained_site_transform`` for the per-site
    bijection + Jacobian logic; this wrapper only adds the
    component-tuple reshape required by the composite vector field.
    """

    def _model() -> tuple[dict[str, Array], ...]:
        return compiled.sample_params()

    tr = trace(seed(_model, rng_seed=init_seed)).get_trace()
    site_info: dict[str, dict[str, Any]] = {
        name: {
            "transform": biject_to(info["fn"].support),
            "value": info["value"],
            "distribution": info["fn"],
        }
        for name, info in tr.items()
        if info["type"] == "sample"
    }
    shared = build_unconstrained_site_transform(site_info)

    with seed(rng_seed=init_seed):
        example_tuple = compiled.sample_params()
    component_layout: tuple[tuple[tuple[str, str], ...], ...] = tuple(
        tuple((key, f"{site_prefix}_{i}_{key}") for key in slice_params)
        for i, slice_params in enumerate(example_tuple)
    )

    def constrain_to_tuple(z_flat: Array) -> tuple[dict[str, Array], ...]:
        constrained = shared.constrain_dict(z_flat)
        return tuple(
            {key: constrained[site_name] for key, site_name in component_keys}
            for component_keys in component_layout
        )

    return _UnconstrainedTransform(
        flat_init=shared.flat_init,
        dim=shared.dim,
        constrain_to_tuple=constrain_to_tuple,
        log_abs_det_jacobian=shared.log_abs_det_jacobian,
    )


def build_composite_fitted_artifact(
    canonical: RuntimeSSM,
    result: InferenceResult,
    *,
    runtime_times: Array,
    latent_names: list[str],
    manifest_names: list[str] | None = None,
    observation_support: Any = None,
    ppc_result: dict[str, Any] | None = None,
) -> Any:
    """Package a composite fit into a :class:`FittedArtifact`.

    Closes the integration gap where composite fits couldn't go through
    Stage 6 without manually constructing a builder shim. The synthetic
    builder carries just the ``spec.latent_names`` / ``manifest_names``
    that ``_prepare_stage6_simulation`` reads — Stage 6 has its own
    composite dispatch path that doesn't need the heavyweight linear
    ``SSMBuilder`` / ``model`` attributes.

    Args:
        canonical: Canonical model envelope used for fitting.
        result: ``InferenceResult`` from ``fit_composite_aux_kalman``.
        runtime_times: ``(T,)`` observation times.
        latent_names: Names for each latent dimension (length ``n_latent``).
        manifest_names: Optional names for each observation channel.
        observation_support: Optional support runtime (pass-through).
        ppc_result: Optional pre-computed posterior-predictive results.

    Returns:
        :class:`FittedArtifact` ready for Stage 6 / artifact persistence.
        The synthetic builder carries enough state for
        ``_prepare_stage6_simulation`` and the composite Stage 6 dispatch
        to find the canonical model + per-draw param tuples in
        ``result.diagnostics``.
    """
    from dataclasses import dataclass

    from nof1_causal_lab.models.ssm.inference.types import FittedArtifact

    @dataclass(frozen=True)
    class _CompositeSpecShim:
        latent_names: list[str]
        manifest_names: list[str] | None = None

    @dataclass(frozen=True)
    class _CompositeBuilderShim:
        spec: _CompositeSpecShim
        model: Any = None
        canonical: RuntimeSSM | None = None

    spec_shim = _CompositeSpecShim(
        latent_names=list(latent_names),
        manifest_names=list(manifest_names) if manifest_names else None,
    )
    builder_shim = _CompositeBuilderShim(spec=spec_shim, canonical=canonical)

    return FittedArtifact(
        result=result,
        builder=builder_shim,
        times=runtime_times,
        observation_support=observation_support,
        ppc_result=ppc_result,
    )


def _vanilla_kalman_log_evidence(
    Ad: Array,
    Qd: Array,
    cd: Array,
    init_mean: Array,
    init_cov: Array,
    observations: Array,
    H: Array,
    d_meas: Array,
    R: Array,
    *,
    jitter: float = 1e-6,
) -> Array:
    """Forward Kalman filter on the linearised LGSSM; returns log p(y_{1:T}).

    Standard innovation-form recursion under a Gaussian observation
    model ``y_t = H · x_t + d + ε``, ``ε ~ N(0, R)``. Marginalises out
    the latent trajectory analytically. Used by the marginal-likelihood
    pathfinder objective.

    Delegates to the existing ``filter_lgssm`` (cuthbert square-root
    filter via ``parallel_kalman.py``) — the same numerical primitive
    the linear inference stack uses. Closes one of the two hand-rolled
    Kalman filter duplications I introduced before this audit.
    """
    from nof1_causal_lab.models.ssm.inference.parallel_kalman import filter_lgssm

    T = observations.shape[0]
    Hs = jnp.broadcast_to(H, (T, *H.shape))
    Rs = jnp.broadcast_to(R, (T, *R.shape))
    cs = jnp.broadcast_to(d_meas, (T, d_meas.shape[0]))
    state = filter_lgssm(
        init_mean=init_mean,
        init_cov=init_cov,
        Fs=Ad,
        Qs=Qd,
        bs=cd,
        Hs=Hs,
        Rs=Rs,
        cs=cs,
        ys=observations,
        jitter=jitter,
        parallel=False,
    )
    return jnp.sum(state.loglik)


def _composite_marginal_log_post_unc(
    z_flat: Array,
    x_lin: Array,
    bundle: CompositeAuxKalmanBundle,
    transform: _UnconstrainedTransform,
) -> Array:
    """Marginal log-posterior ``log p(z, y | linearisation at x_lin)``.

    Integrates out the latent trajectory analytically via a vanilla
    Kalman filter on the linearised system. Used by the marginal-
    likelihood pathfinder objective when the observation kernel is
    Gaussian. Non-Gaussian observations fall back to the joint-at-
    fixed-x posterior (:func:`_composite_log_post_unc`).
    """
    params_tuple = transform.constrain_to_tuple(z_flat)
    log_prior_constrained = bundle.log_prior_fn(params_tuple)
    log_jac = transform.log_abs_det_jacobian(z_flat)
    ctx = bundle.context_builder(params_tuple)(x_lin)
    marginal_log_y = _vanilla_kalman_log_evidence(
        ctx.Ad, ctx.Qd, ctx.cd, ctx.init_mean, ctx.init_cov,
        bundle.observations, ctx.H, ctx.d_meas, ctx.R,
    )
    return log_prior_constrained + log_jac + marginal_log_y


def pathfinder_init_z_unc(
    bundle: CompositeAuxKalmanBundle,
    transform: _UnconstrainedTransform,
    x_lin: Array,
    *,
    n_starts: int = 4,
    maxiter: int = 20,
    elbo_samples: int = 10,
    rng_seed: int = 0,
) -> tuple[Array, dict[str, Any]]:
    """Pathfinder initialisation for the composite NUTS parameter block.

    Runs ``scipy_pathfinder`` on the *joint* log-posterior at a fixed
    trajectory ``x_lin``: ``log p(z, x_lin | y) = log_prior_constrained(z)
    + log_jac(z) + log_prior(x_lin | z) + log_obs(y | x_lin, z)``.

    A proper marginal-likelihood pathfinder would integrate out the
    latent trajectory via a Kalman filter; this fixed-``x_lin`` variant
    is the lighter wedge that doesn't require building a vanilla
    Kalman filter alongside the existing auxiliary-Kalman machinery.
    The result is still a meaningfully better init than a random prior
    draw, because the optimisation accounts for both the prior and the
    likelihood at the initial trajectory.

    Returns ``(z_unc, diagnostics)`` where ``z_unc`` is the pathfinder
    mean and ``diagnostics`` is the raw ``ScipyPathfinderResult.diagnostics``
    payload.
    """
    import numpy as np

    init_key = random.PRNGKey(rng_seed)
    start_keys = random.split(init_key, n_starts)
    starts: list[np.ndarray] = []
    for k in start_keys:
        # Project a prior draw into the unconstrained space the
        # transform expects. Reuse the trace bound to the same seed so
        # the dict-keyed unflattening order is consistent.
        from jax.flatten_util import ravel_pytree
        from numpyro.distributions.transforms import biject_to

        tr = trace(seed(lambda: bundle.compiled.sample_params(), rng_seed=int(k[0]))).get_trace()
        init_unc = {}
        for name, info in tr.items():
            if info["type"] != "sample":
                continue
            bij = biject_to(info["fn"].support)
            init_unc[name] = bij.inv(info["value"])
        z_start, _ = ravel_pytree(init_unc)
        starts.append(np.asarray(z_start))

    # Marginal-likelihood pathfinder for Gaussian observations
    # (integrates the latent trajectory out via a vanilla Kalman filter).
    # Falls back to the joint log-posterior at fixed ``x_lin`` for
    # non-Gaussian families where the Kalman recursion doesn't apply.
    use_marginal = bool(bundle.obs_kernel.is_gaussian)
    log_post_fn = (
        _composite_marginal_log_post_unc if use_marginal else _composite_log_post_unc
    )

    def _log_post_and_grad(z_np):
        z = jnp.asarray(z_np)
        lp, grad = jax.value_and_grad(
            lambda zz: log_post_fn(zz, x_lin, bundle, transform)
        )(z)
        return float(lp), np.asarray(grad)

    pf_result = scipy_pathfinder(
        _log_post_and_grad,
        starts,
        maxiter=maxiter,
        elbo_samples=elbo_samples,
        seed=rng_seed,
    )
    # Expose the local Gaussian approximation (chol of inverse-Hessian)
    # and ELBO alongside the optimisation diagnostics — needed for
    # Laplace marginal-likelihood computation in fit_composite_map.
    diag = dict(pf_result.diagnostics)
    diag["chol"] = np.asarray(pf_result.chol)
    diag["best_elbo"] = float(pf_result.best_elbo)

    # NaN/Inf guard: pathfinder can return non-finite values when the
    # log-posterior gradient explodes (degenerate priors, ill-conditioned
    # systems). Falling back to the prior-draw init keeps the chain
    # alive rather than seeding NUTS with NaN. The diagnostic records
    # the fallback so callers can investigate.
    mean = np.asarray(pf_result.mean)
    if not np.all(np.isfinite(mean)):
        diag["nonfinite_fallback"] = True
        return transform.flat_init, diag
    diag["nonfinite_fallback"] = False
    return jnp.asarray(mean), diag


def _composite_log_post_unc(
    z_flat: Array,
    x_traj: Array,
    bundle: CompositeAuxKalmanBundle,
    transform: _UnconstrainedTransform,
) -> Array:
    """Unconstrained log posterior ``log p(z, x_traj | y)`` with
    ``x_traj`` held fixed. Used by NUTS to sample the parameters."""
    params_tuple = transform.constrain_to_tuple(z_flat)
    log_prior_constrained = bundle.log_prior_fn(params_tuple)
    log_jac = transform.log_abs_det_jacobian(z_flat)

    ctx = bundle.context_builder(params_tuple)(x_traj)
    prior_terms = build_gaussian_trajectory_prior_terms(
        ctx.Ad, ctx.Qd, ctx.cd, ctx.init_mean, ctx.init_cov, jitter=1e-6
    )
    prior_x = trajectory_prior_log_prob_from_terms(
        x_traj, ctx.Ad, ctx.cd, prior_terms
    )
    obs_lp, _ = bundle.observation_log_prob_and_grad_fn(
        ctx, x_traj, bundle.observations
    )
    return log_prior_constrained + log_jac + prior_x + obs_lp


def _param_nuts_step(
    z: Array,
    x_traj: Array,
    key: Array,
    bundle: CompositeAuxKalmanBundle,
    transform: _UnconstrainedTransform,
    *,
    step_size: float,
    inverse_mass_matrix: Array,
    max_num_doublings: int,
) -> tuple[Array, dict[str, Any]]:
    """One blackjax NUTS step on ``z`` (parameters in unconstrained space)
    with ``x_traj`` fixed.

    NUTS always advances (unlike RWM there is no rejection per se), but
    we still report a ``divergent`` flag and the energy / step count for
    diagnostics.
    """

    def logdens(z_flat: Array) -> Array:
        return _composite_log_post_unc(z_flat, x_traj, bundle, transform)

    hmc_state = bjx_nuts.init(z, logdens)
    new_state, info = bjx_nuts.build_kernel()(
        key,
        hmc_state,
        logdens,
        step_size,
        inverse_mass_matrix,
        max_num_doublings,
    )
    extras: dict[str, Any] = {
        "divergent": info.is_divergent.astype(jnp.float32),
        "energy": jnp.asarray(info.energy, dtype=jnp.float32),
        "num_integration_steps": jnp.asarray(
            info.num_integration_steps, dtype=jnp.float32
        ),
        "acceptance_rate": jnp.asarray(info.acceptance_rate, dtype=jnp.float32),
    }
    return new_state.position, extras


# ---------------------------------------------------------------------------
# Main fit loop
# ---------------------------------------------------------------------------


def _run_single_chain(
    bundle: CompositeAuxKalmanBundle,
    *,
    n_iterations: int,
    latent_delta: float,
    param_step_size: float,
    rng_key: Array,
    initial_x_traj: Array | None,
    initial_params: tuple[dict[str, Array], ...] | None,
    parallel: bool,
    param_kernel: Literal["rwm", "nuts"],
    nuts_max_num_doublings: int,
    nuts_inverse_mass_matrix: Array | None,
    num_warmup_adapt: int = 0,
    adapt_step_size: bool = False,
    target_param_accept: float = 0.3,
    target_traj_accept: float = 0.65,
    adapt_learning_rate: float = 0.1,
    init_method: Literal["prior", "pathfinder"] = "prior",
    pathfinder_n_starts: int = 4,
    pathfinder_maxiter: int = 20,
    pathfinder_elbo_samples: int = 10,
) -> dict[str, Any]:
    """Run ``n_iterations`` Gibbs-MCMC steps for one chain.

    Returns a dict of per-iteration sample lists. The outer driver
    (:func:`fit_composite_aux_kalman`) slices warmup and concatenates
    across chains.

    When ``adapt_step_size`` is true and ``num_warmup_adapt > 0``, the
    trajectory-MH step (``latent_delta``) and RWM parameter step
    (``param_step_size``) are adapted via Robbins-Monro on the
    log-scale during the first ``num_warmup_adapt`` iterations:
    ``log(step) += eta/√(i+1) · (observed_accept − target_accept)``.
    After warmup the adapted step sizes are held fixed for the
    sampling iterations (preserves detailed balance). NUTS step-size
    adaptation is not yet implemented; ``param_kernel="nuts"`` keeps
    the fixed ``param_step_size``.
    """
    init_key, run_key = random.split(rng_key, 2)
    if param_kernel not in {"rwm", "nuts"}:
        raise ValueError(f"Unknown param_kernel: {param_kernel!r}")

    if initial_params is None:
        with seed(rng_seed=int(init_key[0])):
            initial_params = bundle.compiled.sample_params()

    n_latent = bundle.init_mean.shape[0]
    T = bundle.observations.shape[0]
    if initial_x_traj is None:
        initial_x_traj = jnp.broadcast_to(bundle.init_mean, (T, n_latent))

    state = CompositeLatentMHState(
        position=jnp.zeros(0),
        latent_trajectory=initial_x_traj,
        latent_delta=jnp.asarray(latent_delta),
        trajectory_log_prob=jnp.asarray(0.0),
        complete_log_posterior=jnp.asarray(0.0),
    )
    params = initial_params

    transform: _UnconstrainedTransform | None = None
    z_unc: Array | None = None
    inverse_mass_matrix: Array | None = None
    if param_kernel == "nuts":
        transform = build_unconstrained_transform(bundle.compiled)
        z_unc = transform.flat_init
        if init_method == "pathfinder":
            x_lin_for_pf = initial_x_traj
            z_unc, _pf_diag = pathfinder_init_z_unc(
                bundle,
                transform,
                x_lin_for_pf,
                n_starts=pathfinder_n_starts,
                maxiter=pathfinder_maxiter,
                elbo_samples=pathfinder_elbo_samples,
                rng_seed=int(rng_key[0]),
            )
        if nuts_inverse_mass_matrix is None:
            inverse_mass_matrix = jnp.ones((transform.dim,), dtype=z_unc.dtype)
        else:
            inverse_mass_matrix = jnp.asarray(nuts_inverse_mass_matrix)
        params = transform.constrain_to_tuple(z_unc)

    traj_samples: list[Array] = []
    param_samples: list[tuple[dict[str, Array], ...]] = []
    traj_accepts: list[float] = []
    param_diagnostics: list[dict[str, float]] = []
    log_alpha_traj_history: list[float] = []
    latent_delta_history: list[float] = []
    param_step_size_history: list[float] = []
    # Warmup z_unc samples used to estimate the inverse mass matrix at
    # the end of the warmup window. Only populated for NUTS + adaptation.
    z_unc_warmup_samples: list[Array] = []
    initial_inverse_mass_matrix: Array | None = inverse_mass_matrix
    adapted_inverse_mass_matrix: Array | None = None

    # Live step sizes — mutated during warmup adaptation, frozen after.
    current_param_step_size = float(param_step_size)

    def _log_prior_unc_dummy(_z):
        return jnp.asarray(0.0)

    for i in range(n_iterations):
        step_key = random.fold_in(run_key, i)
        traj_key, param_key = random.split(step_key, 2)

        builder = bundle.context_builder(params)
        state, traj_extras = composite_latent_mh_step_eq10_11(
            state,
            traj_key,
            bundle.observations,
            context_builder=builder,
            log_prior_unc_fn=_log_prior_unc_dummy,
            observation_log_prob_and_grad_fn=bundle.observation_log_prob_and_grad_fn,
            parallel=parallel,
        )

        if param_kernel == "rwm":
            params, param_extras = _param_rwm_step(
                params, state.latent_trajectory, param_key, bundle,
                current_param_step_size,
            )
            param_diagnostics.append(
                {
                    "accepted": float(param_extras["accepted"]),
                    "log_alpha": float(param_extras["log_alpha"]),
                }
            )
        else:
            assert transform is not None
            assert z_unc is not None
            assert inverse_mass_matrix is not None
            z_unc, param_extras = _param_nuts_step(
                z_unc,
                state.latent_trajectory,
                param_key,
                bundle,
                transform,
                step_size=current_param_step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                max_num_doublings=int(nuts_max_num_doublings),
            )
            params = transform.constrain_to_tuple(z_unc)
            if adapt_step_size and i < num_warmup_adapt:
                z_unc_warmup_samples.append(z_unc)
            param_diagnostics.append(
                {
                    "divergent": float(param_extras["divergent"]),
                    "energy": float(param_extras["energy"]),
                    "num_integration_steps": float(param_extras["num_integration_steps"]),
                    "acceptance_rate": float(param_extras["acceptance_rate"]),
                }
            )

        traj_accept_obs = float(traj_extras["accepted"])
        traj_samples.append(state.latent_trajectory)
        param_samples.append(params)
        traj_accepts.append(traj_accept_obs)
        log_alpha_traj_history.append(float(traj_extras["log_alpha"]))
        latent_delta_history.append(float(state.latent_delta))
        param_step_size_history.append(current_param_step_size)

        # Robbins-Monro adaptation during warmup window. Targets the
        # acceptance rate of each kernel — RWM uses the binary accept
        # signal (0/1), NUTS uses the proposal-trajectory harmonic-mean
        # acceptance_rate (∈ [0, 1]) reported by blackjax. Note: full
        # NUTS dual averaging (with mass-matrix adaptation) is left for
        # Phase D-3; this is a lighter-weight single-parameter adapter
        # that brings NUTS into the same adaptation surface as RWM.
        if adapt_step_size and i < num_warmup_adapt:
            eta = adapt_learning_rate / float(jnp.sqrt(i + 1.0))
            # Trajectory MH step adaptation (always — both kernels share this)
            new_log_delta = float(jnp.log(state.latent_delta)) + eta * (
                traj_accept_obs - target_traj_accept
            )
            state = state._replace(
                latent_delta=jnp.asarray(jnp.exp(new_log_delta), dtype=state.latent_delta.dtype),
            )
            # Parameter step adaptation — kernel-dependent acceptance signal
            if param_kernel == "rwm":
                param_accept_obs = float(param_extras["accepted"])
            else:  # nuts
                param_accept_obs = float(param_extras["acceptance_rate"])
            new_log_step = (
                float(jnp.log(jnp.asarray(current_param_step_size)))
                + eta * (param_accept_obs - target_param_accept)
            )
            current_param_step_size = float(jnp.exp(new_log_step))

        # End-of-warmup hook: freeze NUTS inverse mass matrix to the
        # diagonal sample variance of warmup z_unc draws (+ jitter).
        # Standard NUTS preconditioning trick — bigger inv_mass on
        # high-variance dims gives the sampler bigger steps in those
        # directions. Done once at the warmup→sampling boundary so the
        # sampling phase is detailed-balance-preserving.
        if (
            adapt_step_size
            and param_kernel == "nuts"
            and i == num_warmup_adapt - 1
            and len(z_unc_warmup_samples) >= 2
        ):
            z_stack = jnp.stack(z_unc_warmup_samples, axis=0)
            z_var = jnp.var(z_stack, axis=0) + 1e-6
            adapted_inverse_mass_matrix = z_var
            inverse_mass_matrix = adapted_inverse_mass_matrix

    return {
        "traj_samples": traj_samples,
        "param_samples": param_samples,
        "traj_accepts": traj_accepts,
        "param_diagnostics": param_diagnostics,
        "log_alpha_traj_history": log_alpha_traj_history,
        "latent_delta_history": latent_delta_history,
        "param_step_size_history": param_step_size_history,
        "final_state": state,
        "final_params": params,
        "final_latent_delta": float(state.latent_delta),
        "final_param_step_size": current_param_step_size,
        "initial_inverse_mass_matrix": initial_inverse_mass_matrix,
        "adapted_inverse_mass_matrix": adapted_inverse_mass_matrix,
    }


def fit_composite_aux_kalman(
    bundle: CompositeAuxKalmanBundle,
    *,
    n_iterations: int = 50,
    num_warmup: int = 0,
    num_chains: int = 1,
    latent_delta: float = 0.05,
    param_step_size: float = 0.05,
    rng_key: Array,
    initial_x_traj: Array | None = None,
    initial_params: tuple[dict[str, Array], ...] | None = None,
    parallel: bool = False,
    param_kernel: Literal["rwm", "nuts"] = "rwm",
    nuts_max_num_doublings: int = 6,
    nuts_inverse_mass_matrix: Array | None = None,
    adapt_step_size: bool = False,
    target_param_accept: float = 0.3,
    target_traj_accept: float = 0.65,
    adapt_learning_rate: float = 0.1,
    init_method: Literal["prior", "pathfinder"] = "prior",
    pathfinder_n_starts: int = 4,
    pathfinder_maxiter: int = 20,
    pathfinder_elbo_samples: int = 10,
) -> InferenceResult:
    """Run a Gibbs-style MCMC on the composite-spec target.

    Phase-D-1 features (warmup + multi-chain):

    - ``num_warmup`` iterations run first per chain and are discarded
      from posterior samples. Tuning could happen here in a future
      iteration; today it's strictly burn-in.
    - ``num_chains`` independent chains run sequentially (Python loop;
      composite is smoke-scale). Each chain gets its own ``rng_key``
      derived via :func:`jax.random.fold_in`. Samples are concatenated
      across chains; chain-grouped samples are stored on
      ``diagnostics["chain_samples"]`` for r̂ / ESS computation.

    Args:
        bundle: Built via ``build_composite_aux_kalman_bundle``.
        n_iterations: Post-warmup MCMC iterations per chain.
        num_warmup: Warmup iterations per chain (discarded).
        num_chains: Number of independent chains.
        latent_delta: Step size for the trajectory MH.
        param_step_size: Step size for the parameter kernel.
        rng_key: JAX PRNG key seeding all chains.
        initial_x_traj: ``(T, n_latent)`` initial trajectory (shared across chains).
        initial_params: Initial parameter tuple (shared across chains).
        parallel: parallel-scan flag for the auxiliary filter.
        param_kernel: ``"rwm"`` (default) or ``"nuts"``.
        nuts_max_num_doublings: Trajectory-doubling cap for NUTS.
        nuts_inverse_mass_matrix: ``(dim,)`` or ``(dim, dim)`` mass
            matrix for NUTS. Defaults to identity.

    Returns:
        :class:`InferenceResult` with samples concatenated across chains
        (leading axis = ``num_chains * n_iterations``). Diagnostics
        include ``chain_samples`` (shape ``(num_chains, n_iterations, …)``),
        ``num_chains``, ``num_samples_per_chain``, ``num_warmup``.
    """
    if num_warmup < 0:
        raise ValueError(f"num_warmup must be ≥ 0, got {num_warmup}")
    if num_chains < 1:
        raise ValueError(f"num_chains must be ≥ 1, got {num_chains}")

    total_iters_per_chain = num_warmup + n_iterations

    # Run each chain sequentially. Composite fits are smoke-scale today;
    # vectorising the chain loop is a Phase-D-2 follow-up.
    chain_results: list[dict[str, Any]] = []
    for chain_idx in range(num_chains):
        chain_key = random.fold_in(rng_key, chain_idx)
        ch = _run_single_chain(
            bundle,
            n_iterations=total_iters_per_chain,
            latent_delta=latent_delta,
            param_step_size=param_step_size,
            rng_key=chain_key,
            initial_x_traj=initial_x_traj,
            initial_params=initial_params,
            parallel=parallel,
            param_kernel=param_kernel,
            nuts_max_num_doublings=nuts_max_num_doublings,
            nuts_inverse_mass_matrix=nuts_inverse_mass_matrix,
            num_warmup_adapt=num_warmup,
            adapt_step_size=adapt_step_size,
            target_param_accept=target_param_accept,
            target_traj_accept=target_traj_accept,
            adapt_learning_rate=adapt_learning_rate,
            init_method=init_method,
            pathfinder_n_starts=pathfinder_n_starts,
            pathfinder_maxiter=pathfinder_maxiter,
            pathfinder_elbo_samples=pathfinder_elbo_samples,
        )
        # Discard warmup from this chain. Step-size history is preserved
        # for diagnostics (warmup adaptation trajectory is useful to see).
        if num_warmup > 0:
            ch["traj_samples"] = ch["traj_samples"][num_warmup:]
            ch["param_samples"] = ch["param_samples"][num_warmup:]
            ch["traj_accepts"] = ch["traj_accepts"][num_warmup:]
            ch["param_diagnostics"] = ch["param_diagnostics"][num_warmup:]
            ch["log_alpha_traj_history"] = ch["log_alpha_traj_history"][num_warmup:]
        chain_results.append(ch)

    # Concatenate across chains for the flat InferenceResult.samples shape.
    all_param_samples: list[tuple[dict[str, Array], ...]] = []
    all_traj_samples: list[Array] = []
    all_traj_accepts: list[float] = []
    all_param_diagnostics: list[dict[str, float]] = []
    all_log_alpha: list[float] = []
    for ch in chain_results:
        all_param_samples.extend(ch["param_samples"])
        all_traj_samples.extend(ch["traj_samples"])
        all_traj_accepts.extend(ch["traj_accepts"])
        all_param_diagnostics.extend(ch["param_diagnostics"])
        all_log_alpha.extend(ch["log_alpha_traj_history"])

    traj_stack = jnp.stack(all_traj_samples, axis=0)
    samples_flat = _stack_param_samples(all_param_samples, prefix="vf")

    # Chain-grouped samples (shape ``(num_chains, n_iterations, *param_shape)``)
    # — what NumPyro / ArviZ r̂ + ESS routines consume.
    if all_param_samples:
        per_chain_stacked = [
            _stack_param_samples(ch["param_samples"], prefix="vf")
            for ch in chain_results
        ]
        site_names = list(per_chain_stacked[0].keys())
        chain_samples = {
            name: jnp.stack([pc[name] for pc in per_chain_stacked], axis=0)
            for name in site_names
        }
    else:
        chain_samples = {}

    # Reconstruct a canonical-model envelope from the bundle so Stage 6
    # / abduction estimators / downstream consumers can run on a self-
    # contained handle. The predictive_sampler (if any) doesn't ride
    # along — EKS/IEKS only need obs_kernel.response_fn / variance_fn,
    # which the kernel carries directly.
    from nof1_causal_lab.models.ssm.dynamics import runtime_from_composite

    canonical_for_diag = runtime_from_composite(
        bundle.compiled,
        init_mean=bundle.init_mean,
        init_cov=bundle.init_cov,
        diffusion_cov=bundle.diffusion_cov,
        H=bundle.H,
        d_meas=bundle.d_meas,
        R=bundle.R,
        obs_kernel=bundle.obs_kernel,
    )

    last_chain = chain_results[-1]
    diagnostics: dict[str, Any] = {
        "trajectory_samples": traj_stack,
        "param_samples": all_param_samples,
        "chain_samples": chain_samples,
        "num_chains": num_chains,
        "num_samples_per_chain": n_iterations,
        "num_warmup": num_warmup,
        "vector_field": bundle.compiled.vector_field,
        "canonical_model": canonical_for_diag,
        "trajectory_accept": jnp.asarray(all_traj_accepts),
        "log_alpha_traj": jnp.asarray(all_log_alpha),
        "final_state": last_chain["final_state"],
        "final_params": last_chain["final_params"],
        "param_kernel": param_kernel,
        "param_diagnostics": all_param_diagnostics,
        "adapt_step_size": adapt_step_size,
        "final_latent_delta_per_chain": [ch["final_latent_delta"] for ch in chain_results],
        "final_param_step_size_per_chain": [
            ch["final_param_step_size"] for ch in chain_results
        ],
        "latent_delta_history_per_chain": [
            ch["latent_delta_history"] for ch in chain_results
        ],
        "param_step_size_history_per_chain": [
            ch["param_step_size_history"] for ch in chain_results
        ],
        "adapted_inverse_mass_matrix_per_chain": [
            ch["adapted_inverse_mass_matrix"] for ch in chain_results
        ],
    }
    if param_kernel == "rwm":
        diagnostics["param_accept"] = jnp.asarray(
            [d["accepted"] for d in all_param_diagnostics]
        )
    else:
        diagnostics["param_divergent"] = jnp.asarray(
            [d["divergent"] for d in all_param_diagnostics]
        )
        diagnostics["param_energy"] = jnp.asarray(
            [d["energy"] for d in all_param_diagnostics]
        )

    return InferenceResult(
        _samples=samples_flat,
        method="composite_aux_kalman",
        diagnostics=diagnostics,
    )
