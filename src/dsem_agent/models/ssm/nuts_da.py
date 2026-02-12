"""NUTS Data Augmentation for state-space models.

Data augmentation MCMC (Tanner & Wong 1987) augments the parameter space
with all latent states and samples the joint posterior p(theta, eta_{1:T} | y_{1:T})
using NUTS. This is the "everything is parameters" approach -- the standard
approach in Stan/NumPyro.

Supports two parameterizations:

- Centered (default): Sample eta_t ~ N(A_d @ eta_{t-1} + c_d, Q_d) directly.
  With SVI warmstart, this works well even at high T because the initialization
  places the chain near the posterior mode and provides a good mass matrix
  estimate.

- Non-centered: Sample standardized innovations eps_t ~ N(0, I)
  and compute eta_t = A_d @ eta_{t-1} + c_d + Q_d_chol @ eps_t.
  Better without warmstart when process noise dominates, but worse when
  observations are informative (creates high curvature in eps directions).

SVI warmstart (enabled by default) runs variational inference on the
Kalman-likelihood model (parameter space only, ~50D) to find good parameter
estimates. A Kalman smoother then provides optimal state estimates. Combined,
these initialize NUTS near the posterior mode, breaking the chicken-and-egg
problem of mass matrix adaptation in high dimensions.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax import vmap
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median, init_to_value
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import ClippedAdam

from dsem_agent.models.ssm.discretization import discretize_system_batched
from dsem_agent.models.ssm.inference import InferenceResult

if TYPE_CHECKING:
    from dsem_agent.models.ssm.model import SSMModel


def _da_model(
    ssm_model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,  # noqa: ARG001
    centered: bool = True,
) -> None:
    """NumPyro model for data-augmentation MCMC.

    Samples model parameters AND latent states jointly.

    With centered parameterization (default), samples states directly from
    the transition distribution N(A_d @ eta_{t-1} + c_d, Q_d).

    With non-centered parameterization, samples standardized innovations
    eps_t ~ N(0, I) and computes states deterministically as
    eta_t = A_d @ eta_{t-1} + c_d + Q_d_chol @ eps_t.

    Uses numpyro.contrib.control_flow.scan for efficient sequential
    sampling within NUTS.
    """
    spec = ssm_model.spec
    n_l = spec.n_latent
    T = observations.shape[0]

    # --- Sample all model parameters (reuse SSMModel machinery) ---
    drift = ssm_model._sample_drift(spec)
    diffusion_chol = ssm_model._sample_diffusion(spec)
    cint = ssm_model._sample_cint(spec)
    lambda_mat = ssm_model._sample_lambda(spec)
    manifest_means, manifest_chol = ssm_model._sample_manifest_params(spec)
    t0_means, t0_chol = ssm_model._sample_t0_params(spec)

    # Convert to covariances for discretization
    if diffusion_chol.ndim == 2:
        diffusion_cov = diffusion_chol @ diffusion_chol.T
    else:
        diffusion_cov = vmap(lambda x: x @ x.T)(diffusion_chol)

    # --- Discretize system for all time intervals ---
    time_intervals = jnp.diff(times, prepend=times[0])
    time_intervals = time_intervals.at[0].set(1e-6)

    Ad_all, Qd_all, cd_all = discretize_system_batched(drift, diffusion_cov, cint, time_intervals)

    # Cholesky of Q_d for scan time steps only.
    # Skip index 0 (dt=1e-6 → nearly singular Qd) to avoid 0*NaN=NaN gradient.
    reg = 1e-5 * jnp.eye(n_l)
    Qd_chol_scan = vmap(lambda Q: jnp.linalg.cholesky(Q + reg))(Qd_all[1:])

    # --- Initial state ---
    if centered:
        eta_0 = numpyro.sample("eta_0", dist.MultivariateNormal(t0_means, scale_tril=t0_chol))
    else:
        # NCP: sample standardized innovation, transform deterministically
        eps_0 = numpyro.sample("eps_0", dist.Normal(0, 1).expand([n_l]).to_event(1))
        eta_0 = t0_means + t0_chol @ eps_0

    # Score first observation
    pred_0 = lambda_mat @ eta_0 + manifest_means
    numpyro.sample(
        "obs_0",
        dist.MultivariateNormal(loc=pred_0, scale_tril=manifest_chol),
        obs=observations[0],
    )

    # --- Sequential state sampling via scan ---
    cd_scan = cd_all[1:] if cd_all is not None else jnp.zeros((T - 1, n_l))

    if centered:

        def transition(eta_prev, inputs):
            Ad, Qd_chol, cd, obs_t = inputs
            mean = Ad @ eta_prev + cd
            eta_t = numpyro.sample("eta", dist.MultivariateNormal(mean, scale_tril=Qd_chol))
            pred_t = lambda_mat @ eta_t + manifest_means
            numpyro.sample(
                "obs",
                dist.MultivariateNormal(loc=pred_t, scale_tril=manifest_chol),
                obs=obs_t,
            )
            return eta_t, None

        scan(
            transition,
            eta_0,
            (Ad_all[1:], Qd_chol_scan, cd_scan, observations[1:]),
            length=T - 1,
        )
    else:

        def transition(eta_prev, inputs):
            Ad, Qd_chol, cd, obs_t = inputs
            eps = numpyro.sample("eps", dist.Normal(0, 1).expand([n_l]).to_event(1))
            eta_t = Ad @ eta_prev + cd + Qd_chol @ eps
            pred_t = lambda_mat @ eta_t + manifest_means
            numpyro.sample(
                "obs",
                dist.MultivariateNormal(loc=pred_t, scale_tril=manifest_chol),
                obs=obs_t,
            )
            return eta_t, None

        scan(
            transition,
            eta_0,
            (Ad_all[1:], Qd_chol_scan, cd_scan, observations[1:]),
            length=T - 1,
        )


# ---------------------------------------------------------------------------
# Kalman smoother for state initialization
# ---------------------------------------------------------------------------


def _kalman_smooth_states(
    observations: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
) -> jnp.ndarray:
    """Kalman filter + RTS smoother for linear Gaussian SSM via cuthbert.

    Returns smoothed state means (T, D).
    """
    from cuthbert.filtering import filter as cuthbert_filter
    from cuthbert.gaussian.moments import build_filter, build_smoother
    from cuthbert.smoothing import smoother as cuthbert_smoother

    T, n_m = observations.shape
    n = Ad.shape[1]
    jitter_n = 1e-6 * jnp.eye(n)
    jitter_m = 1e-6 * jnp.eye(n_m)

    # Cholesky factors for cuthbert (square-root form)
    chol_Qd = vmap(lambda Q: jnp.linalg.cholesky(Q + jitter_n))(Qd)
    chol_R = jnp.linalg.cholesky(R + jitter_m)
    chol_P0 = jnp.linalg.cholesky(init_cov + jitter_n)

    # model_inputs with leading dim T (cuthbert convention:
    # [0] → init_prepare, [k>=1] → dynamics k-1→k + obs k)
    model_inputs = {
        "m0": jnp.broadcast_to(init_mean, (T, n)),
        "chol_P0": jnp.broadcast_to(chol_P0, (T, n, n)),
        "F": Ad,
        "c": cd,
        "chol_Q": chol_Qd,
        "H": jnp.broadcast_to(H, (T, n_m, n)),
        "d": jnp.broadcast_to(d, (T, n_m)),
        "chol_R": jnp.broadcast_to(chol_R, (T, n_m, n_m)),
        "y": observations,
    }

    # Callbacks matching KalmanLikelihood pattern
    def get_init_params(inputs):
        return inputs["m0"], inputs["chol_P0"]

    def get_dynamics_params(state, inputs):
        F_t, c_t, chol_Q_t = inputs["F"], inputs["c"], inputs["chol_Q"]

        def dynamics_fn(x):
            return F_t @ x + c_t, chol_Q_t

        return dynamics_fn, state.mean

    def get_observation_params(state, inputs):
        H_t, d_t, chol_R_t, y_t = inputs["H"], inputs["d"], inputs["chol_R"], inputs["y"]

        def obs_fn(x):
            return H_t @ x + d_t, chol_R_t

        return obs_fn, state.mean, y_t

    filter_obj = build_filter(
        get_init_params=get_init_params,
        get_dynamics_params=get_dynamics_params,
        get_observation_params=get_observation_params,
        associative=False,
    )
    filter_states = cuthbert_filter(filter_obj, model_inputs)

    smoother_obj = build_smoother(get_dynamics_params=get_dynamics_params)
    smoothed_states = cuthbert_smoother(smoother_obj, filter_states)

    return smoothed_states.mean


# ---------------------------------------------------------------------------
# Kalman-based warmstart
# ---------------------------------------------------------------------------


def _kalman_warmstart(
    ssm_model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None,
    centered: bool = True,
    num_steps: int = 2000,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> dict[str, jnp.ndarray] | None:
    """Warmstart for NUTS-DA via SVI on Kalman model + Kalman smoother.

    Strategy (with fallbacks):
    1. Try SVI on Kalman-likelihood model for parameter estimates (~50D).
       Falls back to SVI on particle-filter model if Kalman SVI fails.
    2. Use parameter estimates + Kalman smoother for state initialization.
       Falls back to pseudo-inverse of observations if smoother fails.
    3. Combine into init_to_value dict for the DA model.

    Returns None only if all strategies fail.
    """
    spec = ssm_model.spec
    n_l = spec.n_latent

    # --- Phase 1: Get parameter estimates via SVI ---
    param_init = None
    det_values = None

    # Try SVI on Kalman model first (fast, deterministic likelihood)
    param_init, det_values = _try_svi(
        ssm_model,
        observations,
        times,
        subject_ids,
        backend_type="kalman",
        num_steps=num_steps,
        learning_rate=learning_rate,
        seed=seed,
    )

    # Fall back to SVI on particle filter model (robust, stochastic)
    if param_init is None:
        param_init, det_values = _try_svi(
            ssm_model,
            observations,
            times,
            subject_ids,
            backend_type="particle",
            num_steps=num_steps,
            learning_rate=learning_rate * 0.5,
            seed=seed,
        )

    # --- Phase 2: Get state estimates ---
    smoothed = None

    if det_values is not None:
        # Try Kalman smoother with estimated parameters
        smoothed = _try_smoother(
            ssm_model,
            observations,
            times,
            det_values,
        )

    # Fall back to pseudo-inverse of observations
    if smoothed is None:
        print("  Using observation-based state initialization (pseudo-inverse)")
        # Lambda has identity top block: eta_t ≈ obs_t[:n_l]
        # This works for standard SSM specs where n_manifest >= n_latent
        smoothed = observations[:, :n_l].copy()

    print(
        f"  State init: shape={smoothed.shape}, "
        f"range=[{float(smoothed.min()):.3f}, {float(smoothed.max()):.3f}]"
    )

    # --- Phase 3: Build init_values dict ---
    init_values = dict(param_init) if param_init is not None else {}

    if centered:
        init_values["eta_0"] = smoothed[0]
        init_values["eta"] = smoothed[1:]
    else:
        # NCP: compute standardized innovations from smoothed states
        if det_values is not None:
            t0_cov = det_values["t0_cov"]
            t0_mean = det_values["t0_means"]
            t0_chol_val = jnp.linalg.cholesky(t0_cov + 1e-6 * jnp.eye(n_l))
            init_values["eps_0"] = jnp.linalg.solve(t0_chol_val, smoothed[0] - t0_mean)

            drift = det_values["drift"]
            diffusion_chol = det_values["diffusion"]
            diffusion_cov = diffusion_chol @ diffusion_chol.T
            cint = det_values.get("cint")

            time_intervals = jnp.diff(times, prepend=times[0])
            time_intervals = time_intervals.at[0].set(1e-6)
            Ad_all, Qd_all, cd_all = discretize_system_batched(
                drift, diffusion_cov, cint, time_intervals
            )
            cd_ = cd_all if cd_all is not None else jnp.zeros((len(time_intervals), n_l))
            Qd_chol_ncp = vmap(lambda Q: jnp.linalg.cholesky(Q + 1e-6 * jnp.eye(n_l)))(Qd_all)
            eps = jax.vmap(lambda prev, curr, A, Qc, c: jnp.linalg.solve(Qc, curr - A @ prev - c))(
                smoothed[:-1], smoothed[1:], Ad_all[1:], Qd_chol_ncp[1:], cd_[1:]
            )
            init_values["eps"] = eps
        else:
            # No parameter estimates available, use zero innovations
            init_values["eps_0"] = jnp.zeros(n_l)
            init_values["eps"] = jnp.zeros((observations.shape[0] - 1, n_l))

    # Diagnostic: print init_values summary
    for k, v in init_values.items():
        v_arr = jnp.asarray(v)
        print(
            f"  init[{k}]: shape={v_arr.shape}, "
            f"range=[{float(v_arr.min()):.4f}, {float(v_arr.max()):.4f}], "
            f"finite={bool(jnp.all(jnp.isfinite(v_arr)))}"
        )

    return init_values


def _try_svi(
    ssm_model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None,
    backend_type: str = "kalman",
    num_steps: int = 2000,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> tuple[dict | None, dict | None]:
    """Try SVI warmstart on a given backend. Returns (param_init, det_values) or (None, None)."""
    spec = ssm_model.spec

    if backend_type == "kalman":
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

        backend = KalmanLikelihood(n_latent=spec.n_latent, n_manifest=spec.n_manifest)
    else:
        backend = ssm_model.make_likelihood_backend()

    model_fn = functools.partial(ssm_model.model, likelihood_backend=backend)

    try:
        guide = AutoNormal(model_fn, init_loc_fn=init_to_median(num_samples=15))
        optimizer = ClippedAdam(step_size=learning_rate)
        svi = SVI(model_fn, guide, optimizer, Trace_ELBO())

        rng_key = random.PRNGKey(seed)
        svi_result = svi.run(
            rng_key, num_steps, observations, times, subject_ids, progress_bar=False
        )

        losses = svi_result.losses
        loss_0 = float(losses[0])
        loss_mid = float(losses[min(num_steps // 2, len(losses) - 1)])
        loss_end = float(losses[-1])
        print(
            f"  SVI ({backend_type}): loss@0={loss_0:.1f}, "
            f"loss@{num_steps // 2}={loss_mid:.1f}, loss@end={loss_end:.1f}"
        )

        if not jnp.isfinite(loss_end):
            print(f"  SVI ({backend_type}) failed: NaN/Inf loss")
            return None, None

        print(f"  SVI ({backend_type}) warmstart: {num_steps} steps, ELBO={-loss_end:.1f}")

        # Get parameter values from guide median (constrained space).
        # NOTE: Predictive(model, guide=guide) does NOT return sample sites
        # that are covered by the guide — it only returns deterministic sites
        # and observations. So we must get parameter values directly from the guide.
        param_init = guide.median(svi_result.params)
        print(f"  SVI param sites: {sorted(param_init.keys())}")

        # Get deterministic values from Predictive (model conditioned on guide params)
        predictive = Predictive(model_fn, guide=guide, params=svi_result.params, num_samples=1)
        sample = predictive(random.PRNGKey(seed + 1), observations, times, subject_ids)

        det_sites = {"drift", "diffusion", "cint", "lambda", "manifest_cov", "t0_means", "t0_cov"}
        det_values = {}
        for k, v in sample.items():
            if k in det_sites or k == "manifest_means":
                det_values[k] = v[0]

        return param_init, det_values

    except Exception as e:
        print(f"  SVI ({backend_type}) failed: {e}")
        return None, None


def _try_smoother(
    ssm_model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    det_values: dict,
) -> jnp.ndarray | None:
    """Try running Kalman smoother with estimated parameters."""
    spec = ssm_model.spec
    n_l = spec.n_latent

    try:
        drift = det_values["drift"]
        diffusion_chol = det_values["diffusion"]
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        lambda_mat = det_values["lambda"]
        manifest_cov = det_values["manifest_cov"]
        t0_mean = det_values["t0_means"]
        t0_cov = det_values["t0_cov"]
        cint = det_values.get("cint")

        # Get manifest means (prefer SVI estimate over spec default)
        if "manifest_means" in det_values:
            manifest_means_val = det_values["manifest_means"]
        elif isinstance(spec.manifest_means, jnp.ndarray):
            manifest_means_val = spec.manifest_means
        else:
            manifest_means_val = jnp.zeros(spec.n_manifest)

        time_intervals = jnp.diff(times, prepend=times[0])
        time_intervals = time_intervals.at[0].set(1e-6)

        Ad_all, Qd_all, cd_all = discretize_system_batched(
            drift, diffusion_cov, cint, time_intervals
        )
        cd_for_smoother = cd_all if cd_all is not None else jnp.zeros((len(time_intervals), n_l))

        smoothed = _kalman_smooth_states(
            observations,
            Ad_all,
            Qd_all,
            cd_for_smoother,
            lambda_mat,
            manifest_means_val,
            manifest_cov,
            t0_mean,
            t0_cov,
        )

        if not jnp.all(jnp.isfinite(smoothed)):
            print("  Kalman smoother produced NaN/Inf states")
            return None

        print(
            f"  Kalman smoother: states shape={smoothed.shape}, "
            f"range=[{float(smoothed.min()):.3f}, {float(smoothed.max()):.3f}]"
        )
        return smoothed

    except Exception as e:
        print(f"  Kalman smoother failed: {e}")
        return None


def _check_init_log_density(model_fn, init_values, observations, times, subject_ids, seed):
    """Diagnostic: check if init values produce finite log-density in the DA model."""
    import numpyro.handlers as handlers

    try:
        sub_model = handlers.seed(handlers.substitute(model_fn, data=init_values), rng_seed=seed)
        tr = handlers.trace(sub_model).get_trace(observations, times, subject_ids)

        total_lp = 0.0
        for name, site in tr.items():
            if site["type"] != "sample":
                continue
            lp = jnp.sum(site["fn"].log_prob(site["value"]))
            is_finite = bool(jnp.isfinite(lp))
            if not is_finite:
                v = site["value"]
                print(
                    f"  DIAG NaN/Inf: site={name}, logprob={float(lp):.2f}, "
                    f"value_range=[{float(jnp.min(v)):.4f}, {float(jnp.max(v)):.4f}], "
                    f"shape={v.shape}"
                )
            total_lp += lp

        print(
            f"  DIAG log-density at init: {float(total_lp):.2f}, finite={bool(jnp.isfinite(total_lp))}"
        )
    except Exception as e:
        print(f"  DIAG check failed: {e}")


def fit_nuts_da(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    dense_mass: bool = False,
    target_accept_prob: float = 0.80,
    max_tree_depth: int = 10,
    centered: bool = True,
    svi_warmstart: bool = True,
    svi_num_steps: int = 2000,
    svi_learning_rate: float = 0.01,
    **kwargs: Any,
) -> InferenceResult:
    """Fit using NUTS with data augmentation (joint parameter + state sampling).

    Data augmentation MCMC (Tanner & Wong 1987): augments the parameter space
    with all latent states eta_{0:T} and samples the joint posterior
    p(theta, eta_{0:T} | y_{1:T}) using NUTS.

    No particle filter or Kalman filter is used during sampling. Instead,
    NUTS explores the full joint space using gradient information from the
    Gaussian transition and observation densities.

    By default uses Kalman-based warmstart:
    1. SVI on Kalman-likelihood model (~50D params) for parameter estimates
    2. Kalman smoother for optimal state trajectory estimates
    3. Initialize NUTS from the combined parameter + state estimates

    Args:
        model: SSMModel instance defining the model specification and priors
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: (T,) subject indices (not yet supported, reserved)
        num_warmup: Number of warmup (adaptation) samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        seed: Random seed
        dense_mass: Use dense mass matrix
        target_accept_prob: Target acceptance probability for NUTS
        max_tree_depth: Maximum tree depth for NUTS
        centered: If True (default), use centered parameterization (sample
            states directly). If False, use non-centered (sample innovations).
        svi_warmstart: If True (default), run Kalman SVI + smoother warmstart.
        svi_num_steps: Number of SVI optimization steps for warmstart.
        svi_learning_rate: Learning rate for SVI warmstart.
        **kwargs: Additional MCMC arguments

    Returns:
        InferenceResult with posterior samples (latent states excluded)
    """
    model_fn = functools.partial(_da_model, model, centered=centered)

    # Determine initialization strategy
    init_values = None
    if svi_warmstart:
        init_values = _kalman_warmstart(
            model,
            observations,
            times,
            subject_ids,
            centered=centered,
            num_steps=svi_num_steps,
            learning_rate=svi_learning_rate,
            seed=seed + 1000,
        )

    # Diagnostic: check log-density at init values before NUTS
    if init_values is not None:
        _check_init_log_density(model_fn, init_values, observations, times, subject_ids, seed)

    # Try initialization strategies with fallback chain
    rng_key = random.PRNGKey(seed)
    state_sites = {"eta_0", "eta", "eps_0", "eps"}
    mcmc = None

    strategies = []
    if init_values is not None:
        strategies.append(("init_to_value (full)", init_to_value(values=init_values)))
        # Fallback: params only (let states init from default)
        param_only = {k: v for k, v in init_values.items() if k not in state_sites}
        if param_only:
            strategies.append(("init_to_value (params only)", init_to_value(values=param_only)))
    strategies.append(("init_to_median", init_to_median(num_samples=15)))

    for strategy_name, init_strategy in strategies:
        try:
            kernel = NUTS(
                model_fn,
                init_strategy=init_strategy,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                dense_mass=dense_mass,
                regularize_mass_matrix=True,
            )
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                **kwargs,
            )
            mcmc.run(rng_key, observations, times, subject_ids)
            print(f"  NUTS initialized with: {strategy_name}")
            break
        except RuntimeError as e:
            if "Cannot find valid initial parameters" in str(e):
                print(f"  {strategy_name} failed: {e}")
                continue
            raise

    # Get samples, excluding the large per-timestep state arrays
    all_samples = mcmc.get_samples()
    # Exclude latent state sites (both CP and NCP variants)
    state_sites = {"eta_0", "eta", "eps_0", "eps"}
    samples = {k: v for k, v in all_samples.items() if k not in state_sites}

    return InferenceResult(
        _samples=samples,
        method="nuts_da",
        diagnostics={"mcmc": mcmc},
    )
