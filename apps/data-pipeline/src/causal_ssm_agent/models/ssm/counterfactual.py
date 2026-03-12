"""Do-operator for posterior predictive intervention effects.

Implements the standard Bayesian causal inference pattern:
1. Take posterior draws of drift (A) and continuous intercept (c)
2. Compute CT steady state: η* = -A⁻¹c
3. Apply do(X=x) by solving the modified linear system
4. Compare to baseline → treatment effect
5. (Optional) Forward-simulate intervention trajectory over time

Uses exact analytic solutions (no scan approximation) for steady-state,
and discrete-time forward simulation for temporal trajectories.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import vmap

from causal_ssm_agent.models.likelihoods.base import CHOL_JITTER
from causal_ssm_agent.models.ssm.discretization import discretize_system

logger = logging.getLogger(__name__)


def steady_state(drift: jnp.ndarray, cint: jnp.ndarray) -> jnp.ndarray:
    """Baseline CT steady state: η* = -A⁻¹c.

    Args:
        drift: (n, n) continuous-time drift matrix A (must be stable)
        cint: (n,) continuous intercept c

    Returns:
        (n,) steady-state latent vector
    """
    return -jnp.linalg.solve(drift, cint)


def do(
    drift: jnp.ndarray,
    cint: jnp.ndarray,
    do_idx: int,
    do_value: float | jnp.ndarray,
) -> jnp.ndarray:
    """CT steady state under do(η_j = v).

    Solves the modified linear system where the do-variable row is
    replaced with the constraint η_j = v.

    Args:
        drift: (n, n) continuous-time drift matrix A
        cint: (n,) continuous intercept c
        do_idx: index of the variable to intervene on
        do_value: value to clamp to

    Returns:
        (n,) steady-state latent vector under intervention
    """
    # Replace row do_idx with identity constraint: η[do_idx] = do_value
    A_mod = drift.at[do_idx, :].set(0.0).at[do_idx, do_idx].set(1.0)
    rhs = (-cint).at[do_idx].set(do_value)

    # Warn if modified drift matrix is near-singular (safe inside vmap/jit)
    cond = jnp.linalg.cond(A_mod)
    jax.lax.cond(
        cond > 1e8,
        lambda: jax.debug.print(
            "do(): drift matrix near-singular (cond={c:.2e}), intervention may be unreliable",
            c=cond,
        ),
        lambda: None,
    )

    return jnp.linalg.solve(A_mod, rhs)


def treatment_effect(
    drift: jnp.ndarray,
    cint: jnp.ndarray,
    treat_idx: int,
    outcome_idx: int,
    shift_size: float = 1.0,
) -> jnp.ndarray:
    """Effect of intervention: do(treat = baseline + shift_size) vs baseline.

    Args:
        drift: (n, n) continuous-time drift matrix A
        cint: (n,) continuous intercept c
        treat_idx: index of treatment variable
        outcome_idx: index of outcome variable
        shift_size: size of the intervention shift (default 1.0). Callers can
            normalise by baseline SD or use percentage-based shifts so that
            effects are comparable across latents with different scales.

    Returns:
        Scalar treatment effect on outcome for this single posterior draw.
        Vmap over draws externally.
    """
    baseline = steady_state(drift, cint)
    do_value = baseline[treat_idx] + shift_size
    intervened = do(drift, cint, treat_idx, do_value)
    return intervened[outcome_idx] - baseline[outcome_idx]


def forward_simulate_intervention(
    drift: jnp.ndarray,
    cint: jnp.ndarray,
    treat_idx: int,
    outcome_idx: int,
    shift_size: float,
    dt: float,
    horizon_steps: int,
) -> jnp.ndarray:
    """Forward-simulate an intervention trajectory over time.

    Discretizes the CT system once, finds a baseline via iterative convergence,
    then clamps the treatment variable at each step and records the outcome
    trajectory.

    Args:
        drift: (n, n) continuous-time drift matrix A
        cint: (n,) continuous intercept c
        treat_idx: Index of the treatment variable to clamp
        outcome_idx: Index of the outcome variable to track
        shift_size: Size of the intervention shift above baseline
        dt: Time step in fractional days
        horizon_steps: Number of forward steps to simulate

    Returns:
        (horizon_steps,) array of outcome effects relative to baseline
    """
    n = drift.shape[0]
    # Discretize once
    diffusion_cov = jnp.zeros((n, n))  # no noise for mean trajectory
    Ad, _, cd = discretize_system(drift, diffusion_cov, cint, dt)
    # cd may be None if cint is zero; handle gracefully
    if cd is None:
        cd = jnp.zeros(n)

    # Find baseline via iterative convergence (avoids A⁻¹)
    def _converge_step(eta, _):
        return Ad @ eta + cd, None

    eta0 = jnp.zeros(n)
    baseline, _ = jax.lax.scan(_converge_step, eta0, None, length=500)
    baseline_outcome = baseline[outcome_idx]

    # Intervened value
    do_value = baseline[treat_idx] + shift_size

    # Forward simulate with treatment clamped
    def _step(eta, _):
        eta_next = Ad @ eta + cd
        eta_next = eta_next.at[treat_idx].set(do_value)
        return eta_next, eta_next[outcome_idx] - baseline_outcome

    init = baseline.at[treat_idx].set(do_value)
    _, trajectory = jax.lax.scan(_step, init, None, length=horizon_steps)
    return trajectory


def _summarize_trajectory(
    trajectory: jnp.ndarray,
    dt: float,
) -> dict[str, float]:
    """Extract summary statistics from an effect trajectory.

    Args:
        trajectory: (horizon_steps,) array of effects over time
        dt: Time step in fractional days

    Returns:
        Dict with temporal summary keys
    """
    steps_1d = min(int(1.0 / dt), len(trajectory))
    steps_7d = min(int(7.0 / dt), len(trajectory))
    steps_30d = min(int(30.0 / dt), len(trajectory))

    effect_1d = float(trajectory[steps_1d - 1]) if steps_1d > 0 else 0.0
    effect_7d = float(trajectory[steps_7d - 1]) if steps_7d > 0 else 0.0
    effect_30d = float(trajectory[steps_30d - 1]) if steps_30d > 0 else 0.0

    abs_traj = jnp.abs(trajectory)
    peak_idx = int(jnp.argmax(abs_traj))
    peak_effect = float(trajectory[peak_idx])
    time_to_peak_days = float((peak_idx + 1) * dt)

    return {
        "effect_1d": effect_1d,
        "effect_7d": effect_7d,
        "effect_30d": effect_30d,
        "peak_effect": peak_effect,
        "time_to_peak_days": time_to_peak_days,
    }


def compute_interventions(
    samples: dict[str, jnp.ndarray],
    treatments: list[str],
    outcome: str,
    latent_names: list[str],
    causal_spec: dict | None = None,
    ppc_result: dict | None = None,
    manifest_names: list[str] | None = None,
    ps_result: dict | None = None,
    times: jnp.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compute intervention effects for all treatments from posterior samples.

    Shared logic used by both local and GPU inference paths. Pure function
    that takes posterior samples and returns intervention result dicts.

    Args:
        samples: Posterior samples dict with keys "drift", "cint", optionally "lambda".
        treatments: List of treatment construct names.
        outcome: Name of the outcome variable.
        latent_names: Ordered list of latent construct names (maps to drift indices).
        causal_spec: Optional CausalSpec dict with identifiability status.
        ppc_result: Optional PPC result dict for attaching per-treatment warnings.
        manifest_names: Manifest variable names (needed if ppc_result provided).
        ps_result: Optional power-scaling result dict for flagging prior-dominated effects.
        times: Optional observation time points (fractional days). When provided,
            forward simulation is run alongside steady-state analysis.

    Returns:
        List of intervention result dicts, sorted by |effect_size| descending.
    """
    # Parse identifiability status
    id_status = causal_spec.get("identifiability") if causal_spec else None
    non_identifiable: set[str] = set()
    if id_status:
        ni_map = id_status.get("non_identifiable_treatments", {})
        non_identifiable = set(ni_map.keys())

    name_to_idx = {name: i for i, name in enumerate(latent_names)}
    outcome_idx = name_to_idx.get(outcome)

    def _skeleton(treatment_name: str) -> dict[str, Any]:
        return {
            "treatment": treatment_name,
            "effect_size": None,
            "identifiable": treatment_name not in non_identifiable,
        }

    # Guard: outcome not in latent names
    if outcome_idx is None:
        logger.warning("Outcome '%s' not found in latent names %s", outcome, latent_names)
        return [_skeleton(t) for t in treatments]

    # Guard: no drift draws
    drift_draws = samples.get("drift")
    if drift_draws is None:
        logger.warning("No 'drift' in posterior samples")
        return [_skeleton(t) for t in treatments]

    n_latent = drift_draws.shape[-1]
    cint_draws = samples.get("cint")
    if cint_draws is None:
        cint_draws = jnp.zeros((drift_draws.shape[0], n_latent))

    # Pre-compute forward simulation parameters
    # Use model_clock directly if available (exact); fall back to median(diff(times))
    dt_median: float | None = None
    horizon_steps: int | None = None
    model_clock_str = (causal_spec or {}).get("measurement", {}).get("model_clock")
    if model_clock_str:
        from causal_ssm_agent.orchestrator.schemas import parse_duration_to_hours
        dt_median = parse_duration_to_hours(model_clock_str) / 24.0
    elif times is not None and len(times) > 1:
        diffs = jnp.diff(times)
        dt_median = float(jnp.median(diffs))
    if dt_median is not None and dt_median > 0:
        horizon_steps = int(30.0 / dt_median)  # 30-day horizon

    # Lambda for manifest-level projection
    lambda_draws = samples.get("lambda")
    lambda_mean: jnp.ndarray | None = None
    if lambda_draws is not None:
        lambda_mean = jnp.mean(lambda_draws, axis=0) if lambda_draws.ndim == 3 else lambda_draws

    results: list[dict[str, Any]] = []
    for treatment_name in treatments:
        treat_idx = name_to_idx.get(treatment_name)
        if treat_idx is None:
            logger.warning("'%s' not in latent model — skipping", treatment_name)
            results.append(_skeleton(treatment_name))
            continue

        effects = vmap(lambda d, c, ti=treat_idx, oi=outcome_idx: treatment_effect(d, c, ti, oi))(
            drift_draws, cint_draws
        )

        mean_effect = float(jnp.mean(effects))
        prob_positive = float(jnp.mean(effects > 0))

        entry: dict[str, Any] = {
            "treatment": treatment_name,
            "effect_size": mean_effect,
            "posterior_draws": effects.tolist(),
            "prob_positive": prob_positive,
            "identifiable": treatment_name not in non_identifiable,
        }

        # Forward simulation for temporal effects
        if dt_median is not None and horizon_steps is not None and horizon_steps > 0:
            try:
                trajectories = vmap(
                    lambda d, c, ti=treat_idx, oi=outcome_idx: forward_simulate_intervention(
                        d, c, ti, oi, shift_size=1.0, dt=dt_median, horizon_steps=horizon_steps
                    )
                )(drift_draws, cint_draws)
                mean_traj = jnp.mean(trajectories, axis=0)
                entry["temporal"] = _summarize_trajectory(mean_traj, dt_median)

                # Manifest-level effects via lambda projection
                # lambda_mean is (n_manifest, n_latent); column outcome_idx
                # gives each manifest's loading on the outcome latent.
                if lambda_mean is not None and lambda_mean.ndim == 2:
                    m_names = manifest_names or []
                    loadings = lambda_mean[:, outcome_idx]
                    manifest_effects = {}
                    for mi in range(len(loadings)):
                        loading_val = float(loadings[mi])
                        if abs(loading_val) > CHOL_JITTER:
                            name = m_names[mi] if mi < len(m_names) else f"manifest_{mi}"
                            manifest_effects[name] = loading_val * mean_effect
                    if manifest_effects:
                        entry["manifest_effects"] = manifest_effects
            except Exception:
                logger.debug("Forward simulation failed for '%s'", treatment_name, exc_info=True)

        results.append(entry)

    # Sort by |effect_size| descending
    results.sort(
        key=lambda x: abs(x["effect_size"]) if x["effect_size"] is not None else 0,
        reverse=True,
    )

    # Attach PPC warnings to each treatment entry
    ppc_warnings = ppc_result.get("per_variable_warnings", []) if ppc_result else []
    if ppc_result and ppc_result.get("checked", False) and ppc_warnings:
        from causal_ssm_agent.models.posterior_predictive import get_relevant_manifest_variables

        lambda_mat = samples.get("lambda")
        if lambda_mat is not None and lambda_mat.ndim == 3:
            lambda_mat = jnp.mean(lambda_mat, axis=0)

        m_names = manifest_names or []

        if lambda_mat is not None:
            for entry in results:
                ti = name_to_idx.get(entry["treatment"])
                relevant_vars = get_relevant_manifest_variables(
                    lambda_mat, ti, outcome_idx, m_names
                )
                entry_ppc = [w for w in ppc_warnings if w.get("variable") in relevant_vars]
                if entry_ppc:
                    entry["ppc_warnings"] = entry_ppc
        else:
            # Lambda is fixed (identity) — not in posterior samples.
            # Attach all PPC warnings to every treatment.
            for entry in results:
                entry["ppc_warnings"] = ppc_warnings

    # Attach power-scaling sensitivity warnings to intervention entries.
    # Flag treatments whose drift parameters are prior-dominated.
    if ps_result and ps_result.get("checked", False):
        diagnosis = ps_result.get("diagnosis", {})
        prior_dominated = {k for k, v in diagnosis.items() if v == "prior_dominated"}
        if prior_dominated:
            for entry in results:
                t_name = entry["treatment"]
                t_idx = name_to_idx.get(t_name)
                if t_idx is None:
                    continue
                # Check drift parameters involving this treatment
                relevant = []
                for param_name in prior_dominated:
                    if param_name.startswith("drift_offdiag") or (
                        (param_name.startswith("drift_diag") or param_name.startswith("beta_"))
                        and t_name in param_name
                    ):
                        relevant.append(param_name)
                if relevant:
                    entry["prior_sensitivity_warning"] = (
                        f"Effect may be prior-driven: parameters {relevant} "
                        f"are prior-dominated per power-scaling diagnostic"
                    )

    return results
