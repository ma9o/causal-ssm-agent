"""Do-operator for posterior predictive intervention effects.

Implements the standard Bayesian causal inference pattern:
1. Take posterior draws of drift (A) and continuous intercept (c)
2. Compute CT steady state: η* = -A⁻¹c
3. Apply do(X=x) by solving the modified linear system
4. Compare to baseline → treatment effect

Uses exact analytic solutions (no scan approximation).
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import vmap

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
    do_value: float,
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
) -> float:
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


def compute_interventions(
    samples: dict[str, jnp.ndarray],
    treatments: list[str],
    outcome: str,
    latent_names: list[str],
    causal_spec: dict | None = None,
    ppc_result: dict | None = None,
    manifest_names: list[str] | None = None,
    ps_result: dict | None = None,
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

    Returns:
        List of intervention result dicts, sorted by |effect_size| descending.
    """
    # Parse identifiability status
    id_status = causal_spec.get("identifiability") if causal_spec else None
    non_identifiable: set[str] = set()
    blocker_details: dict[str, list[str]] = {}
    if id_status:
        ni_map = id_status.get("non_identifiable_treatments", {})
        non_identifiable = set(ni_map.keys())
        blocker_details = {
            t: d.get("confounders", []) for t, d in ni_map.items() if isinstance(d, dict)
        }

    name_to_idx = {name: i for i, name in enumerate(latent_names)}
    outcome_idx = name_to_idx.get(outcome)

    def _skeleton(treatment_name: str) -> dict[str, Any]:
        return {
            "treatment": treatment_name,
            "effect_size": None,
            "credible_interval": None,
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

    results: list[dict[str, Any]] = []
    for treatment_name in treatments:
        treat_idx = name_to_idx.get(treatment_name)
        if treat_idx is None:
            entry = _skeleton(treatment_name)
            entry["warning"] = f"'{treatment_name}' not in latent model"
            results.append(entry)
            continue

        effects = vmap(lambda d, c, ti=treat_idx, oi=outcome_idx: treatment_effect(d, c, ti, oi))(
            drift_draws, cint_draws
        )

        mean_effect = float(jnp.mean(effects))
        q025 = float(jnp.percentile(effects, 2.5))
        q975 = float(jnp.percentile(effects, 97.5))
        prob_positive = float(jnp.mean(effects > 0))

        entry: dict[str, Any] = {
            "treatment": treatment_name,
            "effect_size": mean_effect,
            "credible_interval": (q025, q975),
            "prob_positive": prob_positive,
            "identifiable": treatment_name not in non_identifiable,
        }

        if treatment_name in non_identifiable:
            blockers = blocker_details.get(treatment_name, [])
            if blockers:
                entry["warning"] = f"Effect not identifiable (blocked by: {', '.join(blockers)})"
            else:
                entry["warning"] = "Effect not identifiable (missing proxies)"

        results.append(entry)

    # Sort by |effect_size| descending
    results.sort(
        key=lambda x: abs(x["effect_size"]) if x["effect_size"] is not None else 0,
        reverse=True,
    )

    # Attach PPC warnings to each treatment entry
    if ppc_result and ppc_result.get("checked", False) and ppc_result.get("warnings"):
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
                entry_ppc = [
                    w for w in ppc_result["warnings"] if w.get("variable") in relevant_vars
                ]
                if entry_ppc:
                    entry["ppc_warnings"] = entry_ppc
        else:
            # Lambda is fixed (identity) — not in posterior samples.
            # Attach all PPC warnings to every treatment.
            for entry in results:
                entry["ppc_warnings"] = ppc_result["warnings"]

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
