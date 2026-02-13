"""Do-operator for posterior predictive intervention effects.

Implements the standard Bayesian causal inference pattern:
1. Take posterior draws of drift (A) and continuous intercept (c)
2. Compute CT steady state: η* = -A⁻¹c
3. Apply do(X=x) by solving the modified linear system
4. Compare to baseline → treatment effect

Uses exact analytic solutions (no scan approximation).
"""

import jax
import jax.numpy as jnp


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
