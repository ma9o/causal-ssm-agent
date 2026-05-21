"""Prior-predictive sampling for composite vector fields.

Composite prior predictive validation for nonlinear dynamics. The
linear registry runtime and this composite runtime both consume
canonical prior configs at their boundaries, while this module owns the
vector-field-specific trajectory and stability checks.

This is the validation hook Stage 4 (or the agentic repair flow) calls
when it has a composite spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.random as jr
from numpyro.handlers import seed

from nof1_causal_lab.models.ssm.dynamics.composite import compile_composite
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.serialization import composite_spec_from_dict
from nof1_causal_lab.models.ssm.dynamics.simulator import simulate
from nof1_causal_lab.models.ssm.dynamics.stability import check_jacobian_stability

if TYPE_CHECKING:
    from jax import Array

    from nof1_causal_lab.models.ssm.dynamics.composite import CompiledComposite


@dataclass(frozen=True)
class CompositePriorPredictive:
    """One draw of trajectories + per-draw stability verdicts.

    ``trajectories`` has shape ``(n_draws, T, n_latent)``. ``param_draws``
    is the per-draw parameter tuple suitable for replaying inference
    deterministically. ``stable`` is a boolean per-draw mask; ``False``
    means the Jacobian at the linearisation point had at least one
    non-negative-real eigenvalue.
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
        "failing_draw_indices": [int(i) for i in jnp.where(~pp.stable | ~pp.finite)[0].tolist()],
        "primary_score": float((n_unstable + n_nonfinite) / max(1, n_total)),
        "max_real_eigenvalue_per_draw": pp.max_real_eigenvalue,
    }


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

    1. Compile the spec via ``composite_spec_from_dict`` + ``compile_composite``. Compile
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
        compiled = compile_composite(composite_spec_from_dict(composite_spec_config))
    except (ValueError, KeyError) as exc:
        return CompositeAssemblyValidation(compile_ok=False, compile_error=str(exc))

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
