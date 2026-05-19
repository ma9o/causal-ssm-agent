"""Practical identifiability via profile likelihood."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.optimize

from nof1_causal_lab.models.ssm.parameterization import sample_prior_unconstrained

from .context import ParametricIdContext, get_diagnostics_sweep_context
from .results import ProfileLikelihoodResult

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMModel

CHI2_THRESHOLD_95 = 1.92
CHI2_THRESHOLD_99 = 3.32


def profile_likelihood(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    profile_params: list[str] | None = None,
    profile_indices: list[int] | None = None,
    n_grid: int = 20,
    confidence: float = 0.95,
    seed: int = 42,
    sweep_context: ParametricIdContext | None = None,
) -> ProfileLikelihoodResult:
    """Profile likelihood identifiability diagnostic."""
    rng_key = random.PRNGKey(seed)
    context = sweep_context or get_diagnostics_sweep_context(model)

    flat_dim = context.flat_dim
    param_names = context.param_names
    prior_state = model.get_prior_runtime_bundle().prior_state

    def neg_log_post(z, ps):
        value = -(context.log_lik_fn(z, observations, times) + context.log_prior_unc_fn(z, ps))
        return jnp.where(jnp.isfinite(value), value, jnp.array(1e10))

    prior_z, rng_key = sample_prior_unconstrained(rng_key, context.registry, prior_state)
    prior_stds = jnp.std(prior_z, axis=0)
    prior_stds = jnp.maximum(prior_stds, 0.1)

    z_init = jnp.median(prior_z, axis=0)
    map_result = jax.scipy.optimize.minimize(
        lambda z: neg_log_post(z, prior_state),
        z_init,
        method="BFGS",
    )
    z_map = map_result.x
    if not jnp.all(jnp.isfinite(z_map)):
        z_map = z_init
    mle_ll = float(context.log_lik_fn(z_map, observations, times))

    param_index = context.param_index
    scalar_names = context.scalar_names

    if profile_indices is not None:
        indices = [idx for idx in profile_indices if idx < flat_dim]
    elif profile_params is not None:
        indices = []
        for pname in profile_params:
            if pname in param_index:
                offset, size = param_index[pname]
                indices.extend(range(offset, offset + size))
    else:
        indices = list(range(flat_dim))

    threshold = CHI2_THRESHOLD_99 if confidence >= 0.99 else CHI2_THRESHOLD_95
    unc_map = context.unravel_fn(z_map)

    parameter_profiles = {}

    for param_idx in indices:
        scalar_name = scalar_names[param_idx]
        prior_std_j = float(prior_stds[param_idx])
        z_map_j = float(z_map[param_idx])

        grid_unc = jnp.linspace(
            z_map_j - 3 * prior_std_j,
            z_map_j + 3 * prior_std_j,
            n_grid,
        )

        profile_ll = []

        if flat_dim > 1:
            _param_idx = param_idx

            def _profile_point(z_mj_init, z_j_val, ps, _param_idx=_param_idx):
                def _obj(z_mj):
                    z_full = jnp.concatenate([z_mj[:_param_idx], z_j_val[None], z_mj[_param_idx:]])
                    return neg_log_post(z_full, ps)

                res = jax.scipy.optimize.minimize(_obj, z_mj_init, method="BFGS")
                z_opt = jnp.concatenate([res.x[:_param_idx], z_j_val[None], res.x[_param_idx:]])
                ll_val = context.log_lik_fn(z_opt, observations, times)
                return res.x, ll_val

            z_mj_warm = jnp.concatenate([z_map[:param_idx], z_map[param_idx + 1 :]])

            for grid_idx in range(n_grid):
                grid_val = grid_unc[grid_idx]
                z_mj_opt, ll_val = _profile_point(z_mj_warm, grid_val, prior_state)
                if jnp.all(jnp.isfinite(z_mj_opt)):
                    z_mj_warm = z_mj_opt
                profile_ll.append(float(ll_val))
        else:
            for grid_idx in range(n_grid):
                z_full = grid_unc[grid_idx : grid_idx + 1]
                profile_ll.append(float(context.log_lik_fn(z_full, observations, times)))

        profile_ll = jnp.array(profile_ll)

        grid_con = grid_unc
        mle_value = z_map_j
        for name in param_names:
            offset, size = param_index[name]
            if offset <= param_idx < offset + size:
                local_idx = param_idx - offset
                con_vals = []
                for grid_val in grid_unc:
                    z_temp = z_map.at[param_idx].set(grid_val)
                    unc_dict = context.unravel_fn(z_temp)
                    con_val = context.transforms[name](unc_dict[name])
                    flat_con = con_val.reshape(-1)
                    con_vals.append(float(flat_con[local_idx]))
                grid_con = jnp.array(con_vals)
                con_map = context.transforms[name](unc_map[name])
                flat_map = con_map.reshape(-1)
                mle_value = float(flat_map[local_idx])
                break

        parameter_profiles[scalar_name] = {
            "grid_unc": grid_unc,
            "grid_con": grid_con,
            "profile_ll": profile_ll,
            "mle_value": mle_value,
        }

    mle_params = {name: context.transforms[name](unc_map[name]) for name in unc_map}

    return ProfileLikelihoodResult(
        parameter_profiles=parameter_profiles,
        mle_ll=mle_ll,
        mle_params=mle_params,
        threshold=threshold,
        parameter_names=[scalar_names[idx] for idx in indices],
    )
