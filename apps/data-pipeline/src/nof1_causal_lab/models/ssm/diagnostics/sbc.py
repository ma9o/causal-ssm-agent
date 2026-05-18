"""Simulation-based calibration for SSM diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.ssm.inference.utils import _build_runtime_eval_fns_from_registry
from nof1_causal_lab.models.ssm.parameterization import sample_prior_unconstrained

from .results import SBCResult
from .simulation import _simulate_from_params

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMModel

logger = get_prefect_logger(__name__)


def sbc_check(
    model: SSMModel,
    T: int = 100,
    dt: float = 0.5,
    n_sbc: int = 50,
    method: Literal[
        "map",
        "svi",
        "aux_gibbs",
        "particle_mgrad",
    ] = "map",
    seed: int = 42,
    **fit_kwargs,
) -> SBCResult:
    """Simulation-based calibration check (Modrak et al. 2023)."""
    from nof1_causal_lab.models.ssm.inference import fit

    rng_key = random.PRNGKey(seed)
    times = jnp.arange(T, dtype=jnp.float64) * dt

    prior_runtime = model.get_prior_runtime_bundle()
    site_runtime = prior_runtime.site_runtime
    prior_state = prior_runtime.prior_state
    backend = model.make_likelihood_backend()
    log_lik_fn, _ = _build_runtime_eval_fns_from_registry(
        model.spec,
        site_runtime.registry,
        site_runtime.unravel_fn,
        site_runtime.transforms,
        model.structure_runtime,
        backend,
    )

    param_names = site_runtime.param_names
    param_index = site_runtime.param_index
    scalar_names = site_runtime.scalar_names
    registry = site_runtime.registry

    all_ranks: dict[str, list[int]] = {scalar_name: [] for scalar_name in scalar_names}
    ll_ranks: list[int] = []
    n_post = 0
    n_failed = 0

    for rep in range(n_sbc):
        prior_z, rng_key = sample_prior_unconstrained(
            rng_key,
            registry,
            prior_state,
            n_samples=1,
        )
        true_z = prior_z[0]
        true_con = site_runtime.constrain(true_z)

        rng_key, sim_key = random.split(rng_key)
        try:
            y_star = _simulate_from_params(true_con, model.spec, times, sim_key, registry=registry)
        except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
            logger.info("SBC replicate %d: simulation failed: %s", rep, exc)
            n_failed += 1
            continue

        if not jnp.all(jnp.isfinite(y_star)):
            n_failed += 1
            continue

        rng_key, fit_key = random.split(rng_key)
        try:
            fit_result = fit(
                model, y_star, times, method=method, seed=int(fit_key[0]), **fit_kwargs
            )
        except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
            logger.info("SBC replicate %d: fit failed: %s", rep, exc)
            n_failed += 1
            continue

        samples = fit_result.get_samples()
        if not samples:
            continue
        n_post = next(iter(samples.values())).shape[0]

        available = [name for name in param_names if name in samples]

        for name in available:
            _offset, size = param_index[name]
            true_flat = true_con[name].reshape(-1)
            post_flat = samples[name].reshape(n_post, -1)

            for idx in range(size):
                scalar_name = name if size == 1 else f"{name}[{idx}]"
                rank = int(jnp.sum(post_flat[:, idx] < true_flat[idx]))
                all_ranks[scalar_name].append(rank)

        if available:
            true_ll = float(log_lik_fn(true_z, y_star, times))

            post_z_list = []
            for draw_idx in range(n_post):
                parts = []
                for name in param_names:
                    if name in samples:
                        unc = site_runtime.transforms[name].inv(samples[name][draw_idx])
                        parts.append(jnp.asarray(unc).reshape(-1))
                if parts:
                    post_z_list.append(jnp.concatenate(parts))

            if post_z_list:
                post_z = jnp.stack(post_z_list)
                batch_ll = jax.vmap(log_lik_fn, in_axes=(0, None, None))
                post_lls = []
                chunk_size = 32
                for start in range(0, post_z.shape[0], chunk_size):
                    post_lls.append(batch_ll(post_z[start : start + chunk_size], y_star, times))
                post_lls = jnp.concatenate(post_lls)
                ll_rank = int(jnp.sum(post_lls < true_ll))
            else:
                ll_rank = 0
        else:
            ll_rank = 0
        ll_ranks.append(ll_rank)

    n_attempted = n_sbc
    failure_rate = n_failed / n_attempted if n_attempted > 0 else 0.0
    if failure_rate > 0.8:
        raise RuntimeError(
            f"SBC: {n_failed}/{n_attempted} replicates failed ({failure_rate:.0%}) "
            "— likely a model specification bug"
        )
    if failure_rate > 0.2:
        logger.warning(
            "SBC: %d/%d replicates failed (%.0f%%). Results may be biased toward stable "
            "parameter regimes.",
            n_failed,
            n_attempted,
            failure_rate * 100,
        )

    ranks_dict = {
        scalar_name: jnp.array(values) for scalar_name, values in all_ranks.items() if values
    }

    return SBCResult(
        ranks=ranks_dict,
        likelihood_ranks=jnp.array(ll_ranks) if ll_ranks else jnp.zeros(0),
        n_sbc=len(ll_ranks),
        n_posterior_samples=n_post,
        parameter_names=list(ranks_dict.keys()),
        n_failed=n_failed,
        n_attempted=n_attempted,
    )
