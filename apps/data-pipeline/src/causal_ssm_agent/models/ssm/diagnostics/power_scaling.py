"""Post-fit parametric diagnostics for state-space models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.inference.utils import _build_eval_fns, _discover_sites

from .results import PowerScalingResult

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.inference import InferenceResult
    from causal_ssm_agent.models.ssm.model import SSMModel

logger = get_prefect_logger(__name__)


def _sum_sample_terms(values: jnp.ndarray) -> jnp.ndarray:
    """Sum event terms while preserving the leading sample axis."""
    arr = jnp.asarray(values)
    if arr.ndim <= 1:
        return arr
    return arr.reshape((arr.shape[0], -1)).sum(axis=1)


def power_scaling_sensitivity(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    result: InferenceResult,
    seed: int = 0,
    alpha_delta: float = 0.01,
) -> PowerScalingResult:
    """Post-fit power-scaling sensitivity diagnostic."""
    rng_key = random.PRNGKey(seed)

    backend = model.make_likelihood_backend()
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, trace_key, backend)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    _, unravel_fn = ravel_pytree(example_unc)

    log_lik_fn, _log_prior_unc_fn = _build_eval_fns(
        model,
        observations,
        times,
        site_info,
        unravel_fn,
        backend,
    )

    param_names = sorted(site_info.keys())
    parameter_bindings = list(getattr(model, "parameter_bindings", []) or [])
    bindings_by_site = {
        (str(entry["site_name"]), int(entry["flat_index"])): str(entry["parameter"])
        for entry in parameter_bindings
    }

    samples = result.get_samples()
    n_samples = next(iter(samples.values())).shape[0]

    constrained_by_site: dict[str, jnp.ndarray] = {}
    unconstrained_by_site: dict[str, jnp.ndarray] = {}
    flat_samples = []
    for sample_idx in range(n_samples):
        parts = []
        for name in param_names:
            if name in samples:
                if name not in constrained_by_site:
                    constrained_by_site[name] = jnp.asarray(samples[name])
                    unconstrained_by_site[name] = jax.vmap(site_info[name]["transform"].inv)(
                        constrained_by_site[name]
                    )
                unc_val = unconstrained_by_site[name][sample_idx]
                parts.append(unc_val.reshape(-1))
        if parts:
            flat_samples.append(jnp.concatenate(parts))

    if not flat_samples:
        return PowerScalingResult(
            prior_sensitivity={},
            likelihood_sensitivity={},
            diagnosis={},
        )

    z_samples = jnp.stack(flat_samples)

    batch_log_lik = jax.vmap(log_lik_fn)

    chunk_size = 32
    log_liks_parts = []
    for start in range(0, n_samples, chunk_size):
        chunk = z_samples[start : start + chunk_size]
        log_liks_parts.append(batch_log_lik(chunk))

    log_liks = jnp.concatenate(log_liks_parts)

    alpha = alpha_delta
    lik_log_weights = alpha * log_liks
    lik_log_weights = lik_log_weights - jax.nn.logsumexp(lik_log_weights)
    lik_weights = jnp.exp(lik_log_weights)

    prior_sensitivity = {}
    likelihood_sensitivity = {}
    diagnosis = {}
    psis_k_hat = {}

    for name in param_names:
        if name not in constrained_by_site:
            continue

        con_vals = constrained_by_site[name]
        unc_vals = unconstrained_by_site[name]
        flat_vals = unc_vals.reshape((n_samples, -1))
        param_mean = jnp.mean(flat_vals, axis=0)

        site_log_prob = _sum_sample_terms(site_info[name]["distribution"].log_prob(con_vals))
        site_log_jac = _sum_sample_terms(
            jax.vmap(site_info[name]["transform"].log_abs_det_jacobian)(unc_vals, con_vals)
        )
        site_prior_log_weights = alpha * (site_log_prob + site_log_jac)
        site_prior_log_weights = site_prior_log_weights - jax.nn.logsumexp(site_prior_log_weights)
        site_prior_weights = jnp.exp(site_prior_log_weights)

        prior_weighted_mean = jnp.sum(site_prior_weights[:, None] * flat_vals, axis=0)
        lik_weighted_mean = jnp.sum(lik_weights[:, None] * flat_vals, axis=0)

        prior_shift_vec = jnp.abs(prior_weighted_mean - param_mean) / alpha_delta
        lik_shift_vec = jnp.abs(lik_weighted_mean - param_mean) / alpha_delta
        _, kss = az.psislw(np.asarray(site_prior_log_weights))

        for flat_index in range(flat_vals.shape[1]):
            param_label = bindings_by_site.get((name, flat_index))
            if parameter_bindings and param_label is None:
                continue
            if param_label is None:
                param_label = name if flat_vals.shape[1] == 1 else f"{name}[{flat_index}]"

            prior_shift = float(prior_shift_vec[flat_index])
            lik_shift = float(lik_shift_vec[flat_index])

            prior_sensitivity[param_label] = prior_shift
            likelihood_sensitivity[param_label] = lik_shift
            psis_k_hat[param_label] = float(kss)

            if prior_shift > 0.05 and lik_shift < 0.05:
                diagnosis[param_label] = "prior_dominated"
            elif prior_shift > 0.05 and lik_shift > 0.05:
                diagnosis[param_label] = "prior_data_conflict"
            else:
                diagnosis[param_label] = "well_identified"

    return PowerScalingResult(
        prior_sensitivity=prior_sensitivity,
        likelihood_sensitivity=likelihood_sensitivity,
        diagnosis=diagnosis,
        psis_k_hat=psis_k_hat,
    )
