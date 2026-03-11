"""Post-fit parametric diagnostics for state-space models.

Power-scaling sensitivity: detect prior-dominated or conflicting parameters
by perturbing each component's contribution and measuring posterior shift
(Kallioinen et al. 2023).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.models.ssm.utils import (
    _build_eval_fns,
    _discover_sites,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.inference import InferenceResult
    from causal_ssm_agent.models.ssm.model import SSMModel

# Use parent module's logger so existing log-level filters work unchanged
logger = logging.getLogger("causal_ssm_agent.utils.parametric_id")


@dataclass
class PowerScalingResult:
    """Results from post-fit power-scaling sensitivity analysis."""

    prior_sensitivity: dict[str, float]
    likelihood_sensitivity: dict[str, float]
    diagnosis: dict[str, str]  # "prior_dominated" | "well_identified" | "prior_data_conflict"
    psis_k_hat: dict[str, float] = field(default_factory=dict)

    def print_report(self) -> None:
        """Log a human-readable power-scaling report."""
        lines = ["=== Power-Scaling Sensitivity Report ==="]
        for name in self.diagnosis:
            prior_s = self.prior_sensitivity.get(name, 0.0)
            lik_s = self.likelihood_sensitivity.get(name, 0.0)
            diag = self.diagnosis[name]
            k_hat = self.psis_k_hat.get(name, float("nan"))
            reliable = "reliable" if k_hat < 0.7 else "UNRELIABLE"
            lines.append(
                f"  {name}: prior_sens={prior_s:.3f}, lik_sens={lik_s:.3f} "
                f"-> {diag} (k_hat={k_hat:.2f}, {reliable})"
            )
        logger.info("\n%s", "\n".join(lines))


def power_scaling_sensitivity(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    result: InferenceResult,
    seed: int = 0,
    alpha_delta: float = 0.01,
) -> PowerScalingResult:
    """Post-fit power-scaling sensitivity diagnostic.

    Detects whether posterior is driven by prior or likelihood by
    perturbing each component's contribution and measuring the
    resulting shift in posterior means.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) real observed data
        times: (T,) observation times
        result: InferenceResult from fitting
        seed: random seed
        alpha_delta: perturbation size for power scaling (default 0.01)

    Returns:
        PowerScalingResult with per-parameter sensitivity diagnostics
    """
    rng_key = random.PRNGKey(seed)

    # 1. Discover sites and build eval functions
    backend = model.make_likelihood_backend()
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, trace_key, backend)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    _, unravel_fn = ravel_pytree(example_unc)

    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, site_info, unravel_fn, backend
    )

    param_names = sorted(site_info.keys())

    # 2. Extract posterior samples -> unconstrained flat vectors
    samples = result.get_samples()
    n_samples = next(iter(samples.values())).shape[0]

    flat_samples = []
    for i in range(n_samples):
        parts = []
        for name in param_names:
            if name in samples:
                con_val = samples[name][i]
                unc_val = site_info[name]["transform"].inv(con_val)
                parts.append(unc_val.reshape(-1))
        if parts:
            flat_samples.append(jnp.concatenate(parts))

    if not flat_samples:
        return PowerScalingResult(
            prior_sensitivity={},
            likelihood_sensitivity={},
            diagnosis={},
        )

    z_samples = jnp.stack(flat_samples)  # (n_samples, D)

    # 3. Evaluate log-prior and log-likelihood for each sample
    batch_log_lik = jax.vmap(log_lik_fn)
    batch_log_prior = jax.vmap(log_prior_unc_fn)

    # Chunk to avoid OOM
    chunk_size = 32
    log_liks_parts = []
    log_priors_parts = []
    for start in range(0, n_samples, chunk_size):
        chunk = z_samples[start : start + chunk_size]
        log_liks_parts.append(batch_log_lik(chunk))
        log_priors_parts.append(batch_log_prior(chunk))

    log_liks = jnp.concatenate(log_liks_parts)
    log_priors = jnp.concatenate(log_priors_parts)

    # 4. Power-scaling: compute weighted means under perturbed distributions
    alpha = alpha_delta

    # Prior perturbation weights: w_i propto exp(alpha * log_prior_i)
    prior_log_weights = alpha * log_priors
    prior_log_weights = prior_log_weights - jax.nn.logsumexp(prior_log_weights)
    prior_weights = jnp.exp(prior_log_weights)

    # Likelihood perturbation weights: w_i propto exp(alpha * log_lik_i)
    lik_log_weights = alpha * log_liks
    lik_log_weights = lik_log_weights - jax.nn.logsumexp(lik_log_weights)
    lik_weights = jnp.exp(lik_log_weights)

    # 5. Compute per-parameter sensitivity
    prior_sensitivity = {}
    likelihood_sensitivity = {}
    diagnosis = {}
    psis_k_hat = {}

    # Extract per-parameter values from flat samples
    offset = 0
    for name in param_names:
        if name not in samples:
            continue
        size = int(jnp.prod(jnp.array(site_info[name]["shape"])))
        param_vals = z_samples[:, offset : offset + size]  # (n_samples, size)
        param_mean = jnp.mean(param_vals, axis=0)

        # Weighted means under perturbation
        prior_weighted_mean = jnp.sum(prior_weights[:, None] * param_vals, axis=0)
        lik_weighted_mean = jnp.sum(lik_weights[:, None] * param_vals, axis=0)

        # Sensitivity = ||shift|| / alpha_delta
        prior_shift = float(jnp.mean(jnp.abs(prior_weighted_mean - param_mean))) / alpha_delta
        lik_shift = float(jnp.mean(jnp.abs(lik_weighted_mean - param_mean))) / alpha_delta

        prior_sensitivity[name] = prior_shift
        likelihood_sensitivity[name] = lik_shift

        # k-hat from prior weights via PSIS
        import arviz as az
        import numpy as np

        _, kss = az.psislw(np.asarray(prior_log_weights))
        psis_k_hat[name] = float(kss)

        # Diagnosis
        if prior_shift > 0.05 and lik_shift < 0.05:
            diagnosis[name] = "prior_dominated"
        elif prior_shift > 0.05 and lik_shift > 0.05:
            diagnosis[name] = "prior_data_conflict"
        else:
            diagnosis[name] = "well_identified"

        offset += size

    return PowerScalingResult(
        prior_sensitivity=prior_sensitivity,
        likelihood_sensitivity=likelihood_sensitivity,
        diagnosis=diagnosis,
        psis_k_hat=psis_k_hat,
    )
