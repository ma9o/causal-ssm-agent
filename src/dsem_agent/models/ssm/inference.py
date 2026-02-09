"""Inference backends for SSM models.

Separates inference from model definition. SSMModel defines the probabilistic
model; this module provides fit() to run inference with different backends:

- SVI (default): Fast approximate posterior via ELBO optimization.
  Tolerates PF gradient noise because SGD is designed for noisy gradients.
- PMMH: Exact posterior via gradient-free MH with PF as unbiased likelihood
  estimator. Slow but correct.
- NUTS: HMC-based sampling. Works well with Kalman likelihood but struggles
  with PF resampling discontinuities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoNormal
from numpyro.optim import ClippedAdam

if TYPE_CHECKING:
    from dsem_agent.models.ssm.model import SSMModel


@dataclass
class InferenceResult:
    """Container for inference results across all backends.

    Provides a uniform interface regardless of whether SVI, PMMH, or NUTS
    was used for inference.
    """

    _samples: dict[str, jnp.ndarray]  # name -> (n_draws, *shape)
    method: Literal["nuts", "svi", "pmmh", "hessmc2"]
    diagnostics: dict = field(default_factory=dict)

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Return posterior samples dict."""
        return self._samples

    def print_summary(self) -> None:
        """Print summary statistics for posterior samples."""
        print(f"\nInference method: {self.method}")
        print(f"{'Parameter':<30} {'Mean':>10} {'Std':>10} {'5%':>10} {'95%':>10}")
        print("-" * 72)
        for name, values in self._samples.items():
            if values.ndim == 1:
                mean = float(jnp.mean(values))
                std = float(jnp.std(values))
                q5 = float(jnp.percentile(values, 5))
                q95 = float(jnp.percentile(values, 95))
                print(f"{name:<30} {mean:>10.4f} {std:>10.4f} {q5:>10.4f} {q95:>10.4f}")
            elif values.ndim >= 2:
                # Flatten parameter dimensions for summary
                flat = values.reshape(values.shape[0], -1)
                for i in range(flat.shape[1]):
                    label = f"{name}[{i}]"
                    mean = float(jnp.mean(flat[:, i]))
                    std = float(jnp.std(flat[:, i]))
                    q5 = float(jnp.percentile(flat[:, i], 5))
                    q95 = float(jnp.percentile(flat[:, i], 95))
                    print(f"{label:<30} {mean:>10.4f} {std:>10.4f} {q5:>10.4f} {q95:>10.4f}")


def fit(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    method: Literal["svi", "pmmh", "nuts", "hessmc2"] = "svi",
    **kwargs: Any,
) -> InferenceResult:
    """Fit an SSM using the specified inference method.

    Args:
        model: SSMModel instance defining the probabilistic model
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        subject_ids: (N,) subject indices (0-indexed, for hierarchical)
        method: Inference method - "svi" (default), "pmmh", or "nuts"
        **kwargs: Method-specific arguments

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    if method == "nuts":
        return _fit_nuts(model, observations, times, subject_ids, **kwargs)
    elif method == "svi":
        return _fit_svi(model, observations, times, subject_ids, **kwargs)
    elif method == "pmmh":
        return _fit_pmmh(model, observations, times, subject_ids, **kwargs)
    elif method == "hessmc2":
        from dsem_agent.models.ssm.hessmc2 import fit_hessmc2

        return fit_hessmc2(model, observations, times, subject_ids, **kwargs)
    else:
        raise ValueError(
            f"Unknown inference method: {method!r}. Use 'svi', 'pmmh', 'nuts', or 'hessmc2'."
        )


def prior_predictive(
    model: SSMModel,
    times: jnp.ndarray,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample from the prior predictive distribution.

    Uses handlers.block to skip the PF likelihood computation,
    which is unnecessary and expensive for prior sampling.

    Args:
        model: SSMModel instance
        times: (T,) time points
        num_samples: Number of prior samples
        seed: Random seed

    Returns:
        Dict of prior predictive samples
    """
    rng_key = random.PRNGKey(seed)
    blocked_model = handlers.block(model.model, hide=["log_likelihood"])
    predictive = Predictive(blocked_model, num_samples=num_samples)
    dummy_obs = jnp.zeros((len(times), model.spec.n_manifest))
    return predictive(rng_key, dummy_obs, times)


def _fit_nuts(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    dense_mass: bool = False,
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 8,
    **kwargs: Any,
) -> InferenceResult:
    """Fit using NUTS (HMC).

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        subject_ids: (N,) subject indices
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        seed: Random seed
        dense_mass: Use dense mass matrix
        target_accept_prob: Target acceptance probability
        max_tree_depth: Max tree depth
        **kwargs: Additional MCMC arguments

    Returns:
        InferenceResult with NUTS samples
    """
    kernel = NUTS(
        model.model,
        init_strategy=init_to_median(num_samples=15),
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

    rng_key = random.PRNGKey(seed)
    mcmc.run(rng_key, observations, times, subject_ids)

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="nuts",
        diagnostics={"mcmc": mcmc},
    )


def _fit_svi(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    guide_type: str = "mvn",
    num_steps: int = 5000,
    num_samples: int = 1000,
    learning_rate: float = 0.01,
    seed: int = 0,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit using Stochastic Variational Inference.

    Uses AutoGuide to learn an approximate posterior. numpyro.factor() sites
    are handled automatically - the guide only models latent sample sites.

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        subject_ids: (N,) subject indices
        guide_type: Guide family - "normal", "mvn", or "delta"
        num_steps: Number of SVI optimization steps
        num_samples: Number of posterior samples to draw after fitting
        learning_rate: Adam learning rate
        seed: Random seed
        **kwargs: Ignored

    Returns:
        InferenceResult with approximate posterior samples
    """
    guide_cls = {
        "normal": AutoNormal,
        "mvn": AutoMultivariateNormal,
        "delta": AutoDelta,
    }[guide_type]
    guide = guide_cls(model.model)

    optimizer = ClippedAdam(step_size=learning_rate)
    svi = SVI(model.model, guide, optimizer, Trace_ELBO())

    rng_key = random.PRNGKey(seed)
    svi_result = svi.run(rng_key, num_steps, observations, times, subject_ids)

    # Draw posterior samples from the fitted guide
    sample_key = random.PRNGKey(seed + 1)
    predictive = Predictive(
        model.model,
        guide=guide,
        params=svi_result.params,
        num_samples=num_samples,
    )
    raw_samples = predictive(sample_key, observations, times, subject_ids)

    # Filter out the log_likelihood factor site (observed)
    samples = {name: values for name, values in raw_samples.items() if name != "log_likelihood"}

    return InferenceResult(
        _samples=samples,
        method="svi",
        diagnostics={"losses": svi_result.losses, "params": svi_result.params},
    )


def _eval_model(
    model_fn,
    params_dict: dict[str, jnp.ndarray],
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None,
) -> tuple[float, float]:
    """Evaluate model with substituted params. Returns (log_joint,).

    Uses numpyro.handlers to substitute parameter values and trace the model,
    computing log_prior + log_likelihood without any code duplication.

    Args:
        model_fn: NumPyro model function
        params_dict: Parameter values to substitute
        observations: Observed data
        times: Time points
        subject_ids: Subject indices

    Returns:
        Tuple of (log_likelihood, log_prior)
    """
    with handlers.seed(rng_seed=0), handlers.substitute(data=params_dict):
        trace = handlers.trace(model_fn).get_trace(observations, times, subject_ids)

    log_lik = 0.0
    log_prior = 0.0
    for name, site in trace.items():
        if site["type"] == "sample":
            if name == "log_likelihood":
                # Factor site: fn is Unit with log_factor attribute
                log_lik = site["fn"].log_factor
            elif not site.get("is_observed", False):
                log_prior = log_prior + jnp.sum(site["fn"].log_prob(site["value"]))

    return log_lik, log_prior


def _fit_pmmh(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    num_warmup: int = 500,
    num_samples: int = 1000,
    seed: int = 0,
    proposal_scale: float = 0.01,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit using Particle Marginal Metropolis-Hastings.

    Gradient-free MH in unconstrained space with PF as unbiased likelihood
    estimator. Essential: uses a fresh PF random key per step for
    pseudo-marginal correctness (Andrieu & Roberts 2009).

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        subject_ids: (N,) subject indices
        num_warmup: Number of warmup steps (for proposal adaptation)
        num_samples: Number of posterior samples
        seed: Random seed
        proposal_scale: Initial proposal standard deviation
        **kwargs: Ignored

    Returns:
        InferenceResult with PMMH samples
    """
    rng_key = random.PRNGKey(seed)

    # 1. Trace model once to discover sample sites (names, shapes, distributions)
    rng_key, trace_key = random.split(rng_key)
    with handlers.seed(rng_seed=int(trace_key[0])):
        trace = handlers.trace(model.model).get_trace(observations, times, subject_ids)

    site_info = {}  # name -> (shape, distribution, transform)
    for name, site in trace.items():
        if (
            site["type"] == "sample"
            and not site.get("is_observed", False)
            and name != "log_likelihood"
        ):
            d = site["fn"]
            transform = dist.transforms.biject_to(d.support)
            site_info[name] = {
                "shape": site["value"].shape,
                "distribution": d,
                "transform": transform,
                "value": site["value"],
            }

    # 2. Initialize: pack current values into unconstrained space
    unconstrained = {}
    constrained = {}
    for name, info in site_info.items():
        constrained[name] = info["value"]
        unconstrained[name] = info["transform"].inv(info["value"])

    # Compute initial log-joint
    rng_key, pf_key = random.split(rng_key)
    model.pf_key = pf_key
    log_lik, log_prior = _eval_model(model.model, constrained, observations, times, subject_ids)

    # Add Jacobian correction for unconstrained -> constrained
    log_jacobian = 0.0
    for name, info in site_info.items():
        log_jacobian = log_jacobian + jnp.sum(
            info["transform"].log_abs_det_jacobian(unconstrained[name], constrained[name])
        )
    current_log_joint = log_lik + log_prior + log_jacobian

    # 3. MH loop
    total_steps = num_warmup + num_samples
    samples_list = []
    n_accepted = 0
    scale = proposal_scale

    for step in range(total_steps):
        rng_key, prop_key, pf_key, accept_key = random.split(rng_key, 4)

        # a. Propose in unconstrained space
        proposed_unconstrained = {}
        proposed_constrained = {}
        for name, info in site_info.items():
            noise = random.normal(prop_key, info["shape"]) * scale
            prop_key, _ = random.split(prop_key)
            proposed_unconstrained[name] = unconstrained[name] + noise
            proposed_constrained[name] = info["transform"](proposed_unconstrained[name])

        # b. Fresh PF key for pseudo-marginal correctness
        model.pf_key = pf_key

        # c. Evaluate proposed parameters
        prop_log_lik, prop_log_prior = _eval_model(
            model.model, proposed_constrained, observations, times, subject_ids
        )
        prop_log_jacobian = 0.0
        for name, info in site_info.items():
            prop_log_jacobian = prop_log_jacobian + jnp.sum(
                info["transform"].log_abs_det_jacobian(
                    proposed_unconstrained[name], proposed_constrained[name]
                )
            )
        proposed_log_joint = prop_log_lik + prop_log_prior + prop_log_jacobian

        # d. MH accept/reject
        log_alpha = proposed_log_joint - current_log_joint
        u = random.uniform(accept_key)
        accept = jnp.log(u) < log_alpha

        if bool(accept) and bool(jnp.isfinite(proposed_log_joint)):
            unconstrained = proposed_unconstrained
            constrained = proposed_constrained
            current_log_joint = proposed_log_joint
            n_accepted += 1

        # e. Adapt proposal scale during warmup
        if step < num_warmup and step > 0 and step % 50 == 0:
            acceptance_rate = n_accepted / (step + 1)
            if acceptance_rate < 0.15:
                scale *= 0.5
            elif acceptance_rate > 0.35:
                scale *= 1.5

        # f. Store post-warmup samples
        if step >= num_warmup:
            samples_list.append({name: val.copy() for name, val in constrained.items()})

    # 4. Stack samples
    stacked_samples = {}
    if samples_list:
        for name in samples_list[0]:
            stacked_samples[name] = jnp.stack([s[name] for s in samples_list])

    acceptance_rate = n_accepted / total_steps

    return InferenceResult(
        _samples=stacked_samples,
        method="pmmh",
        diagnostics={
            "acceptance_rate": acceptance_rate,
            "final_proposal_scale": scale,
        },
    )
