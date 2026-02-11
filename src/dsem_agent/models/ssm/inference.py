"""Inference backends for SSM models.

Separates inference from model definition. SSMModel defines the probabilistic
model; this module provides fit() to run inference with different backends:

- SVI (default): Fast approximate posterior via ELBO optimization.
  Tolerates PF gradient noise because SGD is designed for noisy gradients.
- NUTS: HMC-based sampling. Works well with Kalman likelihood but struggles
  with PF resampling discontinuities.
- Hess-MC²: SMC with gradient-based change-of-variables L-kernels.
- PGAS: Particle Gibbs with ancestor sampling + gradient-informed proposals.
- Tempered SMC: Adaptive tempering with preconditioned HMC/MALA mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import jax.random as random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoNormal
from numpyro.optim import ClippedAdam

if TYPE_CHECKING:
    from dsem_agent.models.ssm.model import SSMModel


@dataclass
class InferenceResult:
    """Container for inference results across all backends.

    Provides a uniform interface regardless of which backend was used.
    """

    _samples: dict[str, jnp.ndarray]  # name -> (n_draws, *shape)
    method: Literal["nuts", "svi", "hessmc2", "pgas", "tempered_smc", "laplace_em", "structured_vi", "dpf"]
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
    method: Literal["svi", "nuts", "hessmc2", "pgas", "tempered_smc", "laplace_em", "structured_vi", "dpf"] = "svi",
    **kwargs: Any,
) -> InferenceResult:
    """Fit an SSM using the specified inference method.

    Args:
        model: SSMModel instance defining the probabilistic model
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        subject_ids: (N,) subject indices (0-indexed, for hierarchical)
        method: Inference method - "svi" (default), "nuts", "hessmc2", "pgas", "tempered_smc"
        **kwargs: Method-specific arguments

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    if method == "nuts":
        return _fit_nuts(model, observations, times, subject_ids, **kwargs)
    elif method == "svi":
        return _fit_svi(model, observations, times, subject_ids, **kwargs)
    elif method == "hessmc2":
        from dsem_agent.models.ssm.hessmc2 import fit_hessmc2

        return fit_hessmc2(model, observations, times, subject_ids, **kwargs)
    elif method == "pgas":
        from dsem_agent.models.ssm.pgas import fit_pgas

        return fit_pgas(model, observations, times, subject_ids, **kwargs)
    elif method == "tempered_smc":
        from dsem_agent.models.ssm.tempered_smc import fit_tempered_smc

        return fit_tempered_smc(model, observations, times, subject_ids, **kwargs)
    elif method == "laplace_em":
        from dsem_agent.models.ssm.laplace_em import fit_laplace_em

        return fit_laplace_em(model, observations, times, subject_ids, **kwargs)
    elif method == "structured_vi":
        from dsem_agent.models.ssm.structured_vi import fit_structured_vi

        return fit_structured_vi(model, observations, times, subject_ids, **kwargs)
    elif method == "dpf":
        from dsem_agent.models.ssm.dpf import fit_dpf

        return fit_dpf(model, observations, times, subject_ids, **kwargs)
    else:
        raise ValueError(
            f"Unknown inference method: {method!r}. "
            "Use 'svi', 'nuts', 'hessmc2', 'pgas', 'tempered_smc', "
            "'laplace_em', 'structured_vi', or 'dpf'."
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
