"""Posterior MCMC backends for SSM models."""

from __future__ import annotations

import functools
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import blackjax
import blackjax.vi.pathfinder as pathfinder
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.inference.methods.map import _build_laplace_em_bundle
from causal_ssm_agent.models.ssm.inference.shared import (
    _apply_reparam,
    _filter_public_samples,
    _trace_public_sites,
)
from causal_ssm_agent.models.ssm.inference.types import InferenceResult
from causal_ssm_agent.models.ssm.inference.utils import _discover_sites, extract_constrained_samples

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMModel


logger = get_prefect_logger(__name__)

_PATHFINDER_NUM_ELBO_SAMPLES = 50
_PATHFINDER_MAXITER = 50
_CHEES_INITIAL_STEP_SIZE = 1e-1
_CHEES_TRAJECTORY_OPT_LR = 1e-2


def _emit_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[blackjax_chees_hmc] {message}", file=sys.stderr, flush=True)


def _block_until_ready_tree(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


@dataclass(frozen=True)
class _BlackJaxMCMCResult:
    """Minimal MCMC-compatible wrapper for BlackJAX sampler outputs."""

    chain_samples: dict[str, jnp.ndarray]
    chain_extra_fields: dict[str, jnp.ndarray]
    num_chains: int
    num_samples: int
    backend: str = "blackjax_chees_hmc"

    def get_samples(self, group_by_chain: bool = False) -> dict[str, jnp.ndarray]:
        if group_by_chain:
            return self.chain_samples
        return {
            name: values.reshape((self.num_chains * self.num_samples, *values.shape[2:]))
            for name, values in self.chain_samples.items()
        }

    def get_extra_fields(self, group_by_chain: bool = False) -> dict[str, jnp.ndarray]:
        if group_by_chain:
            return self.chain_extra_fields
        return {
            name: values.reshape((self.num_chains * self.num_samples, *values.shape[2:]))
            for name, values in self.chain_extra_fields.items()
        }


def _run_pathfinder_approximation(
    log_posterior_fn,
    flat_example: jnp.ndarray,
    *,
    pathfinder_key: jnp.ndarray,
    pathfinder_num_elbo_samples: int,
    pathfinder_maxiter: int,
) -> tuple[Any, dict[str, Any]]:
    """Run Pathfinder and return the fitted approximation with diagnostics."""
    pathfinder_start = time.perf_counter()
    state, _ = pathfinder.approximate(
        pathfinder_key,
        log_posterior_fn,
        flat_example,
        num_samples=pathfinder_num_elbo_samples,
        maxiter=pathfinder_maxiter,
    )
    _block_until_ready_tree((state.position, state.elbo))
    pathfinder_seconds = time.perf_counter() - pathfinder_start
    position = jnp.asarray(state.position, dtype=flat_example.dtype)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(position)))):
        raise RuntimeError("Pathfinder returned a non-finite initialization vector")
    if not bool(jax.device_get(jnp.isfinite(state.elbo))):
        raise RuntimeError("Pathfinder returned a non-finite ELBO")
    return state, {
        "init_method": "pathfinder",
        "pathfinder_elbo": float(jax.device_get(state.elbo)),
        "pathfinder_seconds": pathfinder_seconds,
    }


def _build_pathfinder_init_strategy(
    site_info: dict[str, Any],
    unravel_fn,
    pathfinder_state,
) -> Any:
    """Build a constrained NumPyro init strategy from a Pathfinder mode."""
    unconstrained = unravel_fn(jnp.asarray(pathfinder_state.position))
    init_values = {name: site_info[name]["transform"](unconstrained[name]) for name in site_info}
    return init_to_value(values=init_values)


def _sample_pathfinder_positions(
    pathfinder_state,
    *,
    rng_key: jnp.ndarray,
    num_chains: int,
    dtype,
) -> jnp.ndarray:
    """Draw initial unconstrained positions for all chains from Pathfinder."""
    positions, _log_q = pathfinder.sample(rng_key, pathfinder_state, num_samples=num_chains)
    positions = jnp.asarray(positions, dtype=dtype)
    if positions.ndim != 2:
        raise ValueError(
            f"Expected Pathfinder chain initials with rank 2, received shape {positions.shape}."
        )
    if not bool(jax.device_get(jnp.all(jnp.isfinite(positions)))):
        raise RuntimeError("Pathfinder returned non-finite chain initial positions")
    return positions


def _run_blackjax_chees_hmc(
    log_posterior_fn,
    *,
    init_positions: jnp.ndarray,
    rng_key: jnp.ndarray,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    target_accept_prob: float,
    max_tree_depth: int,
    progress_bar: bool,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, Any]]:
    """Warm up and sample the unconstrained posterior with BlackJAX ChEES-HMC."""
    warmup_key, sample_key = random.split(rng_key)
    warmup = blackjax.chees_adaptation(
        log_posterior_fn,
        num_chains,
        target_acceptance_rate=target_accept_prob,
        max_leapfrog_steps=2**max_tree_depth,
        adaptation_info_fn=blackjax.adaptation.base.get_filter_adapt_info_fn(),
    )
    _emit_progress(
        progress_bar,
        (
            "starting warmup "
            f"(num_warmup={num_warmup}, num_chains={num_chains}, "
            f"max_leapfrog_steps={2**max_tree_depth})"
        ),
    )
    warmup_start = time.perf_counter()
    warmup_result, _warmup_info = warmup.run(
        warmup_key,
        init_positions,
        _CHEES_INITIAL_STEP_SIZE,
        optax.adam(_CHEES_TRAJECTORY_OPT_LR),
        num_warmup,
    )
    _block_until_ready_tree(
        (
            warmup_result.state.position,
            warmup_result.parameters["step_size"],
        )
    )
    warmup_seconds = time.perf_counter() - warmup_start
    _emit_progress(
        progress_bar,
        f"warmup complete in {warmup_seconds:.1f}s; starting posterior sampling",
    )
    sampler = blackjax.dynamic_hmc(log_posterior_fn, **warmup_result.parameters)
    step_fn = jax.vmap(sampler.step)

    def _one_step(state, key):
        chain_keys = random.split(key, num_chains)
        new_state, info = step_fn(chain_keys, state)
        return new_state, {
            "position": new_state.position,
            "lp": new_state.logdensity,
            "accept_prob": info.acceptance_rate,
            "diverging": info.is_divergent,
            "energy": info.energy,
            "num_steps": info.num_integration_steps,
        }

    sample_keys = random.split(sample_key, num_samples)
    sampling_start = time.perf_counter()
    _last_state, history = jax.lax.scan(_one_step, warmup_result.state, sample_keys)
    grouped_positions = jnp.swapaxes(history["position"], 0, 1)
    grouped_extra = {
        name: jnp.swapaxes(values, 0, 1) for name, values in history.items() if name != "position"
    }
    _block_until_ready_tree((grouped_positions, grouped_extra))
    posterior_sampling_seconds = time.perf_counter() - sampling_start
    _emit_progress(
        progress_bar,
        (
            "posterior sampling complete "
            f"({num_samples} draws per chain) in {posterior_sampling_seconds:.1f}s"
        ),
    )
    return (
        grouped_positions,
        grouped_extra,
        {
            "warmup_method": "blackjax_chees_hmc",
            "warmup_initial_step_size": _CHEES_INITIAL_STEP_SIZE,
            "warmup_max_leapfrog_steps": int(2**max_tree_depth),
            "warmup_trajectory_opt_lr": _CHEES_TRAJECTORY_OPT_LR,
            "sampler_backend": "blackjax_chees_hmc",
            "warmup_seconds": warmup_seconds,
            "posterior_sampling_seconds": posterior_sampling_seconds,
        },
    )


def _fit_numpyro_nuts(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    dense_mass: bool,
    target_accept_prob: float,
    max_tree_depth: int,
    n_ieks_iters: int,
    pathfinder_num_elbo_samples: int,
    pathfinder_maxiter: int,
    reparam,
    **kwargs: Any,
) -> InferenceResult:
    """Run the existing NumPyro NUTS path."""
    rng_key = random.PRNGKey(seed)
    if model.likelihood == "kalman":
        backend = model.make_likelihood_backend()
        init_strategy = init_to_median(num_samples=15)
        init_diagnostics: dict[str, Any] = {"init_method": "median"}
        mcmc_key = rng_key
    else:
        backend = model.make_laplace_backend(n_ieks_iters)
        trace_key, pathfinder_key, mcmc_key = random.split(rng_key, 3)
        bundle = _build_laplace_em_bundle(
            model,
            observations,
            times,
            trace_key,
            backend,
            reparam,
        )
        pathfinder_state, init_diagnostics = _run_pathfinder_approximation(
            bundle["log_posterior_fn"],
            bundle["flat_example"],
            pathfinder_key=pathfinder_key,
            pathfinder_num_elbo_samples=pathfinder_num_elbo_samples,
            pathfinder_maxiter=pathfinder_maxiter,
        )
        init_strategy = _build_pathfinder_init_strategy(
            bundle["site_info"],
            bundle["unravel_fn"],
            pathfinder_state,
        )

    base_model_fn = functools.partial(model.model, likelihood_backend=backend)
    public_sites = _trace_public_sites(base_model_fn, observations, times)
    model_fn = _apply_reparam(base_model_fn, reparam)
    kernel = NUTS(
        model_fn,
        init_strategy=init_strategy,
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
        chain_method="vectorized",
        jit_model_args=False,
        **kwargs,
    )
    mcmc.run(
        mcmc_key,
        observations,
        times,
        extra_fields=("diverging", "num_steps", "accept_prob", "energy"),
    )
    return InferenceResult(
        _samples=_filter_public_samples(mcmc.get_samples(), public_sites),
        method="nuts",
        diagnostics={
            "mcmc": mcmc,
            "public_sites": sorted(public_sites),
            **init_diagnostics,
        },
    )


def _fit_blackjax_chees_hmc(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    dense_mass: bool,
    target_accept_prob: float,
    max_tree_depth: int,
    n_ieks_iters: int,
    pathfinder_num_elbo_samples: int,
    pathfinder_maxiter: int,
    reparam,
    **kwargs: Any,
) -> InferenceResult:
    """Run the IEKS-backed parameter posterior sampler with BlackJAX ChEES-HMC."""
    progress_bar = kwargs.pop("progress_bar", False)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(
            f"Unsupported fit_nuts kwargs for the BlackJAX ChEES-HMC path: {unsupported}"
        )
    if dense_mass:
        logger.info(
            "fit_nuts non-kalman path: dense_mass requested but ignored by BlackJAX "
            "ChEES-HMC, which uses a unit diagonal inverse mass matrix"
        )

    fit_start = time.perf_counter()
    timings: dict[str, float] = {}
    rng_key = random.PRNGKey(seed)
    trace_key, public_trace_key, pathfinder_key, pathfinder_sample_key, mcmc_key = random.split(
        rng_key, 5
    )
    _emit_progress(progress_bar, "building IEKS/Laplace posterior bundle")
    backend = model.make_laplace_backend(n_ieks_iters)
    bundle_start = time.perf_counter()
    bundle = _build_laplace_em_bundle(
        model,
        observations,
        times,
        trace_key,
        backend,
        reparam,
    )
    _block_until_ready_tree(bundle["flat_example"])
    timings["bundle_build_seconds"] = time.perf_counter() - bundle_start
    _emit_progress(progress_bar, f"bundle ready in {timings['bundle_build_seconds']:.1f}s")
    _emit_progress(
        progress_bar,
        (
            "running Pathfinder "
            f"(num_elbo_samples={pathfinder_num_elbo_samples}, maxiter={pathfinder_maxiter})"
        ),
    )
    pathfinder_state, init_diagnostics = _run_pathfinder_approximation(
        bundle["log_posterior_fn"],
        bundle["flat_example"],
        pathfinder_key=pathfinder_key,
        pathfinder_num_elbo_samples=pathfinder_num_elbo_samples,
        pathfinder_maxiter=pathfinder_maxiter,
    )
    pathfinder_seconds = init_diagnostics.get("pathfinder_seconds")
    _emit_progress(
        progress_bar,
        (
            f"pathfinder complete (elbo={init_diagnostics['pathfinder_elbo']:.3f})"
            if pathfinder_seconds is None
            else (
                "pathfinder complete "
                f"(elbo={init_diagnostics['pathfinder_elbo']:.3f}, "
                f"{pathfinder_seconds:.1f}s)"
            )
        ),
    )
    _emit_progress(progress_bar, f"drawing {num_chains} initial chain positions")
    init_sample_start = time.perf_counter()
    init_positions = _sample_pathfinder_positions(
        pathfinder_state,
        rng_key=pathfinder_sample_key,
        num_chains=num_chains,
        dtype=bundle["flat_example"].dtype,
    )
    _block_until_ready_tree(init_positions)
    timings["pathfinder_chain_init_seconds"] = time.perf_counter() - init_sample_start
    _emit_progress(
        progress_bar,
        f"initial chain positions ready in {timings['pathfinder_chain_init_seconds']:.1f}s",
    )
    grouped_particles, grouped_extra, sampler_diagnostics = _run_blackjax_chees_hmc(
        bundle["log_posterior_fn"],
        init_positions=init_positions,
        rng_key=mcmc_key,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        progress_bar=progress_bar,
    )
    if pathfinder_seconds is not None:
        timings["pathfinder_seconds"] = pathfinder_seconds
    warmup_seconds = sampler_diagnostics.get("warmup_seconds")
    if warmup_seconds is not None:
        timings["warmup_seconds"] = warmup_seconds
    posterior_sampling_seconds = sampler_diagnostics.get("posterior_sampling_seconds")
    if posterior_sampling_seconds is not None:
        timings["posterior_sampling_seconds"] = posterior_sampling_seconds

    flat_particles = grouped_particles.reshape((num_chains * num_samples, bundle["dim"]))
    _emit_progress(progress_bar, "extracting constrained posterior samples")
    postprocess_start = time.perf_counter()
    site_discovery_start = time.perf_counter()
    original_site_info = _discover_sites(
        model,
        observations,
        times,
        public_trace_key,
        backend,
        reparam=None,
    )
    timings["site_discovery_seconds"] = time.perf_counter() - site_discovery_start
    _emit_progress(
        progress_bar,
        f"original site discovery complete in {timings['site_discovery_seconds']:.1f}s",
    )
    extraction_timings: dict[str, float] = {}
    constrained_samples = extract_constrained_samples(
        flat_particles,
        bundle["site_info"],
        bundle["unravel_fn"],
        model.spec,
        reparam=reparam,
        model=model,
        observations=observations,
        times=times,
        profiling=extraction_timings,
    )
    timings.update(extraction_timings)
    filter_start = time.perf_counter()
    constrained_samples = {
        name: constrained_samples[name]
        for name in original_site_info
        if name in constrained_samples
    }
    timings["sample_site_filter_seconds"] = time.perf_counter() - filter_start
    regroup_start = time.perf_counter()
    grouped_samples = {
        name: values.reshape((num_chains, num_samples, *values.shape[1:]))
        for name, values in constrained_samples.items()
    }
    _block_until_ready_tree(grouped_samples)
    timings["grouped_sample_pack_seconds"] = time.perf_counter() - regroup_start
    public_trace_start = time.perf_counter()
    public_sites = _trace_public_sites(
        functools.partial(model.model, likelihood_backend=backend),
        observations,
        times,
    )
    timings["public_trace_seconds"] = time.perf_counter() - public_trace_start
    timings["postprocessing_seconds"] = time.perf_counter() - postprocess_start
    mcmc = _BlackJaxMCMCResult(
        chain_samples=grouped_samples,
        chain_extra_fields=grouped_extra,
        num_chains=num_chains,
        num_samples=num_samples,
    )
    timings["total_blackjax_fit_seconds"] = time.perf_counter() - fit_start
    _emit_progress(
        progress_bar,
        (
            "posterior sample extraction complete "
            f"(postprocessing {timings['postprocessing_seconds']:.1f}s, "
            f"total fit {timings['total_blackjax_fit_seconds']:.1f}s)"
        ),
    )
    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="nuts",
        diagnostics={
            "mcmc": mcmc,
            "public_sites": sorted(public_sites),
            "dense_mass_requested": bool(dense_mass),
            "dense_mass_used": False,
            **init_diagnostics,
            **sampler_diagnostics,
            "timings": timings,
        },
    )


def fit_nuts(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    dense_mass: bool = False,
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 8,
    n_ieks_iters: int = 5,
    pathfinder_num_elbo_samples: int = _PATHFINDER_NUM_ELBO_SAMPLES,
    pathfinder_maxiter: int = _PATHFINDER_MAXITER,
    reparam=None,
    **kwargs: Any,
) -> InferenceResult:
    """Fit using posterior MCMC.

    For Kalman-eligible models this uses NumPyro NUTS on the exact marginal
    likelihood. For IEKS/Laplace-backed models it samples the unconstrained
    parameter posterior with BlackJAX ChEES-HMC using the Laplace log posterior
    from the IEKS inner solver.
    """
    if model.likelihood == "kalman":
        return _fit_numpyro_nuts(
            model,
            observations,
            times,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
            dense_mass=dense_mass,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            n_ieks_iters=n_ieks_iters,
            pathfinder_num_elbo_samples=pathfinder_num_elbo_samples,
            pathfinder_maxiter=pathfinder_maxiter,
            reparam=reparam,
            **kwargs,
        )
    return _fit_blackjax_chees_hmc(
        model,
        observations,
        times,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        dense_mass=dense_mass,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        n_ieks_iters=n_ieks_iters,
        pathfinder_num_elbo_samples=pathfinder_num_elbo_samples,
        pathfinder_maxiter=pathfinder_maxiter,
        reparam=reparam,
        **kwargs,
    )
