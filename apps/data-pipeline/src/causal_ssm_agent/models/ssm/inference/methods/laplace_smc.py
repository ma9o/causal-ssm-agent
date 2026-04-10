"""SMC-backed Laplace parameter inference.

This preserves the repo's historical behavior: use the IEKS/Laplace
approximation as the inner likelihood target, then sample parameters with the
shared tempered SMC engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.inference.engines.tempered_smc import run_tempered_smc

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.inference.types import InferenceResult

logger = get_prefect_logger(__name__)


def fit_laplace_smc(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    target_accept: float | None = None,
    seed: int = 0,
    n_ieks_iters: int = 5,
    n_leapfrog: int = 5,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    reparam=None,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters with the historical Laplace inner objective + SMC."""
    backend_label = "kalman" if model.likelihood == "kalman" else "laplace_ieks"
    logger.info(
        "Laplace-SMC config: backend=%s n_outer=%s n_particles=%s n_mh=%s "
        "n_leapfrog=%s n_ieks_iters=%s adaptive_tempering=%s target_ess_ratio=%.2f "
        "waste_free=%s n_warmup=%s",
        backend_label,
        n_outer,
        n_csmc_particles,
        n_mh_steps,
        n_leapfrog,
        n_ieks_iters,
        adaptive_tempering,
        target_ess_ratio,
        waste_free,
        n_warmup if n_warmup is not None else 5,
    )

    with jax.profiler.TraceAnnotation("laplace_smc/build_likelihood_backend"):
        if model.likelihood == "kalman":
            backend = model.make_likelihood_backend()
        else:
            backend = model.make_laplace_backend(n_ieks_iters)

    with jax.profiler.TraceAnnotation("laplace_smc/run_tempered_smc"):
        return run_tempered_smc(
            model,
            observations,
            times,
            n_outer=n_outer,
            n_csmc_particles=n_csmc_particles,
            n_mh_steps=n_mh_steps,
            param_step_size=param_step_size,
            n_warmup=n_warmup,
            target_accept=target_accept,
            seed=seed,
            adaptive_tempering=adaptive_tempering,
            target_ess_ratio=target_ess_ratio,
            waste_free=waste_free,
            n_leapfrog=n_leapfrog,
            method_name="laplace_smc",
            likelihood_backend=backend,
            extra_diagnostics={"n_ieks_iters": n_ieks_iters, "likelihood_backend": backend},
            print_prefix="Laplace-SMC",
            reparam=reparam,
        )
