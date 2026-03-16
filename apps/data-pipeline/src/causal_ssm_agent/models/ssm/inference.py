"""Inference backends for SSM models.

Separates inference from model definition. SSMModel defines the probabilistic
model; this module provides fit() to run inference with different backends.

Structural routing (method="auto", the default) selects the best method
from model properties: "nuts" for Kalman-eligible models (all Gaussian +
identity link), "laplace_em" for non-Gaussian emissions. Users can override
to any specific method. See docs/modeling/inference-strategies.md for details.

Available methods:
- NUTS: HMC-based sampling. Gold standard for Kalman-eligible models.
- SVI: Fast approximate posterior via ELBO optimization.
- Tempered SMC: Adaptive tempering with preconditioned HMC/MALA mutations.
- Hess-MC²: SMC with gradient-based change-of-variables L-kernels.
- Laplace-EM: IEKS + Laplace approximation for non-Gaussian emissions.
- Structured VI: Backward-factored variational family.
- DPF: Differentiable Particle Filter with learned proposals.
- NUTS-DA: Data augmentation MCMC — jointly samples params and latent states.
- PGAS: Particle Gibbs with ancestor sampling + gradient-informed proposals.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import jax.random as random
from jax import tree_util
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoNormal
from numpyro.optim import ClippedAdam

from causal_ssm_agent.models.ssm.autoreparam import AutoReparam, reparam_cache_key
from causal_ssm_agent.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES
from causal_ssm_agent.models.ssm.diagnostics_viz import (
    build_energy_diagnostics as _build_energy_diagnostics,
)
from causal_ssm_agent.models.ssm.diagnostics_viz import (
    build_rank_histograms as _build_rank_histograms,
)
from causal_ssm_agent.models.ssm.diagnostics_viz import (
    build_trace_data as _build_trace_data,
)
from causal_ssm_agent.models.ssm.diagnostics_viz import (
    compute_posterior_marginals,
    compute_posterior_pairs,
    format_summary,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMModel, SSMSpec

from causal_ssm_agent.flows import get_prefect_logger

logger = get_prefect_logger(__name__)

# Sentinel for "use AutoReparam with method-appropriate centering".
_AUTO_REPARAM = object()

# Re-export for tests that import from here
HIST_PADDING_RATIO = 0.05
HIST_PADDING_DEFAULT = 0.5

_AUTO_METHOD_CONFIG_KEYS: dict[str, str] = {
    "svi": "svi_config",
    "nuts": "nuts_config",
    "laplace_em": "smc_config",
    "tempered_smc": "smc_config",
    "structured_vi": "smc_config",
    "dpf": "smc_config",
}


def _make_prior_predictive_dummy_observations(spec: SSMSpec, times: jnp.ndarray) -> jnp.ndarray:
    """Create support-compatible observations for prior predictive tracing."""
    n_times = len(times)
    manifest_dists = spec.manifest_dists or [spec.manifest_dist] * spec.n_manifest
    alternating_binary = (jnp.arange(n_times, dtype=jnp.float32) % 2).astype(jnp.float32)

    cols = []
    for dist in manifest_dists:
        if dist.is_discrete:
            cols.append(alternating_binary)
        else:
            cols.append(jnp.full((n_times,), dist.support_interior_point, dtype=jnp.float32))

    if not cols:
        return jnp.zeros((n_times, 0), dtype=jnp.float32)
    return jnp.stack(cols, axis=1)


InferenceMethod = Literal[
    "auto",
    "nuts",
    "nuts_da",
    "svi",
    "hessmc2",
    "pgas",
    "tempered_smc",
    "laplace_em",
    "structured_vi",
    "dpf",
]


@dataclass
class InferenceResult:
    """Container for inference results across all backends.

    Provides a uniform interface regardless of which backend was used.
    """

    _samples: dict[str, jnp.ndarray]  # name -> (n_draws, *shape)
    method: InferenceMethod
    diagnostics: dict = field(default_factory=dict)

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Return posterior samples dict."""
        return self._samples

    def get_mcmc_diagnostics(self) -> dict[str, Any] | None:
        """Extract JSON-serializable MCMC diagnostics.

        Returns per-parameter R-hat, ESS (bulk+tail), MCSE, trace data,
        rank histograms, and sampler-level divergence/tree stats.
        Returns None for non-MCMC methods (SVI, etc.).
        """
        if self.method in ("svi", "structured_vi", "laplace_em"):
            return None

        mcmc = self.diagnostics.get("mcmc")
        if mcmc is None:
            return None

        from numpyro.diagnostics import summary as numpyro_summary

        result: dict[str, Any] = {}
        chain_samples = None

        # Per-parameter convergence diagnostics via numpyro.diagnostics.summary
        try:
            chain_samples = mcmc.get_samples(group_by_chain=True)
            public_sites = self.diagnostics.get("public_sites")
            if public_sites is not None:
                chain_samples = _filter_public_samples(chain_samples, set(public_sites))
            summ = numpyro_summary(chain_samples)
            per_param: list[dict[str, Any]] = []
            for name, stats in summ.items():
                entry: dict[str, Any] = {"parameter": name}
                if "r_hat" in stats:
                    val = stats["r_hat"]
                    entry["r_hat"] = float(val) if val.ndim == 0 else [float(v) for v in val.flat]
                if "n_eff" in stats:
                    val = stats["n_eff"]
                    entry["ess_bulk"] = (
                        float(val) if val.ndim == 0 else [float(v) for v in val.flat]
                    )
                per_param.append(entry)
            result["per_parameter"] = per_param
        except Exception:
            logger.debug("Failed to compute per-parameter diagnostics", exc_info=True)
            result["per_parameter"] = []

        # ArviZ-based ESS-tail and MCSE (enriches per_parameter entries)
        try:
            import arviz as az

            idata = az.from_numpyro(mcmc)
            ess_tail = az.ess(idata, method="tail")
            mcse_mean = az.mcse(idata, method="mean")

            # Merge into per_parameter entries
            for entry in result["per_parameter"]:
                name = entry["parameter"]
                if name in ess_tail:
                    v = ess_tail[name].values
                    entry["ess_tail"] = float(v) if v.ndim == 0 else [float(x) for x in v.flat]
                if name in mcse_mean:
                    v = mcse_mean[name].values
                    entry["mcse_mean"] = float(v) if v.ndim == 0 else [float(x) for x in v.flat]
        except Exception:
            logger.debug("ArviZ ESS-tail/MCSE enrichment failed", exc_info=True)

        # Sampler-level diagnostics from extra fields
        try:
            extra = mcmc.get_extra_fields()
            if "diverging" in extra:
                div = extra["diverging"]
                result["num_divergences"] = int(jnp.sum(div))
                result["divergence_rate"] = float(jnp.mean(div))
            if "num_steps" in extra:
                steps = extra["num_steps"]
                result["tree_depth_mean"] = float(jnp.mean(steps))
                result["tree_depth_max"] = int(jnp.max(steps))
            if "accept_prob" in extra:
                ap = extra["accept_prob"]
                result["accept_prob_mean"] = float(jnp.mean(ap))
            if "energy" in extra:
                energy = extra["energy"]
                # Reshape to (n_chains, n_draws) if possible for per-chain BFMI
                n_ch = int(mcmc.num_chains) if hasattr(mcmc, "num_chains") else 1
                if n_ch > 1 and energy.ndim == 1 and energy.shape[0] % n_ch == 0:
                    energy = energy.reshape(n_ch, -1)
                result["energy"] = _build_energy_diagnostics(energy)
        except Exception:
            logger.debug("Sampler-level diagnostics extraction failed", exc_info=True)

        result["num_chains"] = int(mcmc.num_chains) if hasattr(mcmc, "num_chains") else None
        result["num_samples"] = int(mcmc._num_samples) if hasattr(mcmc, "_num_samples") else None

        # Chain-level trace data (thinned to ~200 points per chain)
        # and rank histograms for chain mixing assessment
        if chain_samples is not None:
            result["trace_data"] = _build_trace_data(chain_samples, max_points=200)
            result["rank_histograms"] = _build_rank_histograms(chain_samples, n_bins=20)

        return result

    def get_smc_diagnostics(self) -> dict[str, Any] | None:
        """Extract JSON-serializable SMC diagnostics.

        Returns tempering schedule, ESS history, and acceptance rates
        for methods backed by tempered SMC (laplace_em, tempered_smc, etc.).
        Returns None for non-SMC methods.
        """
        if self.method in ("nuts", "nuts_da", "svi"):
            return None

        beta = self.diagnostics.get("beta_schedule")
        if beta is None:
            return None

        return {
            "beta_schedule": [float(b) for b in beta],
            "ess_history": [float(e) for e in self.diagnostics.get("ess_history", [])],
            "accept_rates": [float(a) for a in self.diagnostics.get("accept_rates", [])],
            "n_levels": int(self.diagnostics.get("n_levels", len(beta))),
            "n_particles": int(self.diagnostics.get("n_csmc_particles", 0)),
        }

    def get_svi_diagnostics(self) -> dict[str, Any] | None:
        """Extract JSON-serializable SVI diagnostics (ELBO loss curve).

        Returns None for non-SVI methods.
        """
        if self.method != "svi":
            return None

        losses = self.diagnostics.get("losses")
        if losses is None:
            return None

        loss_list = [float(v) for v in losses]
        # Thin to at most 500 points for the frontend
        if len(loss_list) > 500:
            step = len(loss_list) / 500
            loss_list = [loss_list[int(i * step)] for i in range(500)]

        return {"elbo_losses": loss_list}

    def get_loo_diagnostics(
        self,
        model_fn: Any = None,
        observations: jnp.ndarray | None = None,
        times: jnp.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Extract LOO-CV diagnostics via ArviZ using one-step-ahead predictive LL.

        Uses the innovation decomposition from the Kalman/particle filter:
        each "observation" for LOO is one complete timestep (all manifest
        variables at time t). The per-timestep log-likelihoods
        log p(y_t | y_{1:t-1}, θ) are conditionally independent given θ,
        making PSIS-LOO valid for time series via this decomposition.

        Works for both MCMC-based methods (NUTS) and SMC-based methods
        (laplace_em, tempered_smc). For MCMC, uses az.from_numpyro; for SMC,
        constructs InferenceData from the posterior samples dict directly.

        Args:
            model_fn: The NumPyro model function (needed to replay for ll_per_timestep)
            observations: (T, n_manifest) observed data
            times: (T,) time points

        Returns:
            Dict with LOO diagnostics, or None if not computable.
        """
        if model_fn is None or observations is None:
            return None

        mcmc = self.diagnostics.get("mcmc")
        if mcmc is None and not self._samples:
            return None

        try:
            import arviz as az

            # Get posterior samples and chain structure.
            # MCMC: extract from MCMC object (has chain info).
            # SMC: use stored posterior samples directly (1 chain of N particles).
            if mcmc is not None:
                flat_samples = mcmc.get_samples()
                public_sites = self.diagnostics.get("public_sites")
                if public_sites is not None:
                    flat_samples = _filter_public_samples(flat_samples, set(public_sites))
                n_draws = next(iter(flat_samples.values())).shape[0]
                n_chains = int(mcmc.num_chains) if hasattr(mcmc, "num_chains") else 1
            else:
                flat_samples = self._samples
                n_draws = next(iter(flat_samples.values())).shape[0]
                n_chains = 1

            n_per_chain = n_draws // n_chains

            # Try SSM-specific path first: replay model to extract
            # ll_per_timestep deterministic (innovation decomposition).
            # Falls back to standard ArviZ extraction for models with
            # observed sample sites (e.g. numpyro.sample(..., obs=y)).
            ll_per_timestep_found = False
            try:
                pred = Predictive(model_fn, posterior_samples=flat_samples)
                rng_key = random.PRNGKey(0)
                pred_result = pred(rng_key, observations, times)
                if "ll_per_timestep" in pred_result:
                    ll_per_t = pred_result["ll_per_timestep"]  # (n_draws, T)
                    n_timesteps = ll_per_t.shape[1]
                    ll_chained = ll_per_t[: n_chains * n_per_chain].reshape(
                        n_chains, n_per_chain, n_timesteps
                    )
                    if mcmc is not None:
                        idata = az.from_numpyro(
                            mcmc,
                            log_likelihood={"ll_per_timestep": ll_chained},
                        )
                    else:
                        # Build InferenceData from dict for SMC-based methods.
                        # Treat N particles as 1 chain of N draws.
                        import numpy as np

                        posterior_dict = {}
                        n_used = n_chains * n_per_chain
                        for name, vals in flat_samples.items():
                            v = np.asarray(vals[:n_used])
                            posterior_dict[name] = v.reshape(n_chains, n_per_chain, *v.shape[1:])
                        idata = az.from_dict(
                            posterior=posterior_dict,
                            log_likelihood={"ll_per_timestep": np.asarray(ll_chained)},
                        )
                    ll_per_timestep_found = True
            except Exception:
                logger.debug("SSM-specific LOO path failed, trying standard ArviZ", exc_info=True)

            if not ll_per_timestep_found:
                if mcmc is not None:
                    # Standard path: ArviZ extracts LL from observed sample sites
                    idata = az.from_numpyro(mcmc)
                    if not hasattr(idata, "log_likelihood"):
                        return None
                else:
                    # No ll_per_timestep and no MCMC → can't compute LOO
                    return None

            loo_result = az.loo(idata)

            result: dict[str, Any] = {
                "elpd_loo": float(loo_result.elpd_loo),
                "p_loo": float(loo_result.p_loo),
                "se": float(loo_result.se),
                "n_data_points": int(loo_result.n_data_points),
                "observation_unit": "timestep" if ll_per_timestep_found else "observation",
            }

            # Per-data-point Pareto k values
            if hasattr(loo_result, "pareto_k"):
                pk = loo_result.pareto_k
                pk_vals = pk.values if hasattr(pk, "values") else jnp.array(pk)
                result["pareto_k"] = [float(v) for v in pk_vals]
                result["n_bad_k"] = int((pk_vals > 0.7).sum())

            # LOO-PIT for calibration (SSM path only)
            if ll_per_timestep_found:
                try:
                    pit_vals = az.loo_pit(idata, y="ll_per_timestep")
                    if hasattr(pit_vals, "values"):
                        result["loo_pit"] = [float(v) for v in pit_vals.values.flat]
                    else:
                        result["loo_pit"] = [float(v) for v in jnp.array(pit_vals).flatten()]
                except Exception:
                    logger.debug("LOO-PIT computation failed", exc_info=True)

            return result

        except Exception:
            logger.debug("LOO diagnostics computation failed", exc_info=True)
            return None

    def get_posterior_marginals(self, n_bins: int = 50) -> list[dict[str, Any]]:
        """Compute marginal posterior density data for visualization."""
        return compute_posterior_marginals(self._samples, n_bins)

    def get_posterior_pairs(
        self, max_params: int = 6, max_samples: int = 200
    ) -> list[dict[str, Any]]:
        """Compute pairwise scatter data for joint posterior visualization."""
        return compute_posterior_pairs(self._samples, self.diagnostics, max_params, max_samples)

    def print_summary(self) -> None:
        """Log summary statistics for posterior samples."""
        logger.info("\n%s", format_summary(self._samples, self.method))


def select_default_method(spec: SSMSpec) -> InferenceMethod:
    """Select the default inference method based on model structure.

    Implements the structural routing decision tree from
    docs/modeling/inference-strategies.md:

    1. A = Marginalize (structural default for all models)
    2. Determine B from model structure:
       - B = Closed-form (Kalman) if partition.has_particle_block is False
         (all emissions Gaussian + identity link + Gaussian diffusion)
       - B = Deterministic approx (IEKS) otherwise
    3. Select C given B:
       - Kalman → exact smooth gradients → MCMC optimal → "nuts"
       - IEKS → smooth approximate gradients → SMC (multimodality
         protection for non-Gaussian emission posteriors) → "laplace_em"

    User overrides (nuts_da, pgas, svi, hessmc2, dpf, structured_vi)
    bypass this routing entirely.

    Args:
        spec: SSMSpec encoding model structure decisions.

    Returns:
        "nuts" for Kalman-eligible models, "laplace_em" otherwise.
    """
    from causal_ssm_agent.models.likelihoods.graph_analysis import analyze_first_pass_rb

    partition = analyze_first_pass_rb(spec)

    if not partition.has_particle_block:
        # B = Closed-form (Kalman): all latent-obs chains are linear-Gaussian
        # with identity links. Exact, smooth gradients → MCMC is optimal.
        logger.info("Structural routing: Kalman-eligible model → nuts")
        return "nuts"

    # B = Deterministic approx (IEKS/Laplace): non-Gaussian emissions or
    # non-identity links. CT-LTI dynamics are always linear and all 7
    # emission families have C² log-densities, so IEKS is always available.
    # SMC handles multimodality in the parameter posterior.
    logger.info("Structural routing: non-Kalman model → laplace_em")
    return "laplace_em"


def _trace_public_sites(
    model_fn,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    exclude: set[str] | None = None,
) -> set[str]:
    """Trace a model once and return user-facing sample/deterministic site names."""
    excluded = set(INTERNAL_DIAGNOSTIC_SITES)
    if exclude is not None:
        excluded.update(exclude)

    with handlers.seed(rng_seed=0):
        trace = handlers.trace(model_fn).get_trace(observations, times)

    return {
        name
        for name, site in trace.items()
        if site["type"] in ("sample", "deterministic")
        and not site.get("is_observed", False)
        and name not in excluded
    }


def _filter_public_samples(
    samples: dict[str, jnp.ndarray], public_sites: set[str] | None
) -> dict[str, jnp.ndarray]:
    """Drop internal handler sites, keeping only original model outputs."""
    if public_sites is None:
        return samples
    return {name: values for name, values in samples.items() if name in public_sites}


def _all_numeric_leaves_finite(tree: Any) -> bool:
    """Return ``True`` when every numeric leaf in a pytree is finite."""
    for leaf in tree_util.tree_leaves(tree):
        arr = jnp.asarray(leaf)
        if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
            continue
        if not bool(jnp.all(jnp.isfinite(arr))):
            return False
    return True


def _apply_reparam(model_fn, reparam_config):
    """Wrap a model function with reparameterization if config is provided.

    Args:
        model_fn: A NumPyro model function.
        reparam_config: A dict, callable (Strategy), or None.

    Returns:
        The model function, possibly wrapped with handlers.reparam.
    """
    if reparam_config is None:
        return model_fn
    return handlers.reparam(model_fn, config=reparam_config)


def _resolve_reparam(reparam, method: InferenceMethod):
    """Resolve _AUTO_REPARAM sentinel to a concrete AutoReparam config."""
    if reparam is not _AUTO_REPARAM:
        return reparam
    # SVI benefits from learnable centering; all other methods use fixed NCP.
    if method == "svi":
        return AutoReparam()  # centered=None → learnable via numpyro.param
    if method == "pgas":
        return None
    return AutoReparam(centered=0.0)  # fully decentered


def _resolve_auto_method_kwargs(
    method: InferenceMethod,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge backend-specific config blocks emitted by ``method='auto'``."""
    method_configs = {
        "svi_config": kwargs.get("svi_config"),
        "nuts_config": kwargs.get("nuts_config"),
        "smc_config": kwargs.get("smc_config"),
    }
    resolved = {
        key: value
        for key, value in kwargs.items()
        if key not in {"svi_config", "nuts_config", "smc_config"}
    }
    config_key = _AUTO_METHOD_CONFIG_KEYS.get(method)
    if config_key is None:
        return resolved
    method_config = method_configs.get(config_key) or {}
    return {**method_config, **resolved}


def fit(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    method: InferenceMethod = "auto",
    reparam=_AUTO_REPARAM,
    **kwargs: Any,
) -> InferenceResult:
    """Fit an SSM using the specified inference method.

    Args:
        model: SSMModel instance defining the probabilistic model
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        method: Inference method - "auto" (structural routing, default),
            "nuts", "svi", "hessmc2", "pgas", "tempered_smc", "laplace_em",
            "structured_vi", "dpf", or "nuts_da"
        reparam: Reparameterization config. Can be:
            - ``_AUTO_REPARAM`` (default): Uses ``AutoReparam`` with method-appropriate
              centering (learnable for SVI, fully decentered for MCMC/SMC).
            - A ``Strategy`` instance (e.g., ``AutoReparam(centered=0.0)``)
            - A dict mapping site names to ``Reparam`` instances
            - None: no reparameterization
        **kwargs: Method-specific arguments

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    if method == "auto":
        method = select_default_method(model.spec)
        kwargs = _resolve_auto_method_kwargs(method, kwargs)

    reparam = _resolve_reparam(reparam, method)
    if method == "pgas" and reparam is not None:
        raise ValueError("PGAS does not support reparameterization.")
    if method == "nuts":
        return _fit_nuts(model, observations, times, reparam=reparam, **kwargs)
    if method == "nuts_da":
        from causal_ssm_agent.models.ssm.nuts_da import fit_nuts_da

        return fit_nuts_da(model, observations, times, reparam=reparam, **kwargs)
    if method == "svi":
        return _fit_svi(model, observations, times, reparam=reparam, **kwargs)
    if method == "hessmc2":
        from causal_ssm_agent.models.ssm.hessmc2 import fit_hessmc2

        return fit_hessmc2(model, observations, times, reparam=reparam, **kwargs)
    if method == "pgas":
        from causal_ssm_agent.models.ssm.pgas import fit_pgas

        return fit_pgas(model, observations, times, reparam=reparam, **kwargs)
    if method == "tempered_smc":
        from causal_ssm_agent.models.ssm.tempered_smc import fit_tempered_smc

        return fit_tempered_smc(model, observations, times, reparam=reparam, **kwargs)
    if method == "laplace_em":
        from causal_ssm_agent.models.ssm.laplace_em import fit_laplace_em

        return fit_laplace_em(model, observations, times, reparam=reparam, **kwargs)
    if method == "structured_vi":
        from causal_ssm_agent.models.ssm.structured_vi import fit_structured_vi

        return fit_structured_vi(model, observations, times, reparam=reparam, **kwargs)
    if method == "dpf":
        from causal_ssm_agent.models.ssm.dpf import fit_dpf

        return fit_dpf(model, observations, times, reparam=reparam, **kwargs)
    raise ValueError(
        f"Unknown inference method: {method!r}. "
        "Use 'auto', 'svi', 'nuts', 'nuts_da', 'hessmc2', 'pgas', 'tempered_smc', "
        "'laplace_em', 'structured_vi', or 'dpf'."
    )


def prior_predictive(
    model: SSMModel,
    times: jnp.ndarray,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample from the prior predictive distribution.

    Args:
        model: SSMModel instance
        times: (T,) time points
        num_samples: Number of prior samples
        seed: Random seed

    Returns:
        Dict of prior predictive samples
    """
    from causal_ssm_agent.models.ssm.prior_predictive_runtime import (
        sample_prior_predictive_from_priors,
    )

    return sample_prior_predictive_from_priors(
        model.spec,
        model.priors,
        times,
        num_samples=num_samples,
        seed=seed,
    )


def _fit_nuts(
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
    reparam=None,
    **kwargs: Any,
) -> InferenceResult:
    """Fit using NUTS (HMC).

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        seed: Random seed
        dense_mass: Use dense mass matrix
        target_accept_prob: Target acceptance probability
        max_tree_depth: Max tree depth
        reparam: Optional reparameterization config (Strategy, dict, or None)
        **kwargs: Additional MCMC arguments

    Returns:
        InferenceResult with NUTS samples
    """
    base_model_fn = functools.partial(
        model.model, likelihood_backend=model.make_likelihood_backend()
    )
    public_sites = _trace_public_sites(base_model_fn, observations, times)
    model_fn = _apply_reparam(base_model_fn, reparam)
    kernel = NUTS(
        model_fn,
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
    mcmc.run(
        rng_key,
        observations,
        times,
        extra_fields=("diverging", "num_steps", "accept_prob", "energy"),
    )

    samples = _filter_public_samples(mcmc.get_samples(), public_sites)

    return InferenceResult(
        _samples=samples,
        method="nuts",
        diagnostics={"mcmc": mcmc, "public_sites": sorted(public_sites)},
    )


def _fit_svi(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    guide_type: str = "mvn",
    num_steps: int = 5000,
    num_samples: int = 1000,
    learning_rate: float = 0.01,
    progress_bar: bool = False,
    seed: int = 0,
    reparam=None,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit using Stochastic Variational Inference.

    Uses AutoGuide to learn an approximate posterior. numpyro.factor() sites
    are handled automatically - the guide only models latent sample sites.

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        guide_type: Guide family - "normal", "mvn", or "delta"
        num_steps: Number of SVI optimization steps
        num_samples: Number of posterior samples to draw after fitting
        learning_rate: Adam learning rate
        seed: Random seed
        reparam: Optional reparameterization config (Strategy, dict, or None)
        **kwargs: Ignored

    Returns:
        InferenceResult with approximate posterior samples
    """
    guide_cls = {
        "normal": AutoNormal,
        "mvn": AutoMultivariateNormal,
        "delta": AutoDelta,
    }[guide_type]
    base_model_fn = functools.partial(
        model.model,
        likelihood_backend=model.make_likelihood_backend(),
    )
    cache_key = None
    reparam_key = reparam_cache_key(reparam)
    if reparam_key is not None:
        cache_key = (
            "svi",
            tuple(observations.shape),
            tuple(times.shape),
            guide_type,
            float(learning_rate),
            *reparam_key,
        )

    def _build_svi_bundle():
        public_sites = _trace_public_sites(base_model_fn, observations, times)
        model_fn = _apply_reparam(base_model_fn, reparam)
        guide = guide_cls(model_fn)
        optimizer = ClippedAdam(step_size=learning_rate)
        return {
            "public_sites": public_sites,
            "model_fn": model_fn,
            "guide": guide,
            "svi": SVI(model_fn, guide, optimizer, Trace_ELBO()),
        }

    if cache_key is None:
        cached_bundle = _build_svi_bundle()
    else:
        cached_bundle = model.get_cached_artifact(cache_key, _build_svi_bundle)

    public_sites = cached_bundle["public_sites"]
    model_fn = cached_bundle["model_fn"]
    guide = cached_bundle["guide"]
    svi = cached_bundle["svi"]

    rng_key = random.PRNGKey(seed)
    svi_result = svi.run(
        rng_key,
        num_steps,
        observations,
        times,
        progress_bar=progress_bar,
    )

    if not _all_numeric_leaves_finite(svi_result.losses):
        raise FloatingPointError("SVI produced non-finite losses")
    if not _all_numeric_leaves_finite(svi_result.params):
        raise FloatingPointError("SVI produced non-finite guide parameters")

    # Draw posterior samples from the fitted guide
    sample_key = random.PRNGKey(seed + 1)
    predictive = Predictive(
        model_fn,
        guide=guide,
        params=svi_result.params,
        num_samples=num_samples,
    )
    raw_samples = predictive(sample_key, observations, times)

    samples = _filter_public_samples(raw_samples, public_sites)
    if not _all_numeric_leaves_finite(samples):
        raise FloatingPointError("SVI produced non-finite posterior samples")

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
) -> tuple[Any, Any]:
    """Evaluate model with substituted params. Returns (log_likelihood, log_prior).

    Uses numpyro.handlers to substitute parameter values and trace the model,
    computing log_prior + log_likelihood without any code duplication.

    Args:
        model_fn: NumPyro model function
        params_dict: Parameter values to substitute
        observations: Observed data
        times: Time points

    Returns:
        Tuple of (log_likelihood, log_prior)
    """
    with handlers.seed(rng_seed=0), handlers.substitute(data=params_dict):
        trace = handlers.trace(model_fn).get_trace(observations, times)

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
