"""Inference result and artifact types."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Required, TypedDict

import jax.numpy as jnp
import jax.random as random
from numpyro.infer import Predictive

from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    EnergyDiagnosticsData,
    PosteriorMarginalData,
    PosteriorPairData,
    RankHistogramData,
    TraceSeriesData,
    compute_posterior_marginals,
    compute_posterior_pairs,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_energy_diagnostics as _build_energy_diagnostics,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_rank_histograms as _build_rank_histograms,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_trace_data as _build_trace_data,
)
from nof1_causal_lab.models.ssm.inference.shared import _filter_public_samples

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

logger = logging.getLogger(__name__)


InferenceMethod = Literal["marginal_particle_gibbs"]
InferenceResultMethod = Literal["marginal_particle_gibbs", "map"]


class InferenceDiagnostics(TypedDict, total=False):
    """Known live inference diagnostics shared by MAP and particle methods."""

    mcmc: TrajectoryMCMCResult
    public_sites: list[str]
    likelihood_backend: object
    latent_posterior_summary: dict[str, jnp.ndarray]
    latent_paths: jnp.ndarray
    warmup_latent_paths: jnp.ndarray
    all_latent_paths: jnp.ndarray
    beta_schedule: jnp.ndarray
    ess_history: jnp.ndarray
    accept_rates: jnp.ndarray
    n_levels: int
    n_csmc_particles: int
    optimizer: str
    success: bool
    status: int
    n_iters: int
    n_function_evals: int
    objective_at_mode: float
    mode_log_posterior: float
    mode_log_likelihood: float
    mode_log_prior: float
    mode_grad_norm: float | None
    mode_inner_solver: str
    mode_inner_iterations: int
    mode_inner_accepted_steps: int
    mode_inner_rel_change: float
    mode_inner_damping: float
    mode_inner_step_alpha: float
    mode_inner_step_norm: float
    mode_inner_log_joint_gain: float | None
    mode_inner_laplace_logdet: float
    mode_inner_min_chol_diag: float
    init_log_posterior_best: float
    n_init_samples: int
    n_ieks_iters: int
    parameter_covariance: jnp.ndarray | NDArray
    covariance_diag: list[float]
    compute_parameter_hessian: bool
    parameter_posterior_strategy: str
    parameter_covariance_method: str
    hessian_condition_number: float | None
    parameter_hessian_min_eig: float | None
    parameter_hessian_max_eig: float | None
    hessian_jitter: float
    marginal_particle_gibbs: JsonObject
    marginal_particle_gibbs_phase_extra_fields: dict[str, dict[str, jnp.ndarray]]
    chain_complete_log_posterior_history: jnp.ndarray
    warmup_complete_log_posterior_history: jnp.ndarray
    all_complete_log_posterior_history: jnp.ndarray


class MCMCParameterDiagnostic(TypedDict, total=False):
    """Convergence metrics for one scalar or array-valued parameter."""

    parameter: Required[str]
    r_hat: float | list[float]
    ess_bulk: float | list[float]
    ess_tail: float | list[float]
    mcse_mean: float | list[float]


class MCMCResultDiagnostics(TypedDict, total=False):
    """JSON-ready diagnostics exposed by a particle-MCMC result."""

    per_parameter: list[MCMCParameterDiagnostic]
    num_divergences: int
    divergence_rate: float
    tree_depth_mean: float
    tree_depth_max: int
    accept_prob_mean: float
    latent_accept_prob_mean: float
    parameter_accept_prob_mean: float
    energy: EnergyDiagnosticsData
    num_chains: int
    num_samples: int
    trace_data: list[TraceSeriesData]
    rank_histograms: list[RankHistogramData]


@dataclass
class InferenceResult:
    """Container for inference results across all backends."""

    _samples: dict[str, jnp.ndarray]
    method: InferenceResultMethod
    diagnostics: InferenceDiagnostics = field(default_factory=dict)

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Return posterior samples dict."""
        return self._samples

    def get_latent_paths(self) -> jnp.ndarray | None:
        """Return retained latent path samples when available."""
        return self.diagnostics.get("latent_paths")

    def get_mcmc_diagnostics(self) -> MCMCResultDiagnostics | None:
        """Extract JSON-serializable MCMC diagnostics."""
        if self.method == "map":
            return None

        mcmc = self.diagnostics.get("mcmc")
        if mcmc is None:
            return None

        from numpyro.diagnostics import summary as numpyro_summary

        result: MCMCResultDiagnostics = {}

        chain_samples = mcmc.get_samples(group_by_chain=True)
        public_sites = self.diagnostics.get("public_sites")
        if public_sites is not None:
            chain_samples = _filter_public_samples(chain_samples, set(public_sites))

        def _arviz_idata_from_posterior(posterior_dict, *, log_likelihood=None):
            import arviz as az
            import numpy as np

            posterior_np = {name: np.asarray(values) for name, values in posterior_dict.items()}
            kwargs = {"posterior": posterior_np}
            if log_likelihood is not None:
                kwargs["log_likelihood"] = {
                    name: np.asarray(values) for name, values in log_likelihood.items()
                }
            return az.from_dict(**kwargs)

        summ = numpyro_summary(chain_samples)
        per_param: list[MCMCParameterDiagnostic] = []
        for name, stats in summ.items():
            entry: MCMCParameterDiagnostic = {"parameter": name}
            if "r_hat" in stats:
                val = stats["r_hat"]
                entry["r_hat"] = float(val) if val.ndim == 0 else [float(v) for v in val.flat]
            if "n_eff" in stats:
                val = stats["n_eff"]
                entry["ess_bulk"] = float(val) if val.ndim == 0 else [float(v) for v in val.flat]
            per_param.append(entry)
        result["per_parameter"] = per_param

        import arviz as az

        if getattr(mcmc, "backend", None) in {
            "aux_kalman_mcmc",
            "pit_particle_mgrad",
            "marginal_particle_gibbs",
        }:
            idata = _arviz_idata_from_posterior(chain_samples)
        else:
            idata = az.from_numpyro(mcmc)
        ess_tail = az.ess(idata, method="tail")
        mcse_mean = az.mcse(idata, method="mean")

        for entry in per_param:
            name = entry["parameter"]
            if name in ess_tail:
                v = ess_tail[name].values
                entry["ess_tail"] = float(v) if v.ndim == 0 else [float(x) for x in v.flat]
            if name in mcse_mean:
                v = mcse_mean[name].values
                entry["mcse_mean"] = float(v) if v.ndim == 0 else [float(x) for x in v.flat]

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
        if "latent_accept_prob" in extra:
            ap = extra["latent_accept_prob"]
            result["latent_accept_prob_mean"] = float(jnp.mean(ap))
        if "parameter_accept_prob" in extra:
            ap = extra["parameter_accept_prob"]
            result["parameter_accept_prob_mean"] = float(jnp.mean(ap))
        if "energy" in extra:
            energy = extra["energy"]
            n_ch = int(mcmc.num_chains)
            if n_ch > 1 and energy.ndim == 1 and energy.shape[0] % n_ch == 0:
                energy = energy.reshape(n_ch, -1)
            result["energy"] = _build_energy_diagnostics(energy)

        result["num_chains"] = int(mcmc.num_chains)
        result["num_samples"] = int(mcmc.num_samples)

        if chain_samples is not None:
            result["trace_data"] = _build_trace_data(chain_samples, max_points=200)
            result["rank_histograms"] = _build_rank_histograms(chain_samples, n_bins=20)

        return result

    def get_smc_diagnostics(self) -> JsonObject | None:
        """Extract JSON-serializable SMC diagnostics."""
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

    def get_loo_diagnostics(
        self,
        model_fn: Callable[..., object] | None = None,
        observations: jnp.ndarray | None = None,
        times: jnp.ndarray | None = None,
    ) -> JsonObject | None:
        """Extract LOO-CV diagnostics via ArviZ using one-step-ahead predictive LL."""
        if model_fn is None or observations is None:
            return None

        mcmc = self.diagnostics.get("mcmc")
        if mcmc is None and not self._samples:
            return None

        import arviz as az

        if mcmc is not None:
            flat_samples = mcmc.get_samples()
            public_sites = self.diagnostics.get("public_sites")
            if public_sites is not None:
                flat_samples = _filter_public_samples(flat_samples, set(public_sites))
            n_draws = next(iter(flat_samples.values())).shape[0]
            n_chains = int(mcmc.num_chains)
        else:
            flat_samples = self._samples
            n_draws = next(iter(flat_samples.values())).shape[0]
            n_chains = 1

        n_per_chain = n_draws // n_chains
        pred = Predictive(model_fn, posterior_samples=flat_samples)
        pred_result = pred(random.PRNGKey(0), observations, times)

        def _arviz_idata_from_posterior(posterior_dict, *, log_likelihood=None):
            import arviz as az
            import numpy as np

            posterior_np = {name: np.asarray(values) for name, values in posterior_dict.items()}
            kwargs = {"posterior": posterior_np}
            if log_likelihood is not None:
                kwargs["log_likelihood"] = {
                    name: np.asarray(values) for name, values in log_likelihood.items()
                }
            return az.from_dict(**kwargs)

        if "ll_per_timestep" in pred_result:
            ll_per_t = pred_result["ll_per_timestep"]
            if observations.ndim == 2:
                valid_timesteps = jnp.any(~jnp.isnan(observations), axis=1)
                ll_per_t = ll_per_t[:, valid_timesteps]
            if ll_per_t.shape[1] == 0:
                return None
            n_timesteps = ll_per_t.shape[1]
            ll_chained = ll_per_t[: n_chains * n_per_chain].reshape(
                n_chains, n_per_chain, n_timesteps
            )
            if mcmc is not None:
                if getattr(mcmc, "backend", None) in {
                    "aux_kalman_mcmc",
                    "pit_particle_mgrad",
                    "marginal_particle_gibbs",
                }:
                    chain_samples = mcmc.get_samples(group_by_chain=True)
                    if public_sites is not None:
                        chain_samples = _filter_public_samples(chain_samples, set(public_sites))
                    idata = _arviz_idata_from_posterior(
                        chain_samples,
                        log_likelihood={"ll_per_timestep": ll_chained},
                    )
                else:
                    idata = az.from_numpyro(
                        mcmc,
                        log_likelihood={"ll_per_timestep": ll_chained},
                    )
            else:
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
        elif mcmc is not None:
            if getattr(mcmc, "backend", None) in {
                "aux_kalman_mcmc",
                "pit_particle_mgrad",
                "marginal_particle_gibbs",
            }:
                chain_samples = mcmc.get_samples(group_by_chain=True)
                if public_sites is not None:
                    chain_samples = _filter_public_samples(chain_samples, set(public_sites))
                idata = _arviz_idata_from_posterior(chain_samples)
            else:
                idata = az.from_numpyro(mcmc)
            if not hasattr(idata, "log_likelihood"):
                return None
            ll_per_timestep_found = False
        else:
            return None

        loo_result = az.loo(idata)

        result: JsonObject = {
            "elpd_loo": float(loo_result.elpd_loo),
            "p_loo": float(loo_result.p_loo),
            "se": float(loo_result.se),
            "n_data_points": int(loo_result.n_data_points),
            "observation_unit": "timestep" if ll_per_timestep_found else "observation",
        }

        if hasattr(loo_result, "pareto_k"):
            pk = loo_result.pareto_k
            pk_vals = pk.values if hasattr(pk, "values") else jnp.array(pk)
            result["pareto_k"] = [float(v) for v in pk_vals]
            result["n_bad_k"] = int((pk_vals > 0.7).sum())

        if ll_per_timestep_found and hasattr(idata, "observed_data"):
            pit_vals = az.loo_pit(idata, y="ll_per_timestep")
            if hasattr(pit_vals, "values"):
                result["loo_pit"] = [float(v) for v in pit_vals.values.flat]
            else:
                result["loo_pit"] = [float(v) for v in jnp.array(pit_vals).flatten()]

        return result

    def get_posterior_marginals(self, n_bins: int = 50) -> list[PosteriorMarginalData]:
        """Compute marginal posterior density data for visualization."""
        return compute_posterior_marginals(self._samples, n_bins)

    def get_posterior_pairs(
        self, max_params: int = 6, max_samples: int = 200
    ) -> list[PosteriorPairData]:
        """Compute pairwise scatter data for joint posterior visualization."""
        return compute_posterior_pairs(
            self._samples,
            self.diagnostics.get("mcmc"),
            max_params,
            max_samples,
        )


def _serialize_fitted_result(result: InferenceResult | None) -> InferenceResult | None:
    """Reduce persisted inference output to the posterior samples analysis uses.

    Fits store dynamics parameters in ``_samples`` and keep retained latent
    paths in ``diagnostics`` for analysis counterfactual starts. Live inference
    caches such as the MCMC object are not picklable and are dropped.
    """
    if result is None:
        return None
    diagnostics: InferenceDiagnostics = {}
    if "latent_paths" in result.diagnostics:
        diagnostics["latent_paths"] = result.diagnostics["latent_paths"]
    return InferenceResult(
        _samples=result.get_samples(),
        method=result.method,
        diagnostics=diagnostics,
    )


@dataclass
class FittedArtifact:
    """Canonical persisted output of inference."""

    result: InferenceResult | None
    spec: SSMSpec | None
    times: jnp.ndarray
    observation_support: ObservationSupportRuntime | None = None
    ppc_result: JsonObject | None = None

    def __getstate__(self) -> FittedArtifactState:
        """Persist only the analysis inputs, never live inference caches/backends."""
        return {
            "result": _serialize_fitted_result(self.result),
            "spec": self.spec,
            "times": self.times,
            "observation_support": self.observation_support,
            "ppc_result": self.ppc_result,
        }

    def __setstate__(self, state: FittedArtifactState) -> None:
        self.result = state["result"]
        self.spec = state["spec"]
        self.times = state["times"]
        self.observation_support = state["observation_support"]
        self.ppc_result = state["ppc_result"]


class FittedArtifactState(TypedDict):
    """Pickled analysis-only state for a fitted artifact."""

    result: InferenceResult | None
    spec: SSMSpec | None
    times: jnp.ndarray
    observation_support: ObservationSupportRuntime | None
    ppc_result: JsonObject | None
