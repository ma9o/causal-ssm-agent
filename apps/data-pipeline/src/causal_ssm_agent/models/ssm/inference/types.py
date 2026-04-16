"""Inference result and artifact types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
import jax.random as random
from numpyro.infer import Predictive

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.inference.diagnostics_viz import (
    build_energy_diagnostics as _build_energy_diagnostics,
)
from causal_ssm_agent.models.ssm.inference.diagnostics_viz import (
    build_rank_histograms as _build_rank_histograms,
)
from causal_ssm_agent.models.ssm.inference.diagnostics_viz import (
    build_trace_data as _build_trace_data,
)
from causal_ssm_agent.models.ssm.inference.diagnostics_viz import (
    compute_posterior_marginals,
    compute_posterior_pairs,
    format_summary,
)
from causal_ssm_agent.models.ssm.inference.shared import _filter_public_samples

logger = get_prefect_logger(__name__)


InferenceMethod = Literal[
    "auto",
    "nuts",
    "map",
    "svi",
]


@dataclass
class InferenceResult:
    """Container for inference results across all backends."""

    _samples: dict[str, jnp.ndarray]
    method: InferenceMethod
    diagnostics: dict = field(default_factory=dict)

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Return posterior samples dict."""
        return self._samples

    def get_mcmc_diagnostics(self) -> dict[str, Any] | None:
        """Extract JSON-serializable MCMC diagnostics."""
        if self.method in ("svi", "map"):
            return None

        mcmc = self.diagnostics.get("mcmc")
        if mcmc is None:
            return None

        from numpyro.diagnostics import summary as numpyro_summary

        result: dict[str, Any] = {}

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
                entry["ess_bulk"] = float(val) if val.ndim == 0 else [float(v) for v in val.flat]
            per_param.append(entry)
        result["per_parameter"] = per_param

        import arviz as az

        idata = az.from_numpyro(mcmc)
        ess_tail = az.ess(idata, method="tail")
        mcse_mean = az.mcse(idata, method="mean")

        for entry in result["per_parameter"]:
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

    def get_smc_diagnostics(self) -> dict[str, Any] | None:
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

    def get_svi_diagnostics(self) -> dict[str, Any] | None:
        """Extract JSON-serializable SVI diagnostics (ELBO loss curve)."""
        if self.method != "svi":
            return None

        losses = self.diagnostics.get("losses")
        if losses is None:
            return None

        return {"elbo_losses": [float(v) for v in losses]}

    def get_loo_diagnostics(
        self,
        model_fn: Any = None,
        observations: jnp.ndarray | None = None,
        times: jnp.ndarray | None = None,
    ) -> dict[str, Any] | None:
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
            idata = az.from_numpyro(mcmc)
            if not hasattr(idata, "log_likelihood"):
                return None
            ll_per_timestep_found = False
        else:
            return None

        loo_result = az.loo(idata)

        result: dict[str, Any] = {
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


def _serialize_fitted_result(result: InferenceResult | None) -> InferenceResult | None:
    """Reduce persisted inference output to the posterior samples Stage 6 uses."""
    if result is None:
        return None
    return InferenceResult(
        _samples=result.get_samples(),
        method=result.method,
        diagnostics={},
    )


def _serialize_fitted_builder(builder: Any) -> Any:
    """Persist only the compiled spec Stage 6 needs for counterfactual analysis."""
    if builder is None:
        return None
    return FittedBuilderSnapshot(spec=builder.spec)


@dataclass(frozen=True)
class FittedBuilderSnapshot:
    """Minimal persisted builder state required by Stage 6."""

    spec: Any | None


@dataclass
class FittedArtifact:
    """Canonical persisted output of inference."""

    result: InferenceResult | None
    builder: Any | None
    times: Any
    observation_support: Any | None = None
    ppc_result: dict[str, Any] | None = None
    power_scaling_result: dict[str, Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Persist only the Stage 6 inputs, never live inference caches/backends."""
        return {
            "result": _serialize_fitted_result(self.result),
            "builder": _serialize_fitted_builder(self.builder),
            "times": self.times,
            "observation_support": self.observation_support,
            "ppc_result": self.ppc_result,
            "power_scaling_result": self.power_scaling_result,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.result = state["result"]
        self.builder = state["builder"]
        self.times = state["times"]
        self.observation_support = state["observation_support"]
        self.ppc_result = state["ppc_result"]
        self.power_scaling_result = state["power_scaling_result"]
