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
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import jax.random as random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoNormal
from numpyro.optim import ClippedAdam

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMModel, SSMSpec

logger = logging.getLogger(__name__)

# Histogram binning edge padding to avoid artifacts at distribution tails
HIST_PADDING_RATIO = 0.05
HIST_PADDING_DEFAULT = 0.5

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

        Args:
            model_fn: The NumPyro model function (needed to replay for ll_per_timestep)
            observations: (T, n_manifest) observed data
            times: (T,) time points

        Returns:
            Dict with LOO diagnostics, or None if not computable.
        """
        mcmc = self.diagnostics.get("mcmc")
        if mcmc is None or model_fn is None or observations is None:
            return None

        try:
            import arviz as az

            flat_samples = mcmc.get_samples()
            n_draws = next(iter(flat_samples.values())).shape[0]
            n_chains = int(mcmc.num_chains) if hasattr(mcmc, "num_chains") else 1
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
                    idata = az.from_numpyro(
                        mcmc,
                        log_likelihood={"ll_per_timestep": ll_chained},
                    )
                    ll_per_timestep_found = True
            except Exception:
                logger.debug("SSM-specific LOO path failed, trying standard ArviZ", exc_info=True)

            if not ll_per_timestep_found:
                # Standard path: ArviZ extracts LL from observed sample sites
                idata = az.from_numpyro(mcmc)
                if not hasattr(idata, "log_likelihood"):
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
        """Compute marginal posterior density data for visualization.

        For each scalar parameter, computes a histogram-based density estimate.
        For multi-dimensional parameters, flattens to indexed scalars.

        Args:
            n_bins: Number of histogram bins for density estimation.

        Returns:
            List of {parameter, x_values, density} dicts.
        """
        marginals: list[dict[str, Any]] = []

        for name, values in self._samples.items():
            if values.ndim == 1:
                marginals.append(_param_marginal(name, values, n_bins))
            elif values.ndim >= 2:
                flat = values.reshape(values.shape[0], -1)
                # Only include up to 20 elements to avoid payload bloat
                n_elem = min(flat.shape[1], 20)
                for i in range(n_elem):
                    label = f"{name}[{i}]"
                    marginals.append(_param_marginal(label, flat[:, i], n_bins))

        return marginals

    def get_posterior_pairs(
        self, max_params: int = 6, max_samples: int = 200
    ) -> list[dict[str, Any]]:
        """Compute pairwise scatter data for joint posterior visualization.

        Selects up to max_params scalar parameters and returns thinned
        pairwise samples for scatter matrix plots. Includes per-sample
        divergence flags when available (for highlighting in pairs plots).

        Args:
            max_params: Maximum number of parameters to include.
            max_samples: Maximum samples per pair (thinned evenly).

        Returns:
            List of {param_x, param_y, x_values, y_values, divergent?} dicts.
        """
        # Collect scalar parameters
        scalars: list[tuple[str, jnp.ndarray]] = []
        for name, values in self._samples.items():
            if values.ndim == 1:
                scalars.append((name, values))
            elif values.ndim >= 2:
                flat = values.reshape(values.shape[0], -1)
                for i in range(min(flat.shape[1], 4)):
                    scalars.append((f"{name}[{i}]", flat[:, i]))
            if len(scalars) >= max_params:
                break

        scalars = scalars[:max_params]
        n_draws = scalars[0][1].shape[0] if scalars else 0
        step = max(1, n_draws // max_samples)

        # Get divergence mask (flattened across chains) for highlighting
        div_mask: list[bool] | None = None
        mcmc = self.diagnostics.get("mcmc")
        if mcmc is not None:
            try:
                extra = mcmc.get_extra_fields()
                if "diverging" in extra:
                    div_flat = extra["diverging"].reshape(-1)
                    div_mask = [bool(v) for v in div_flat[::step]]
            except Exception:
                logger.debug("Divergence mask extraction failed", exc_info=True)

        pairs: list[dict[str, Any]] = []
        for i in range(len(scalars)):
            for j in range(i + 1, len(scalars)):
                name_x, vals_x = scalars[i]
                name_y, vals_y = scalars[j]
                entry: dict[str, Any] = {
                    "param_x": name_x,
                    "param_y": name_y,
                    "x_values": [float(v) for v in vals_x[::step]],
                    "y_values": [float(v) for v in vals_y[::step]],
                }
                if div_mask is not None and any(div_mask):
                    entry["divergent"] = div_mask
                pairs.append(entry)

        return pairs

    def print_summary(self) -> None:
        """Log summary statistics for posterior samples."""
        lines = [
            f"Inference method: {self.method}",
            f"{'Parameter':<30} {'Mean':>10} {'Std':>10} {'5%':>10} {'95%':>10}",
            "-" * 72,
        ]
        for name, values in self._samples.items():
            if values.ndim == 1:
                mean = float(jnp.mean(values))
                std = float(jnp.std(values))
                q5 = float(jnp.percentile(values, 5))
                q95 = float(jnp.percentile(values, 95))
                lines.append(f"{name:<30} {mean:>10.4f} {std:>10.4f} {q5:>10.4f} {q95:>10.4f}")
            elif values.ndim >= 2:
                flat = values.reshape(values.shape[0], -1)
                for i in range(flat.shape[1]):
                    label = f"{name}[{i}]"
                    mean = float(jnp.mean(flat[:, i]))
                    std = float(jnp.std(flat[:, i]))
                    q5 = float(jnp.percentile(flat[:, i], 5))
                    q95 = float(jnp.percentile(flat[:, i], 95))
                    lines.append(f"{label:<30} {mean:>10.4f} {std:>10.4f} {q5:>10.4f} {q95:>10.4f}")
        logger.info("\n%s", "\n".join(lines))


def _build_trace_data(
    chain_samples: dict[str, jnp.ndarray],
    max_points: int = 200,
) -> list[dict[str, Any]]:
    """Build thinned trace plot data from chain-level samples.

    Args:
        chain_samples: {param: (n_chains, n_samples, *shape)} from get_samples(group_by_chain=True)
        max_points: Maximum samples per chain in the output.

    Returns:
        List of {parameter, chains: [{chain, values}]} dicts.
        Multi-dimensional params are flattened to indexed scalars.
    """
    traces: list[dict[str, Any]] = []

    for name, arr in chain_samples.items():
        n_chains = arr.shape[0]
        n_samples = arr.shape[1]
        step = max(1, n_samples // max_points)

        if arr.ndim == 2:
            # Scalar parameter: (n_chains, n_samples)
            thinned = arr[:, ::step]
            traces.append(
                {
                    "parameter": name,
                    "chains": [
                        {"chain": int(c), "values": [float(v) for v in thinned[c]]}
                        for c in range(n_chains)
                    ],
                }
            )
        elif arr.ndim >= 3:
            # Multi-dim: flatten to indexed scalars, cap at 12 elements
            flat = arr.reshape(n_chains, n_samples, -1)
            n_elem = min(flat.shape[2], 12)
            for i in range(n_elem):
                thinned = flat[:, ::step, i]
                traces.append(
                    {
                        "parameter": f"{name}[{i}]",
                        "chains": [
                            {"chain": int(c), "values": [float(v) for v in thinned[c]]}
                            for c in range(n_chains)
                        ],
                    }
                )

    return traces


def _build_rank_histograms(
    chain_samples: dict[str, jnp.ndarray],
    n_bins: int = 20,
) -> list[dict[str, Any]]:
    """Build rank histogram data for chain mixing assessment.

    Ranks all samples across chains and bins per chain.
    Uniform histograms indicate good mixing.

    Args:
        chain_samples: {param: (n_chains, n_samples, *shape)}
        n_bins: Number of bins for rank histogram.

    Returns:
        List of {parameter, n_bins, expected_per_bin, chains: [{chain, counts}]} dicts.
    """
    histograms: list[dict[str, Any]] = []

    for name, arr in chain_samples.items():
        if arr.ndim > 2:
            # Skip multi-dim params (too many histograms)
            continue

        n_chains, n_samples = arr.shape[:2]
        total = n_chains * n_samples
        all_vals = arr.reshape(-1)
        ranks = jnp.argsort(jnp.argsort(all_vals)) + 1

        ranks_by_chain = ranks.reshape(n_chains, n_samples)
        chain_hists = []
        for c in range(n_chains):
            hist, _ = jnp.histogram(
                ranks_by_chain[c],
                bins=n_bins,
                range=(1, total + 1),
            )
            chain_hists.append(
                {
                    "chain": int(c),
                    "counts": [int(v) for v in hist],
                }
            )

        histograms.append(
            {
                "parameter": name,
                "n_bins": n_bins,
                "expected_per_bin": float(n_samples / n_bins),
                "chains": chain_hists,
            }
        )

    return histograms


def _param_marginal(name: str, values: jnp.ndarray, n_bins: int = 50) -> dict[str, Any]:
    """Compute histogram-based marginal density for a scalar parameter.

    Args:
        name: Parameter name.
        values: (n_draws,) samples.
        n_bins: Number of bins.

    Returns:
        {parameter, x_values, density, mean, sd, hdi_3, hdi_97}
    """
    v_min, v_max = float(jnp.min(values)), float(jnp.max(values))
    # Slight padding to avoid edge artifacts
    padding = (v_max - v_min) * HIST_PADDING_RATIO if v_max > v_min else HIST_PADDING_DEFAULT
    counts, edges = jnp.histogram(values, bins=n_bins, range=(v_min - padding, v_max + padding))
    # Normalize to density
    bin_width = float(edges[1] - edges[0])
    density = counts / (float(jnp.sum(counts)) * bin_width)
    x_centers = (edges[:-1] + edges[1:]) / 2.0

    # HDI (highest density interval) at 94%
    sorted_vals = jnp.sort(values)
    n = len(sorted_vals)
    ci_size = int(jnp.ceil(0.94 * n))
    if ci_size < n:
        widths = sorted_vals[ci_size:] - sorted_vals[: n - ci_size]
        best = int(jnp.argmin(widths))
        hdi_lo = float(sorted_vals[best])
        hdi_hi = float(sorted_vals[best + ci_size])
    else:
        hdi_lo, hdi_hi = v_min, v_max

    return {
        "parameter": name,
        "x_values": [float(v) for v in x_centers],
        "density": [float(v) for v in density],
        "mean": float(jnp.mean(values)),
        "sd": float(jnp.std(values)),
        "hdi_3": hdi_lo,
        "hdi_97": hdi_hi,
    }


def _build_energy_diagnostics(energy: jnp.ndarray, n_bins: int = 40) -> dict[str, Any]:
    """Build NUTS energy diagnostics (Betancourt 2017).

    Computes marginal energy (E) and energy transition (dE) histograms.
    When the marginal and transition distributions diverge, it indicates
    the sampler is struggling to explore the target geometry.

    Args:
        energy: (n_draws,) or (n_chains, n_draws) NUTS energy values.
        n_bins: Number of histogram bins.

    Returns:
        {energy_hist, energy_transition_hist, bfmi} where each hist has
        {bin_centers, density} and bfmi is the Bayesian Fraction of
        Missing Information per chain.
    """
    e_flat = energy.reshape(-1)

    # Energy transition: dE = E[i+1] - E[i] (per chain for BFMI)
    if energy.ndim == 2:
        de_per_chain = jnp.diff(energy, axis=1)
        de_flat = de_per_chain.reshape(-1)
        # BFMI per chain: Var(dE) / Var(E), should be > 0.3
        bfmi = [
            float(jnp.var(de_per_chain[c]) / jnp.var(energy[c]))
            if float(jnp.var(energy[c])) > 0
            else 0.0
            for c in range(energy.shape[0])
        ]
    else:
        de_flat = jnp.diff(e_flat)
        var_e = float(jnp.var(e_flat))
        bfmi = [float(jnp.var(de_flat) / var_e) if var_e > 0 else 0.0]

    def _hist(vals: jnp.ndarray) -> dict[str, list[float]]:
        lo, hi = float(jnp.min(vals)), float(jnp.max(vals))
        pad = (hi - lo) * HIST_PADDING_RATIO if hi > lo else HIST_PADDING_DEFAULT
        counts, edges = jnp.histogram(vals, bins=n_bins, range=(lo - pad, hi + pad))
        bw = float(edges[1] - edges[0])
        total = float(jnp.sum(counts))
        density = counts / (total * bw) if total > 0 else counts
        centers = (edges[:-1] + edges[1:]) / 2.0
        return {
            "bin_centers": [float(v) for v in centers],
            "density": [float(v) for v in density],
        }

    return {
        "energy_hist": _hist(e_flat),
        "energy_transition_hist": _hist(de_flat),
        "bfmi": bfmi,
    }


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


def fit(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    method: InferenceMethod = "auto",
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
        **kwargs: Method-specific arguments

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    if method == "auto":
        method = select_default_method(model.spec)

    if method == "nuts":
        return _fit_nuts(model, observations, times, **kwargs)
    if method == "nuts_da":
        from causal_ssm_agent.models.ssm.nuts_da import fit_nuts_da

        return fit_nuts_da(model, observations, times, **kwargs)
    if method == "svi":
        return _fit_svi(model, observations, times, **kwargs)
    if method == "hessmc2":
        from causal_ssm_agent.models.ssm.hessmc2 import fit_hessmc2

        return fit_hessmc2(model, observations, times, **kwargs)
    if method == "pgas":
        from causal_ssm_agent.models.ssm.pgas import fit_pgas

        return fit_pgas(model, observations, times, **kwargs)
    if method == "tempered_smc":
        from causal_ssm_agent.models.ssm.tempered_smc import fit_tempered_smc

        return fit_tempered_smc(model, observations, times, **kwargs)
    if method == "laplace_em":
        from causal_ssm_agent.models.ssm.laplace_em import fit_laplace_em

        return fit_laplace_em(model, observations, times, **kwargs)
    if method == "structured_vi":
        from causal_ssm_agent.models.ssm.structured_vi import fit_structured_vi

        return fit_structured_vi(model, observations, times, **kwargs)
    if method == "dpf":
        from causal_ssm_agent.models.ssm.dpf import fit_dpf

        return fit_dpf(model, observations, times, **kwargs)
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
    model_fn = functools.partial(model.model, likelihood_backend=model.make_likelihood_backend())
    blocked_model = handlers.block(model_fn, hide=["log_likelihood"])
    predictive = Predictive(blocked_model, num_samples=num_samples)
    dummy_obs = jnp.zeros((len(times), model.spec.n_manifest))
    return predictive(rng_key, dummy_obs, times)


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
        **kwargs: Additional MCMC arguments

    Returns:
        InferenceResult with NUTS samples
    """
    model_fn = functools.partial(model.model, likelihood_backend=model.make_likelihood_backend())
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

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="nuts",
        diagnostics={"mcmc": mcmc},
    )


def _fit_svi(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
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
        guide_type: Guide family - "normal", "mvn", or "delta"
        num_steps: Number of SVI optimization steps
        num_samples: Number of posterior samples to draw after fitting
        learning_rate: Adam learning rate
        seed: Random seed
        **kwargs: Ignored

    Returns:
        InferenceResult with approximate posterior samples
    """
    model_fn = functools.partial(model.model, likelihood_backend=model.make_likelihood_backend())
    guide_cls = {
        "normal": AutoNormal,
        "mvn": AutoMultivariateNormal,
        "delta": AutoDelta,
    }[guide_type]
    guide = guide_cls(model_fn)

    optimizer = ClippedAdam(step_size=learning_rate)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())

    rng_key = random.PRNGKey(seed)
    svi_result = svi.run(rng_key, num_steps, observations, times)

    # Draw posterior samples from the fitted guide
    sample_key = random.PRNGKey(seed + 1)
    predictive = Predictive(
        model_fn,
        guide=guide,
        params=svi_result.params,
        num_samples=num_samples,
    )
    raw_samples = predictive(sample_key, observations, times)

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
