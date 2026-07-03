"""Visualization and JSON-serialization helpers for inference diagnostics.

Functions for converting inference results (posterior samples, MCMC diagnostics,
LOO-CV, energy) into JSON-serializable dicts for the web frontend.

Extracted from inference.py to separate visualization concerns from inference logic.
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Histogram binning edge padding to avoid artifacts at distribution tails
HIST_PADDING_RATIO = 0.05
HIST_PADDING_DEFAULT = 0.5


def build_trace_data(
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


def build_rank_histograms(
    chain_samples: dict[str, jnp.ndarray],
    n_bins: int = 20,
) -> list[dict[str, Any]]:
    """Build rank histogram data for chain mixing assessment.

    Ranks all samples across chains and bins per chain.
    Uniform histograms indicate good mixing.
    """
    histograms: list[dict[str, Any]] = []

    for name, arr in chain_samples.items():
        if arr.ndim > 2:
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


def param_marginal(name: str, values: jnp.ndarray, n_bins: int = 50) -> dict[str, Any]:
    """Compute histogram-based marginal density for a scalar parameter.

    Returns:
        {parameter, x_values, density, mean, sd, hdi_3, hdi_97}
    """
    v_min, v_max = float(jnp.min(values)), float(jnp.max(values))
    padding = (v_max - v_min) * HIST_PADDING_RATIO if v_max > v_min else HIST_PADDING_DEFAULT
    counts, edges = jnp.histogram(values, bins=n_bins, range=(v_min - padding, v_max + padding))
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


def build_energy_diagnostics(energy: jnp.ndarray, n_bins: int = 40) -> dict[str, Any]:
    """Build Hamiltonian energy diagnostics (Betancourt 2017).

    Computes marginal energy (E) and energy transition (dE) histograms.
    """
    e_flat = energy.reshape(-1)

    if energy.ndim == 2:
        de_per_chain = jnp.diff(energy, axis=1)
        de_flat = de_per_chain.reshape(-1)
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


def compute_posterior_marginals(
    samples: dict[str, jnp.ndarray], n_bins: int = 50
) -> list[dict[str, Any]]:
    """Compute marginal posterior density data for visualization."""
    marginals: list[dict[str, Any]] = []

    for name, values in samples.items():
        if values.ndim == 1:
            marginals.append(param_marginal(name, values, n_bins))
        elif values.ndim >= 2:
            flat = values.reshape(values.shape[0], -1)
            n_elem = min(flat.shape[1], 20)
            for i in range(n_elem):
                label = f"{name}[{i}]"
                marginals.append(param_marginal(label, flat[:, i], n_bins))

    return marginals


def compute_posterior_pairs(
    samples: dict[str, jnp.ndarray],
    diagnostics: dict,
    max_params: int = 6,
    max_samples: int = 200,
) -> list[dict[str, Any]]:
    """Compute pairwise scatter data for joint posterior visualization."""
    scalars: list[tuple[str, jnp.ndarray]] = []
    for name, values in samples.items():
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

    div_mask: list[bool] | None = None
    mcmc = diagnostics.get("mcmc")
    if mcmc is not None:
        try:
            extra = mcmc.get_extra_fields()
            if "diverging" in extra:
                div_flat = extra["diverging"].reshape(-1)
                div_mask = [bool(v) for v in div_flat[::step]]
        except (AttributeError, ValueError, RuntimeError):
            logger.warning(
                "Divergence mask extraction failed; pair plots will not show divergent transitions",
                exc_info=True,
            )

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


def format_summary(samples: dict[str, jnp.ndarray], method: str) -> str:
    """Format summary statistics for posterior samples."""
    lines = [
        f"Inference method: {method}",
        f"{'Parameter':<30} {'Mean':>10} {'Std':>10} {'5%':>10} {'95%':>10}",
        "-" * 72,
    ]
    for name, values in samples.items():
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
    return "\n".join(lines)
