"""Result types for SSM parametric diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.ssm.inference.targets.base import NUMERICAL_EPSILON

logger = get_prefect_logger(__name__)


def _chi_squared_uniformity_pvalue(ranks: jnp.ndarray, max_rank: int, n_bins: int) -> float:
    """Chi-squared uniformity test on discrete rank statistics."""
    ranks = jnp.asarray(ranks, dtype=jnp.float64)
    n = ranks.shape[0]
    bin_width = (max_rank + 1) / n_bins
    bin_idx = jnp.clip((ranks / bin_width).astype(jnp.int64), 0, n_bins - 1)
    observed = jnp.array([float(jnp.sum(bin_idx == i)) for i in range(n_bins)], dtype=jnp.float64)
    expected = float(n) / n_bins
    chi2 = jnp.sum((observed - expected) ** 2 / jnp.maximum(expected, NUMERICAL_EPSILON))
    df = n_bins - 1
    return float(1.0 - jax.scipy.special.gammainc(df / 2.0, chi2 / 2.0))


@dataclass
class OutputSensitivityResult:
    """Results from output sensitivity analysis (pre-inference identifiability)."""

    singular_values: list[float]
    normalized_singular_values: list[float]
    deficiency_count: int
    weak_directions: list[dict]
    per_parameter: list[dict]
    n_draws: int
    n_observations: int
    n_parameters: int
    parameter_names: list[str] = field(default_factory=list)
    normalized_right_singular_vectors: list[list[float]] = field(default_factory=list)

    def print_report(self) -> None:
        """Log a human-readable sensitivity analysis report."""
        n_nonsing = sum(1 for sv in self.singular_values if sv > NUMERICAL_EPSILON)
        n_weak = len(self.weak_directions)
        lines = [
            "=== Output Sensitivity Analysis ===",
            f"  Parameters: {self.n_parameters}, Observations: {self.n_observations}",
            f"  Deficient directions: {self.deficiency_count}/{self.n_parameters}",
            f"  Weak directions (<=10): {n_weak}/{self.n_parameters}",
            f"  Prior draws: {self.n_draws}",
            f"  Rank: {n_nonsing}/{min(self.n_observations, self.n_parameters)}",
        ]
        for entry in self.per_parameter:
            tag = "[ok]" if entry["identifiable"] else "[!]"
            lines.append(f"  {tag} {entry['parameter']}: norm={entry['sensitivity_norm']:.4f}")
        logger.info("\n%s", "\n".join(lines))


class OutputSensitivityUnsupportedError(ValueError):
    """Raised when the observation-space sensitivity map is not valid for a model."""


@dataclass
class ProfileLikelihoodResult:
    """Results from profile likelihood identifiability analysis."""

    parameter_profiles: dict[str, dict]
    mle_ll: float
    mle_params: dict[str, jnp.ndarray]
    threshold: float
    parameter_names: list[str]

    def summary(self) -> dict[str, str]:
        """Per-parameter classification based on profile shape."""
        eps = 0.5
        classifications = {}
        for name, prof in self.parameter_profiles.items():
            pll = jnp.asarray(prof["profile_ll"])
            pll_max = float(jnp.max(pll))
            ref = max(pll_max, self.mle_ll)
            ratio = pll - ref
            ll_range = float(pll_max - jnp.min(pll))

            if ll_range < eps:
                classifications[name] = "structurally_unidentifiable"
                continue

            peak = int(jnp.argmax(pll))
            left = ratio[:peak] if peak > 0 else jnp.array([0.0])
            right = ratio[peak + 1 :] if peak < len(pll) - 1 else jnp.array([0.0])
            left_ok = bool(jnp.any(left < -self.threshold))
            right_ok = bool(jnp.any(right < -self.threshold))

            if left_ok and right_ok:
                classifications[name] = "identified"
            else:
                classifications[name] = "practically_unidentifiable"

        return classifications

    def print_report(self) -> None:
        """Log a human-readable profile likelihood report."""
        summary = self.summary()
        markers = {
            "identified": "[ok]",
            "practically_unidentifiable": "[~]",
            "structurally_unidentifiable": "[!]",
        }
        lines = [
            "=== Profile Likelihood Report ===",
            f"  Parameters profiled: {len(self.parameter_profiles)}",
            f"  Threshold: {self.threshold:.2f}",
            f"  MAP log-posterior: {self.mle_ll:.2f}",
        ]
        for name, cls in summary.items():
            lines.append(f"  {markers.get(cls, '[?]')} {name}: {cls}")
        logger.info("\n%s", "\n".join(lines))


@dataclass
class MAPOptimizationRun:
    """One multi-start MAP optimization run."""

    index: int
    start_kind: str
    start_log_posterior: float
    log_posterior: float
    log_likelihood: float
    log_prior: float
    objective: float
    success: bool
    status: int
    message: str
    n_iters: int
    n_function_evals: int
    grad_norm: float
    distance_to_best: float = 0.0


@dataclass
class MAPCurvatureResult:
    """Local curvature summary for one Hessian family at the MAP."""

    eigenvalues: list[float]
    normalized_eigenvalues: list[float]
    negative_direction_count: int
    deficiency_count: int
    positive_definite: bool
    condition_number: float | None
    normalized_condition_number: float | None
    weak_directions: list[dict]
    per_parameter: list[dict]
    parameter_names: list[str] = field(default_factory=list)
    eigenvectors_normalized: list[list[float]] = field(default_factory=list)


@dataclass
class MAPGeometryResult:
    """Dataset-conditioned local geometry around the selected MAP."""

    n_starts: int
    n_successful_starts: int
    best_start_index: int
    map_log_posterior: float
    map_log_likelihood: float
    map_log_prior: float
    final_grad_norm: float
    runner_up_objective_gap: float | None
    starts: list[MAPOptimizationRun]
    likelihood_curvature: MAPCurvatureResult
    posterior_curvature: MAPCurvatureResult
    prior_rescued_parameters: list[str]
    boundary_parameters: list[str]
    z_map_unconstrained: list[float] = field(default_factory=list)
    prior_std_unconstrained: list[float] = field(default_factory=list)

    def print_report(self) -> None:
        """Log a human-readable MAP-geometry report."""
        lines = [
            "=== MAP Geometry Report ===",
            (
                "  Starts: "
                f"{self.n_successful_starts}/{self.n_starts} successful, "
                f"best_start={self.best_start_index}"
            ),
            (
                "  Best mode: "
                f"log_posterior={self.map_log_posterior:.4f} "
                f"log_likelihood={self.map_log_likelihood:.4f} "
                f"log_prior={self.map_log_prior:.4f} "
                f"grad_norm={self.final_grad_norm:.3e}"
            ),
            (
                "  H_lik: "
                f"negative_dirs={self.likelihood_curvature.negative_direction_count} "
                f"weak_dirs={self.likelihood_curvature.deficiency_count}"
            ),
            (
                "  H_post: "
                f"negative_dirs={self.posterior_curvature.negative_direction_count} "
                f"weak_dirs={self.posterior_curvature.deficiency_count}"
            ),
            f"  Prior-rescued parameters: {len(self.prior_rescued_parameters)}",
            f"  Boundary parameters: {len(self.boundary_parameters)}",
        ]
        logger.info("\n%s", "\n".join(lines))


@dataclass
class SBCResult:
    """Results from simulation-based calibration (Modrak et al. 2023)."""

    ranks: dict[str, jnp.ndarray]
    likelihood_ranks: jnp.ndarray
    n_sbc: int
    n_posterior_samples: int
    parameter_names: list[str]
    n_failed: int = 0
    n_attempted: int = 0

    def summary(self) -> dict[str, dict]:
        """Per-parameter uniformity test over rank statistics."""
        result = {}
        n_bins = max(5, int(self.n_sbc**0.5))
        for name, ranks in self.ranks.items():
            p_value = _chi_squared_uniformity_pvalue(ranks, self.n_posterior_samples, n_bins)
            result[name] = {
                "p_value": p_value,
                "uniform": p_value > 0.01,
                "mean_rank": float(jnp.mean(ranks)),
                "expected_mean": self.n_posterior_samples / 2.0,
            }
        ll_p_value = _chi_squared_uniformity_pvalue(
            self.likelihood_ranks,
            self.n_posterior_samples,
            n_bins,
        )
        result["_likelihood"] = {"p_value": ll_p_value, "uniform": ll_p_value > 0.01}
        return result

    def print_report(self) -> None:
        """Log a human-readable SBC report."""
        summary = self.summary()
        lines = [f"=== SBC Calibration Report (n={self.n_sbc}) ==="]
        if self.n_failed > 0:
            lines.append(
                f"  Replicates: {self.n_sbc} succeeded, {self.n_failed} failed "
                f"out of {self.n_attempted} attempted"
            )
        for name, info in summary.items():
            tag = "ok" if info["uniform"] else "FAIL"
            if name == "_likelihood":
                lines.append(f"  [{tag}] likelihood: p={info['p_value']:.4f}")
            else:
                lines.append(
                    f"  [{tag}] {name}: p={info['p_value']:.4f} (mean_rank={info['mean_rank']:.1f})"
                )
        logger.info("\n%s", "\n".join(lines))


@dataclass
class PowerScalingResult:
    """Results from post-fit power-scaling sensitivity analysis."""

    prior_sensitivity: dict[str, float]
    likelihood_sensitivity: dict[str, float]
    diagnosis: dict[str, str]
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
