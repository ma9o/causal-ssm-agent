"""Result types for SSM diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

from nof1_causal_lab.flows import get_prefect_logger

logger = get_prefect_logger(__name__)


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
