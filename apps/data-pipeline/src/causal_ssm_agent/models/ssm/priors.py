"""Leaf prior definitions for the state-space model runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from causal_ssm_agent.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_family_index,
)
from causal_ssm_agent.models.ssm.parameter_names import INITIAL_STATE_CORRELATION_PRIOR_DEFAULTS


@dataclass
class SSMPriors:
    """Prior specifications for state-space model parameters."""

    drift_diag: dict = field(default_factory=lambda: {"mu": -0.5, "sigma": 1.0})
    drift_offdiag: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 0.5})

    diffusion_diag: dict = field(default_factory=lambda: {"sigma": 1.0})
    diffusion_offdiag: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 0.5})

    cint: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})
    static_state_sd: dict = field(default_factory=lambda: {"sigma": 1.0})

    lambda_free: dict = field(default_factory=lambda: {"mu": 0.5, "sigma": 0.5})

    manifest_means: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 2.0})
    manifest_var_diag: dict = field(default_factory=lambda: {"sigma": 1.0})

    obs_df: dict = field(
        default_factory=lambda: {
            "family": get_positive_runtime_family_index(PriorDistributionFamily.GAMMA),
            "concentration": 5.0,
            "rate": 1.0,
        }
    )
    obs_shape: dict = field(
        default_factory=lambda: {
            "family": get_positive_runtime_family_index(PriorDistributionFamily.GAMMA),
            "concentration": 2.0,
            "rate": 1.0,
        }
    )
    obs_r: dict = field(
        default_factory=lambda: {
            "family": get_positive_runtime_family_index(PriorDistributionFamily.GAMMA),
            "concentration": 2.0,
            "rate": 0.5,
        }
    )
    obs_concentration: dict = field(
        default_factory=lambda: {
            "family": get_positive_runtime_family_index(PriorDistributionFamily.GAMMA),
            "concentration": 5.0,
            "rate": 0.5,
        }
    )
    obs_ordered_base: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})
    obs_ordered_gaps: dict = field(default_factory=lambda: {"sigma": 1.0})
    obs_cat_intercepts: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})
    obs_cat_slopes: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})

    t0_means: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 2.0})
    t0_var_diag: dict = field(default_factory=lambda: {"sigma": 2.0})
    t0_var_offdiag: dict = field(
        default_factory=lambda: dict(INITIAL_STATE_CORRELATION_PRIOR_DEFAULTS)
    )
