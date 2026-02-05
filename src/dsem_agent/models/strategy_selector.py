"""Rule-based inference strategy selection.

Selects the appropriate state-space marginalization backend based on
model structure (linearity of dynamics, distribution families).

Decision tree:
| Dynamics        | Observations | Strategy | Backend  |
|-----------------|--------------|----------|----------|
| Linear          | Gaussian     | Kalman   | dynamax  |
| Mildly nonlinear| Gaussian     | UKF      | dynamax  |
| Nonlinear       | Gaussian     | Particle | cuthbert |
| Linear          | Non-Gaussian | Particle*| cuthbert |
| Nonlinear       | Non-Gaussian | Particle | cuthbert |

*Future: Laplace-EKF or Rao-Blackwellization for linear + non-Gaussian

See docs/modeling/inference-strategies.md for theoretical background.
"""

import re
from enum import Enum
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from dsem_agent.models.ssm.model import SSMSpec


class InferenceStrategy(Enum):
    """Available inference strategies for state-space models."""

    KALMAN = "kalman"  # Exact: dynamax Kalman filter
    UKF = "ukf"  # Approximate: dynamax unscented Kalman filter
    PARTICLE = "particle"  # General: cuthbert particle filter


# Pattern to detect state-dependent terms in expressions
# Matches: state#eta1, state#latent_2, ss_eta1, etc.
STATE_DEPENDENT_PATTERN = re.compile(
    r"\bstate#\w+|"  # state#eta1 style
    r"\bss_\w+|"  # ss_level style (state-space variable)
    r"\beta_\d+\b"  # eta_1 when used as state reference
)

# Known nonlinear function calls that break linearity
NONLINEAR_FUNCTIONS = frozenset(
    ["exp", "log", "sin", "cos", "tan", "tanh", "sigmoid", "softmax", "abs", "sqrt"]
)


def select_strategy(spec: "SSMSpec") -> InferenceStrategy:
    """Select inference strategy based on model specification.

    Pure rule-based selection on model structure:
    1. Check if dynamics are linear (drift has no state-dependent terms)
    2. Check if process noise is Gaussian
    3. Check if measurement is linear (lambda has no state-dependent terms)
    4. Check if observation noise is Gaussian

    Args:
        spec: Model specification (SSMSpec)

    Returns:
        InferenceStrategy enum value

    Example:
        >>> from dsem_agent.models.ssm import SSMSpec, NoiseFamily
        >>> spec = SSMSpec(n_latent=2, n_manifest=2)
        >>> select_strategy(spec)
        <InferenceStrategy.KALMAN: 'kalman'>

        >>> spec_nongaussian = SSMSpec(
        ...     n_latent=2, n_manifest=2,
        ...     manifest_dist=NoiseFamily.POISSON
        ... )
        >>> select_strategy(spec_nongaussian)
        <InferenceStrategy.PARTICLE: 'particle'>
    """
    from dsem_agent.models.ssm.model import NoiseFamily

    # Check dynamics linearity
    linear_dynamics = not _has_state_dependent_terms(spec.drift)

    # Check distribution families
    gaussian_process_noise = spec.diffusion_dist == NoiseFamily.GAUSSIAN
    gaussian_observation = spec.manifest_dist == NoiseFamily.GAUSSIAN

    # Check measurement linearity
    linear_measurement = not _has_state_dependent_terms(spec.lambda_mat)

    # Decision tree
    if not gaussian_observation or not gaussian_process_noise:
        # Non-Gaussian noise requires particle filter
        return InferenceStrategy.PARTICLE

    if not linear_measurement:
        # Nonlinear measurement requires particle filter
        return InferenceStrategy.PARTICLE

    if not linear_dynamics:
        # Nonlinear dynamics with Gaussian noise -> UKF
        return InferenceStrategy.UKF

    # Full linear-Gaussian: exact Kalman
    return InferenceStrategy.KALMAN


def _has_state_dependent_terms(matrix_spec) -> bool:
    """Check if matrix specification contains state-dependent terms.

    Used to detect nonlinearity in dynamics or measurement equations.
    State-dependent terms indicate the need for UKF or particle filter.

    Args:
        matrix_spec: Matrix specification. Can be:
            - jnp.ndarray: Constant matrix (linear)
            - "free": Parameter to estimate, constant across states (linear)
            - "diag": Diagonal parameter to estimate (linear)
            - str expression: May contain state references (check for nonlinearity)

    Returns:
        True if matrix contains state-dependent terms (nonlinear)

    Examples:
        >>> _has_state_dependent_terms(jnp.eye(2))
        False
        >>> _has_state_dependent_terms("free")
        False
        >>> _has_state_dependent_terms("drift_eta1 * state#eta1")
        True
        >>> _has_state_dependent_terms("exp(drift_eta1)")
        True
    """
    # Constant matrix - always linear
    if isinstance(matrix_spec, jnp.ndarray):
        return False

    # Standard string specifications - linear
    if matrix_spec in ("free", "diag", None):
        return False

    # String expression - check for state references and nonlinear functions
    if isinstance(matrix_spec, str):
        # Check for state-dependent variable references
        if STATE_DEPENDENT_PATTERN.search(matrix_spec):
            return True

        # Check for nonlinear function calls
        for func in NONLINEAR_FUNCTIONS:
            if re.search(rf"\b{func}\s*\(", matrix_spec):
                return True

    # 2D array of expressions (e.g., matrix of strings)
    if isinstance(matrix_spec, (list, tuple)):
        for row in matrix_spec:
            if isinstance(row, (list, tuple)):
                for elem in row:
                    if _has_state_dependent_terms(elem):
                        return True
            elif _has_state_dependent_terms(row):
                return True

    return False


def _is_mildly_nonlinear(matrix_spec) -> bool:  # noqa: ARG001
    """Check if nonlinearity is mild enough for UKF.

    Mild nonlinearity: smooth functions without discontinuities.
    Strong nonlinearity: discontinuous, highly oscillatory, or
    functions that can cause numerical instability.

    Args:
        matrix_spec: Matrix specification

    Returns:
        True if nonlinearity is mild (UKF appropriate),
        False if strong (particle filter needed)

    Note:
        Currently returns True for any nonlinearity detected.
        Future refinement could classify specific function types.
    """
    # For now, treat all nonlinearity as mild (suitable for UKF)
    # Particle filter fallback happens at runtime if UKF diverges
    return True


def get_likelihood_backend(strategy: InferenceStrategy, **kwargs):
    """Get the likelihood backend for a given strategy.

    Args:
        strategy: InferenceStrategy enum value
        **kwargs: Backend-specific configuration (e.g., n_particles for particle filter)

    Returns:
        Instantiated likelihood backend

    Note:
        UKF and Particle backends are stub implementations that will raise
        NotImplementedError until dynamax/cuthbert dependencies are added.
    """
    from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
    from dsem_agent.models.likelihoods.particle import ParticleLikelihood
    from dsem_agent.models.likelihoods.ukf import UKFLikelihood

    if strategy == InferenceStrategy.KALMAN:
        return KalmanLikelihood()
    elif strategy == InferenceStrategy.UKF:
        return UKFLikelihood(**kwargs)
    elif strategy == InferenceStrategy.PARTICLE:
        return ParticleLikelihood(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
