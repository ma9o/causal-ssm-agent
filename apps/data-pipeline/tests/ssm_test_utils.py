"""Pytest-only assertions for SSM recovery checks.

SSM construction fixtures live alongside this module in ``ssm_spec_fixtures``.
"""

from __future__ import annotations

import jax.numpy as jnp


def assert_recovery_ci(
    samples: jnp.ndarray,
    true_value: float,
    param_name: str,
    transform=None,
    q_low: float = 5.0,
    q_high: float = 95.0,
) -> None:
    """Assert that a true parameter value falls inside a posterior percentile interval."""
    if transform is not None:
        samples = transform(samples)
    lo = float(jnp.percentile(samples, q_low))
    hi = float(jnp.percentile(samples, q_high))
    assert lo <= true_value <= hi, (
        f"{param_name} {true_value:.2f} outside {q_high - q_low:.0f}% CI [{lo:.3f}, {hi:.3f}]"
    )
