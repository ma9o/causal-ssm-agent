"""Pytest-only assertions for SSM recovery checks.

Construction helpers (block/spec builders, support masks, dynamics specs, the
prior-registry helper) live in ``nof1_causal_lab.models.ssm.testing`` so non-test
code (e.g. the benchmark scripts) can reuse them without importing the test
package.
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
