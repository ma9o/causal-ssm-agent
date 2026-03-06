"""Shared test helpers (non-fixtures).

These are utilities that can be imported directly into test modules.
For fixtures, see conftest.py.
"""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class MockPrediction:
    """Mock DSPy prediction object for scoring tests."""

    structure: str


def make_mock_generate(responses: list[str]):
    """Create a mock generate function that returns predefined responses.

    Args:
        responses: List of JSON strings to return in order

    Returns:
        Async function matching OrchestratorGenerateFn signature
    """
    call_count = [0]  # Use list to allow mutation in closure

    async def mock_generate(
        messages: list, tools: list | None, follow_ups: list[str] | None
    ) -> str:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    return mock_generate


def assert_recovery_ci(
    samples: jnp.ndarray,
    true_value: float,
    param_name: str,
    transform=None,
    q_low: float = 5.0,
    q_high: float = 95.0,
):
    """Assert that true_value falls within the [q_low, q_high] percentile CI.

    Args:
        samples: 1D array of posterior samples.
        true_value: Ground truth value.
        param_name: Name for error message.
        transform: Optional transform to apply to samples (e.g. lambda s: -jnp.abs(s)).
        q_low: Lower percentile (default 5 for 90% CI).
        q_high: Upper percentile (default 95 for 90% CI).
    """
    if transform is not None:
        samples = transform(samples)
    lo = float(jnp.percentile(samples, q_low))
    hi = float(jnp.percentile(samples, q_high))
    assert lo <= true_value <= hi, (
        f"{param_name} {true_value:.2f} outside {q_high - q_low:.0f}% CI [{lo:.3f}, {hi:.3f}]"
    )
