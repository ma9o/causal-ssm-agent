"""Pilot: runtime shape/dtype checking via the jaxtyping + beartype import hook.

The ``--jaxtyping-packages=...targets.euler_maruyama,beartype.beartype`` pytest
flag (see ``pyproject.toml``) instruments that one module so its jaxtyping
annotations are enforced at call time. These tests confirm the hook is live:
consistent shapes pass, while named-axis and dtype violations raise a beartype
error at the function boundary (before the body runs).
"""

import jax.numpy as jnp
import jaxtyping
import pytest

from nof1_causal_lab.models.ssm.inference.targets.euler_maruyama import (
    _log_prob_with_chol,
)


def test_consistent_shapes_pass():
    d = 3
    out = _log_prob_with_chol(jnp.zeros((d,)), jnp.zeros((d,)), jnp.eye(d))
    assert out.shape == ()


def test_named_axis_mismatch_raises():
    # value/mean bind the "D" axis to 3; chol is (2, 2) -> inconsistent D.
    with pytest.raises(jaxtyping.TypeCheckError):
        _log_prob_with_chol(jnp.zeros((3,)), jnp.zeros((3,)), jnp.eye(2))


def test_dtype_mismatch_raises():
    # Float[Array, " D"] rejects an integer-dtyped array.
    with pytest.raises(jaxtyping.TypeCheckError):
        _log_prob_with_chol(jnp.zeros((3,), dtype=jnp.int32), jnp.zeros((3,)), jnp.eye(3))
