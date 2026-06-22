"""Runtime shape/dtype checking via the jaxtyping + beartype import hook.

The ``--jaxtyping-packages=...,beartype.beartype`` pytest flag (see
``pyproject.toml``) instruments the listed modules so their jaxtyping
annotations are enforced at call time. These tests confirm the hook is live for
each instrumented cluster: consistent shapes pass, while named-axis and dtype
violations raise a beartype error at the function boundary (before the body
runs). They also gate the convention that an instrumented module omits
``from __future__ import annotations`` so beartype can resolve its annotations.
"""

import jax.numpy as jnp
import jaxtyping
import pytest

from nof1_causal_lab.models.ssm.covariance_utils import (
    inflate_missing_variance,
    symmetrize,
)
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


def test_covariance_consistent_shapes_pass():
    out = symmetrize(jnp.eye(3))
    assert out.shape == (3, 3)


def test_covariance_named_axis_mismatch_raises():
    # cov binds the "N" axis to 3; mask binds it to 2 -> inconsistent N.
    with pytest.raises(jaxtyping.TypeCheckError):
        inflate_missing_variance(jnp.eye(3), jnp.ones((2,)))


def test_covariance_dtype_mismatch_raises():
    # Float[Array, "*batch N N"] rejects an integer-dtyped covariance.
    with pytest.raises(jaxtyping.TypeCheckError):
        symmetrize(jnp.eye(3, dtype=jnp.int32))
