from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.inference.targets.polya_gamma import (
    build_polya_gamma_observation_plan,
    expected_pg1,
    refresh_polya_gamma_auxiliary_state,
    sample_polya_gamma_integer_shape_devroye,
)


def test_exact_integer_shape_devroye_matches_polya_gamma_mean():
    shape = jnp.asarray([1, 2, 5], dtype=jnp.int32)
    eta = jnp.asarray([0.0, 0.75, 2.0], dtype=jnp.float32)
    tiled_shape = jnp.broadcast_to(shape, (1024, shape.shape[0]))
    tiled_eta = jnp.broadcast_to(eta, tiled_shape.shape)

    draws = sample_polya_gamma_integer_shape_devroye(
        jax.random.PRNGKey(10),
        tiled_shape,
        tiled_eta,
        max_shape=5,
    )

    sample_mean = jnp.mean(draws, axis=0)
    expected_mean = shape.astype(jnp.float32) * expected_pg1(eta)
    assert bool(jnp.all(jnp.abs(sample_mean - expected_mean) < 0.04))


def test_exact_negative_binomial_log_refresh_uses_integer_shape_and_log_mean_offset():
    plan = build_polya_gamma_observation_plan(
        [DistributionFamily.NEGATIVE_BINOMIAL],
        [LinkFunction.LOG],
        num_terms=8,
        sampler="devroye_integer",
        max_integer_shape=8,
    )
    observations = jnp.asarray([[0.0], [3.0], [6.0], [jnp.nan]], dtype=jnp.float32)
    latent = jnp.asarray([[0.1], [0.5], [-0.4], [2.0]], dtype=jnp.float32)
    context = SimpleNamespace(
        H=jnp.asarray([[1.0]], dtype=jnp.float32),
        d_meas=jnp.asarray([0.0], dtype=jnp.float32),
        H_rows=None,
        d_rows=None,
        extra_params={"obs_r": jnp.asarray(2.0, dtype=jnp.float32)},
    )

    observed_counts = observations[:3, 0]
    tiled_observations = jnp.broadcast_to(observed_counts, (1024, observed_counts.shape[0]))
    tiled_eta = jnp.broadcast_to(latent[:3, 0], tiled_observations.shape)
    draws = sample_polya_gamma_integer_shape_devroye(
        jax.random.PRNGKey(23),
        tiled_observations + 2.0,
        tiled_eta - jnp.log(2.0),
        max_shape=8,
    )
    refreshed = refresh_polya_gamma_auxiliary_state(
        jax.random.PRNGKey(23),
        plan,
        context,
        latent,
        observations,
    )

    expected_shape = jnp.asarray([2.0, 5.0, 8.0, 2.0], dtype=jnp.float32)
    expected_psi = latent[:, 0] - jnp.log(2.0)
    sample_mean = jnp.mean(draws, axis=0)
    expected_mean = expected_shape[:3] * expected_pg1(expected_psi[:3])
    assert bool(jnp.all(jnp.abs(sample_mean[:3] - expected_mean) < 0.05))
    assert bool(jnp.allclose(refreshed.shape[:, 0], expected_shape))
    assert bool(jnp.allclose(refreshed.linear_offset[:, 0], -jnp.log(2.0)))
    assert refreshed.gamma_base_terms.shape == (4, 1, 0)
    assert float(refreshed.omega[3, 0]) == pytest.approx(0.0)


def test_exact_negative_binomial_plan_requires_static_integer_shape_bound():
    with pytest.raises(ValueError, match="requires max_integer_shape"):
        build_polya_gamma_observation_plan(
            [DistributionFamily.NEGATIVE_BINOMIAL],
            [LinkFunction.LOG],
            num_terms=8,
            sampler="devroye_integer",
        )
