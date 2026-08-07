"""Parity tests for shared predictor- and mean-space observation draws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily
from nof1_causal_lab.models.ssm.execution.emissions import get_mean_param_sample_fn
from nof1_causal_lab.models.ssm.execution.observation_families import FAMILY_REGISTRY
from nof1_causal_lab.models.ssm.execution.observation_sampling import (
    sample_negative_binomial_from_mean,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.contracts import LikelihoodExtraParams


@pytest.mark.parametrize(
    ("family", "link", "predictor", "mean", "extra_params", "std"),
    [
        ("student_t", "identity", 0.4, 0.4, {"obs_df": 4.0}, 0.7),
        ("poisson", "log", 0.4, float(jnp.exp(0.4)), {}, 1.0),
        ("gamma", "log", 0.4, float(jnp.exp(0.4)), {"obs_shape": 2.0}, 1.0),
        ("gamma", "inverse", 2.0, 0.5, {"obs_shape": 2.0}, 1.0),
        ("bernoulli", "logit", 0.4, float(jax.nn.sigmoid(0.4)), {}, 1.0),
        (
            "bernoulli",
            "probit",
            0.4,
            float(jax.scipy.stats.norm.cdf(0.4)),
            {},
            1.0,
        ),
        (
            "negative_binomial",
            "log",
            0.4,
            float(jnp.exp(0.4)),
            {"obs_r": 3.0},
            1.0,
        ),
        (
            "beta",
            "logit",
            0.4,
            float(jax.nn.sigmoid(0.4)),
            {"obs_concentration": 8.0},
            1.0,
        ),
        (
            "beta",
            "probit",
            0.4,
            float(jax.scipy.stats.norm.cdf(0.4)),
            {"obs_concentration": 8.0},
            1.0,
        ),
    ],
)
def test_predictor_and_mean_samplers_use_the_same_draw(
    family: str,
    link: str,
    predictor: float,
    mean: float,
    extra_params: LikelihoodExtraParams,
    std: float,
) -> None:
    key = jax.random.PRNGKey(42)
    point_fn = FAMILY_REGISTRY[DistributionFamily(family)].posterior_predictive_fns[link]
    point_draw = point_fn(
        jnp.asarray(predictor),
        key,
        jnp.asarray(std),
        jnp.asarray(extra_params.get("obs_df", 4.0)),
        jnp.asarray(extra_params.get("obs_shape", 2.0)),
        jnp.asarray(extra_params.get("obs_r", 3.0)),
        jnp.asarray(extra_params.get("obs_concentration", 8.0)),
        jnp.asarray(1),
        jnp.zeros(1),
        jnp.zeros(1),
        jnp.zeros(1),
    )
    mean_draw = get_mean_param_sample_fn(family, extra_params)(
        key,
        jnp.asarray([mean]),
        jnp.asarray([[std**2]]),
    )[0]

    np.testing.assert_array_equal(point_draw, mean_draw)


@pytest.mark.parametrize(
    ("family", "extra_params", "invalid_mean"),
    [
        ("poisson", {}, -1.0),
        ("gamma", {"obs_shape": 2.0}, 0.0),
        ("bernoulli", {}, 1.1),
        ("negative_binomial", {"obs_r": 3.0}, -1.0),
        ("beta", {"obs_concentration": 8.0}, 0.0),
    ],
)
def test_mean_samplers_surface_invalid_domains_as_nan(
    family: str,
    extra_params: LikelihoodExtraParams,
    invalid_mean: float,
) -> None:
    draw = get_mean_param_sample_fn(family, extra_params)(
        jax.random.PRNGKey(0),
        jnp.asarray([invalid_mean]),
        jnp.eye(1),
    )

    assert jnp.isnan(draw[0])


def test_zero_mean_negative_binomial_uses_an_exact_zero_poisson_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_rates: list[jnp.ndarray] = []

    def capture_rate(_key: jax.Array, rate: jnp.ndarray) -> jnp.ndarray:
        captured_rates.append(rate)
        return jnp.zeros_like(rate)

    monkeypatch.setattr(jax.random, "poisson", capture_rate)

    draw = sample_negative_binomial_from_mean(
        jax.random.PRNGKey(3),
        jnp.asarray([0.0, 2.0]),
        4.0,
    )

    np.testing.assert_array_equal(draw, [0.0, 0.0])
    assert captured_rates
    assert captured_rates[0][0] == 0.0
    assert captured_rates[0][1] > 0.0
