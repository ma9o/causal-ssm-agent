"""Shared AutoReparam test models."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def simple_normal_model():
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    y = numpyro.sample("y", dist.Normal(x, 0.5))
    numpyro.sample("obs", dist.Normal(y, 0.1), obs=jnp.array(1.0))
