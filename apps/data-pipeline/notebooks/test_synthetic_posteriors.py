"""Correctness tests for ``synthetic_posteriors``: roundtrips, log-det-jac, normalisation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthetic_posteriors import (
    Bend,
    Cauchy,
    Chain,
    Funnel,
    Gaussian,
    Laplace,
    Logit,
    Mirror,
    Mixture,
    Shear,
    Shift,
    SoftConstraint,
    Softplus,
    StudentT,
    TransformedTarget,
    invariance,
)

jax.config.update("jax_enable_x64", True)


# ----- base distributions -----


def test_gaussian_log_prob_matches_scipy():
    g = Gaussian(dim=2, loc=0.0, scale=1.5)
    x = jnp.array([[0.3, -0.8], [1.1, 2.0]])
    from scipy.stats import norm as sp_norm

    expected = sp_norm.logpdf(np.asarray(x), loc=0.0, scale=1.5).sum(axis=-1)
    assert np.allclose(np.asarray(g.log_prob(x)), expected, atol=1e-10)


def test_base_samples_have_expected_moments():
    key = jax.random.PRNGKey(0)
    for base, expected_var in [
        (Gaussian(dim=2, scale=2.0), 4.0),
        (Laplace(dim=2, scale=1.0), 2.0),
    ]:
        samples = base.sample(key, 100_000)
        assert samples.shape == (100_000, 2)
        assert abs(float(jnp.mean(samples))) < 0.05
        assert abs(float(jnp.var(samples)) - expected_var) < 0.1


# ----- bijector roundtrip -----


@pytest.mark.parametrize(
    "bijector",
    [
        Shear(theta=0.6, scale=(1.5, 0.7)),
        Shift(offset=(0.7, -1.3)),
        Bend(f=lambda x: 0.5 * x**2),
        Funnel(g=lambda x: 0.4 * x),
        Softplus(axes=(0,)),
        Logit(axes=(1,)),
        Chain((Bend(f=lambda x: 0.3 * x**2), Funnel(g=lambda x: 0.2 * x), Shear(theta=0.3))),
    ],
)
def test_bijector_roundtrip(bijector):
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.normal(size=(16, 2)))
    # clip to softplus/logit domains where relevant
    if isinstance(bijector, Logit):
        u = u  # any real is fine for forward
    x = bijector.forward(u)
    u_rec = bijector.inverse(x)
    assert np.allclose(np.asarray(u), np.asarray(u_rec), atol=1e-6)


# ----- log-det-jac vs autodiff -----


def _numerical_log_det_jac(forward_fn, u: jnp.ndarray) -> jnp.ndarray:
    jac = jax.jacfwd(forward_fn)(u)
    return jnp.linalg.slogdet(jac)[1]


@pytest.mark.parametrize(
    "bijector",
    [
        Shear(theta=0.35, scale=(1.3, 0.8)),
        Shift(offset=(1.0, -0.5)),
        Bend(f=lambda x: 0.5 * x**2 - 0.2 * x),
        Funnel(g=lambda x: 0.5 * jnp.sin(x)),
        Softplus(axes=(0,)),
        Logit(axes=(0, 1)),
    ],
)
def test_bijector_log_det_jac_matches_autodiff(bijector):
    rng = np.random.default_rng(2)
    for _ in range(4):
        u = jnp.asarray(rng.normal(size=(2,)))
        expected = _numerical_log_det_jac(bijector.forward, u)
        got = bijector.forward_log_det_jac(u)
        assert np.allclose(np.asarray(got), np.asarray(expected), atol=1e-6), (
            f"{type(bijector).__name__}: expected {expected}, got {got}"
        )


def test_chain_log_det_jac_matches_autodiff():
    chain = Chain(
        (
            Bend(f=lambda x: 0.4 * x**2),
            Funnel(g=lambda x: 0.3 * x),
            Shear(theta=0.4, scale=(1.2, 0.9)),
        )
    )
    rng = np.random.default_rng(3)
    for _ in range(4):
        u = jnp.asarray(rng.normal(size=(2,)))
        expected = _numerical_log_det_jac(chain.forward, u)
        got = chain.forward_log_det_jac(u)
        assert np.allclose(np.asarray(got), np.asarray(expected), atol=1e-6)


# ----- inverse-log-det-jac self-consistency -----


@pytest.mark.parametrize(
    "bijector",
    [
        Shear(theta=0.3, scale=(1.1, 0.7)),
        Bend(f=lambda x: 0.4 * jnp.sin(x)),
        Funnel(g=lambda x: 0.3 * x),
    ],
)
def test_inverse_log_det_jac_negates_forward(bijector):
    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.normal(size=(2,)))
    x = bijector.forward(u)
    forward_ldj = bijector.forward_log_det_jac(u)
    inverse_ldj = bijector.inverse_log_det_jac(x)
    assert np.allclose(float(forward_ldj), -float(inverse_ldj), atol=1e-6)


# ----- transformed target normalisation -----


def _integrate_on_grid(target, a_range=(-6.0, 6.0), b_range=(-6.0, 6.0), n=300) -> float:
    a = jnp.linspace(a_range[0], a_range[1], n)
    b = jnp.linspace(b_range[0], b_range[1], n)
    A, B = jnp.meshgrid(a, b, indexing="xy")
    grid = jnp.stack([A, B], axis=-1)
    log_p = target.log_prob(grid)
    da = float(a[1] - a[0])
    db = float(b[1] - b[0])
    return float(jnp.exp(log_p).sum() * da * db)


def test_transformed_target_normalises_to_one():
    target = TransformedTarget(
        base=Gaussian(dim=2, scale=1.0),
        bijector=Chain(
            (Bend(f=lambda x: 0.4 * x**2), Shear(theta=0.3, scale=(1.2, 0.9)))
        ),
    )
    mass = _integrate_on_grid(target)
    assert abs(mass - 1.0) < 0.01


def test_mirror_target_normalises_to_one():
    # shift base so reflection produces two distinct modes
    shifted = TransformedTarget(
        base=Gaussian(dim=2, loc=0.0, scale=0.6),
        bijector=Bend(f=lambda x: 1.5 * jnp.ones_like(x)),  # constant shift in b
    )
    mirror = Mirror(shifted, flip_axes=(0,))
    mass = _integrate_on_grid(mirror)
    assert abs(mass - 1.0) < 0.01


def test_mixture_normalises_and_has_expected_log_prob():
    c1 = TransformedTarget(Gaussian(dim=2, loc=0.0, scale=0.5), Shear(theta=0.0))
    c2 = TransformedTarget(
        Gaussian(dim=2, scale=0.5),
        Bend(f=lambda x: 2.0 * jnp.ones_like(x)),  # offset in axis 1
    )
    mix = Mixture(components=(c1, c2), weights=(0.3, 0.7))
    mass = _integrate_on_grid(mix, a_range=(-5.0, 5.0), b_range=(-5.0, 7.0))
    assert abs(mass - 1.0) < 0.01

    # log_prob at a point equals logsumexp of weighted component log-probs
    x = jnp.array([0.1, 1.9])
    lp_mix = float(mix.log_prob(x))
    lp_c1 = float(c1.log_prob(x))
    lp_c2 = float(c2.log_prob(x))
    expected = float(
        jax.scipy.special.logsumexp(jnp.array([jnp.log(0.3) + lp_c1, jnp.log(0.7) + lp_c2]))
    )
    assert abs(lp_mix - expected) < 1e-8


# ----- Mirror sampling empirically matches log_prob -----


def test_mirror_sampling_matches_density_symmetry():
    base = TransformedTarget(Gaussian(dim=2, loc=0.0, scale=0.4), Bend(f=lambda x: 1.0 + 0.0 * x))
    mirror = Mirror(base, flip_axes=(0,))
    key = jax.random.PRNGKey(42)
    samples = mirror.sample(key, 20_000)
    # modes should be near (±·, 1)
    positive_mass = float(jnp.mean(samples[:, 0] > 0))
    assert 0.45 < positive_mass < 0.55


# ----- soft constraint / invariance -----


def test_soft_constraint_subtracts_penalty():
    base = TransformedTarget(Gaussian(dim=2, scale=1.0), Shear(theta=0.0))
    x = jnp.array([0.3, -0.5])
    constrained = SoftConstraint(base=base, penalty=lambda xy: xy[..., 0] ** 2, weight=0.7)
    expected = float(base.log_prob(x)) - 0.7 * float(x[0] ** 2)
    assert abs(float(constrained.log_prob(x)) - expected) < 1e-10


def test_invariance_enforces_projection():
    base = TransformedTarget(Gaussian(dim=2, scale=3.0), Shear(theta=0.0))
    # hyperbolic ridge: x*y ≈ 1
    target = invariance(base, phi=lambda x: x[..., 0] * x[..., 1], target_value=1.0, tol=0.1)
    # on the manifold the penalty is zero
    on_manifold = float(target.log_prob(jnp.array([2.0, 0.5])))
    off_manifold = float(target.log_prob(jnp.array([2.0, 2.0])))  # xy = 4 ≫ 1
    # off-manifold must be much smaller (dominated by penalty)
    assert on_manifold - off_manifold > 10.0


# ----- StudentT and Cauchy log-prob -----


def test_studentt_and_cauchy_run():
    key = jax.random.PRNGKey(7)
    for base in (StudentT(dim=2, df=3.0, scale=1.0), Cauchy(dim=2, scale=0.5)):
        x = base.sample(key, 128)
        lp = base.log_prob(x)
        assert lp.shape == (128,)
        assert jnp.all(jnp.isfinite(lp))


# ----- compositional sanity: use every primitive in one target -----


def test_everything_composes():
    """A kitchen-sink target: heavy-tail base, banana + funnel + shear, mirror for bimodality,
    hyperbolic invariance factor. Must evaluate without errors and produce finite log_prob."""

    warped = TransformedTarget(
        base=StudentT(dim=2, df=4.0, scale=1.0),
        bijector=Chain(
            (
                Bend(f=lambda x: 0.3 * x**2),
                Funnel(g=lambda x: 0.25 * x),
                Shear(theta=0.3, scale=(1.2, 0.9)),
            )
        ),
    )
    bimodal = Mirror(warped, flip_axes=(0,))
    with_invariance = invariance(
        bimodal, phi=lambda x: x[..., 0] * x[..., 1], target_value=0.5, tol=0.3
    )

    grid = jnp.stack(jnp.meshgrid(jnp.linspace(-2, 2, 20), jnp.linspace(-2, 2, 20)), axis=-1)
    lp = with_invariance.log_prob(grid)
    assert lp.shape == (20, 20)
    assert jnp.all(jnp.isfinite(lp))
