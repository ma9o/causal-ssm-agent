"""Comprehensive tests for AutoReparam: automatic reparameterization strategies.

Test design inspired by:
- pyro-ppl/pyro: tests/infer/reparam/test_strategies.py (trace structure verification)
- pyro-ppl/numpyro: test/infer/test_reparam.py (moment/gradient preservation)
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_allclose
from numpyro import handlers
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.reparam import LocScaleReparam, ProjectedNormalReparam
from numpyro.optim import Adam

from causal_ssm_agent.models.ssm.autoreparam import (
    AutoReparam,
    MinimalReparam,
    _is_unconstrained,
    _loc_scale_reparam,
    _minimal_reparam,
)

# ---------------------------------------------------------------------------
# Helpers (ported from NumPyro's test_reparam.py)
# ---------------------------------------------------------------------------


def get_moments(x):
    """Extract first four central moments from samples."""
    m1 = jnp.mean(x, axis=0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = jnp.mean(xx, axis=0)
    m3 = jnp.mean(xxx, axis=0) / m2**1.5
    m4 = jnp.mean(xxxx, axis=0) / m2**2
    return jnp.stack([m1, m2, m3, m4])


def trace_name_type(model_fn, *args, **kwargs):
    """Trace the model and return [(name, type)] for all sample/deterministic sites.

    Mirrors Pyro's trace_name_is_observed but adapted for NumPyro's trace
    structure where reparameterized sites become 'deterministic' type.
    """
    with handlers.seed(rng_seed=0):
        trace = handlers.trace(model_fn).get_trace(*args, **kwargs)
    return [
        (name, site["type"])
        for name, site in trace.items()
        if site["type"] in ("sample", "deterministic")
    ]


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


def simple_normal_model():
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    y = numpyro.sample("y", dist.Normal(x, 0.5))
    numpyro.sample("obs", dist.Normal(y, 0.1), obs=jnp.array(1.0))


def comprehensive_model():
    """Model exercising the full AutoReparam cascade.

    Covers: Normal (loc-scale, real), LogNormal (TransformedDistribution),
    HalfNormal (no loc, positive), Gamma (positive), StudentT (loc-scale + shape),
    batched Normal, Independent-wrapped Normal, observed sites.
    """
    a = numpyro.sample("a", dist.Normal(0, 1))
    b = numpyro.sample("b", dist.LogNormal(0, 1))
    c = numpyro.sample("c", dist.Normal(a, b))
    d = numpyro.sample("d", dist.HalfNormal(1.0))
    e = numpyro.sample("e", dist.Gamma(2.0, 1.0))
    f = numpyro.sample("f", dist.StudentT(5.0, a, b))
    g = numpyro.sample("g", dist.Normal(jnp.zeros(2), 1.0).to_event(1))
    h = numpyro.sample("h", dist.Normal(0, 1), obs=a)
    return a, b, c, d, e, f, g, h


def projected_normal_model():
    x = numpyro.sample("x", dist.ProjectedNormal(jnp.zeros(3)))
    numpyro.sample("obs", dist.Normal(jnp.sum(x), 0.1), obs=jnp.array(0.5))


def observed_projected_normal_model():
    numpyro.sample(
        "x",
        dist.ProjectedNormal(jnp.zeros(3)),
        obs=jnp.array([1.0, 0.0, 0.0]),
    )


def truncated_normal_model():
    """TruncatedNormal: has loc/scale but constrained support. Should NOT be reparameterized."""
    x = numpyro.sample("x", dist.TruncatedNormal(0.0, 1.0, low=-2.0, high=2.0))
    numpyro.sample("obs", dist.Normal(x, 0.1), obs=jnp.array(0.5))


def plated_model():
    """Model with numpyro.plate."""
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    with numpyro.plate("data", 5):
        numpyro.sample("x", dist.Normal(mu, sigma))


def neals_funnel(dim=10):
    """Neal's funnel: the canonical test for non-centered parameterization."""
    y = numpyro.sample("y", dist.Normal(0, 3))
    with numpyro.plate("D", dim):
        numpyro.sample("x", dist.Normal(0, jnp.exp(y / 2)))


# ---------------------------------------------------------------------------
# I. Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestIsUnconstrained:
    def test_real(self):
        assert _is_unconstrained(dist.constraints.real) is True

    def test_positive(self):
        assert _is_unconstrained(dist.constraints.positive) is False

    def test_unit_interval(self):
        assert _is_unconstrained(dist.constraints.unit_interval) is False

    def test_independent_real(self):
        assert _is_unconstrained(dist.constraints.independent(dist.constraints.real, 1)) is True

    def test_independent_positive(self):
        assert (
            _is_unconstrained(dist.constraints.independent(dist.constraints.positive, 1)) is False
        )


class TestLocScaleReparamHelper:
    def test_normal(self):
        assert isinstance(_loc_scale_reparam("x", dist.Normal(0.0, 1.0), 0.0), LocScaleReparam)

    def test_half_normal_skipped(self):
        assert _loc_scale_reparam("sigma", dist.HalfNormal(1.0), 0.0) is None

    def test_gamma_skipped(self):
        assert _loc_scale_reparam("df", dist.Gamma(5.0, 1.0), 0.0) is None

    def test_laplace(self):
        assert isinstance(_loc_scale_reparam("x", dist.Laplace(0.0, 1.0), 0.5), LocScaleReparam)

    def test_decentered_name_skipped(self):
        assert _loc_scale_reparam("x_decentered", dist.Normal(0.0, 1.0), 0.0) is None

    def test_student_t_shape_params(self):
        result = _loc_scale_reparam("x", dist.StudentT(5.0, 0.0, 1.0), None)
        assert isinstance(result, LocScaleReparam)
        assert "df" in result.shape_params

    def test_truncated_normal_skipped(self):
        """TruncatedNormal has interval support, not real."""
        result = _loc_scale_reparam("x", dist.TruncatedNormal(0.0, 1.0, low=-2.0, high=2.0), 0.0)
        assert result is None

    def test_log_normal_skipped(self):
        """LogNormal has positive support."""
        result = _loc_scale_reparam("x", dist.LogNormal(0.0, 1.0), 0.0)
        assert result is None


class TestMinimalReparamHelper:
    def test_normal_returns_none(self):
        assert _minimal_reparam(dist.Normal(0.0, 1.0), is_observed=False) is None

    def test_projected_normal(self):
        assert isinstance(
            _minimal_reparam(dist.ProjectedNormal(jnp.zeros(3)), is_observed=False),
            ProjectedNormalReparam,
        )

    def test_transformed_with_normal_base(self):
        td = dist.TransformedDistribution(dist.Normal(0.0, 1.0), dist.transforms.ExpTransform())
        assert _minimal_reparam(td, is_observed=False) is None

    def test_observed_projected_normal_returns_none(self):
        assert _minimal_reparam(dist.ProjectedNormal(jnp.zeros(3)), is_observed=True) is None


class TestAutoReparamValidation:
    def test_centered_above_one(self):
        with pytest.raises(ValueError, match="centered must be in"):
            AutoReparam(centered=1.5)

    def test_centered_negative(self):
        with pytest.raises(ValueError, match="centered must be in"):
            AutoReparam(centered=-0.1)


# ---------------------------------------------------------------------------
# II. Trace structure (Pyro-style trace_name_is_observed)
# ---------------------------------------------------------------------------


class TestTraceStructure:
    """Verify exact trace structure after reparameterization.

    Ported from Pyro's test_strategies.py::test_normal_auto pattern.
    """

    def test_comprehensive_minimal(self):
        """MinimalReparam should not touch normal/loc-scale sites."""
        model = MinimalReparam()(comprehensive_model)
        actual = trace_name_type(model)
        expected = [
            ("a", "sample"),
            ("b", "sample"),
            ("c", "sample"),
            ("d", "sample"),
            ("e", "sample"),
            ("f", "sample"),
            ("g", "sample"),
            ("h", "sample"),
        ]
        assert actual == expected

    def test_comprehensive_auto_decentered(self):
        """AutoReparam(centered=0.0): Normal/StudentT → decentered, LogNormal → TransformReparam."""
        strategy = AutoReparam(centered=0.0)
        model = strategy(comprehensive_model)
        actual = trace_name_type(model)
        expected = [
            # a: Normal → LocScaleReparam → decentered aux + deterministic original
            ("a_decentered", "sample"),
            ("a", "deterministic"),
            # b: LogNormal = TransformedDistribution → TransformReparam → base (Normal)
            #    Then b_base (Normal) is also reparameterized by LocScaleReparam
            ("b_base_decentered", "sample"),
            ("b_base", "deterministic"),
            ("b", "deterministic"),
            # c: Normal(a, b) → LocScaleReparam
            ("c_decentered", "sample"),
            ("c", "deterministic"),
            # d: HalfNormal → not reparameterized (positive support)
            ("d", "sample"),
            # e: Gamma → not reparameterized (positive support)
            ("e", "sample"),
            # f: StudentT → LocScaleReparam (has loc, scale, real support)
            ("f_decentered", "sample"),
            ("f", "deterministic"),
            # g: Normal(..).to_event(1) → LocScaleReparam (unwraps Independent)
            ("g_decentered", "sample"),
            ("g", "deterministic"),
            # h: observed Normal → not reparameterized
            ("h", "sample"),
        ]
        assert actual == expected

    def test_comprehensive_auto_centered(self):
        """AutoReparam(centered=1.0): fully centered = no-op for loc-scale, but TransformReparam still fires."""
        strategy = AutoReparam(centered=1.0)
        model = strategy(comprehensive_model)
        actual = trace_name_type(model)
        expected = [
            ("a", "sample"),  # centered=1.0 is identity
            # b: LogNormal → TransformReparam still applies (not loc-scale cascade)
            ("b_base", "sample"),
            ("b", "deterministic"),
            ("c", "sample"),
            ("d", "sample"),
            ("e", "sample"),
            ("f", "sample"),
            ("g", "sample"),
            ("h", "sample"),
        ]
        assert actual == expected

    def test_projected_normal_auto(self):
        """ProjectedNormalReparam creates x_normal (Normal), which then
        gets LocScaleReparam-ed to x_normal_decentered."""
        strategy = AutoReparam(centered=0.0)
        model = strategy(projected_normal_model)
        actual = trace_name_type(model)
        expected = [
            ("x_normal_decentered", "sample"),
            ("x_normal", "deterministic"),
            ("x", "deterministic"),
            ("obs", "sample"),
        ]
        assert actual == expected

    @pytest.mark.parametrize(
        ("strategy_factory"),
        [MinimalReparam, lambda: AutoReparam(centered=0.0)],
    )
    def test_observed_projected_normal_not_reparameterized(self, strategy_factory):
        model = strategy_factory()(observed_projected_normal_model)
        actual = trace_name_type(model)
        assert actual == [("x", "sample")]

    def test_truncated_normal_not_reparameterized(self):
        """TruncatedNormal has constrained support — should be left alone."""
        strategy = AutoReparam(centered=0.0)
        model = strategy(truncated_normal_model)
        actual = trace_name_type(model)
        expected = [
            ("x", "sample"),
            ("obs", "sample"),
        ]
        assert actual == expected

    def test_plated_sites(self):
        """Sites inside numpyro.plate should be reparameterized correctly."""
        strategy = AutoReparam(centered=0.0)
        model = strategy(plated_model)
        actual = trace_name_type(model)
        expected = [
            ("mu_decentered", "sample"),
            ("mu", "deterministic"),
            ("sigma", "sample"),  # HalfNormal, not reparameterized
            ("x_decentered", "sample"),
            ("x", "deterministic"),
        ]
        assert actual == expected

    def test_config_dict_reuse(self):
        """After first run, strategy.config can be used as standalone dict config.

        Ported from Pyro's test_strategies.py::test_normal_auto.
        """
        strategy = AutoReparam(centered=0.0)
        model = strategy(comprehensive_model)
        first_result = trace_name_type(model)

        # Extract config dict and use it directly
        config_dict = strategy.config
        assert isinstance(config_dict, dict)
        model_from_dict = handlers.reparam(comprehensive_model, config=config_dict)
        second_result = trace_name_type(model_from_dict)

        assert first_result == second_result


# ---------------------------------------------------------------------------
# III. Moment + gradient preservation (NumPyro-style)
# ---------------------------------------------------------------------------


class TestMomentPreservation:
    """Verify reparameterized samples have the same distribution.

    Ported from NumPyro's test_reparam.py::test_loc_scale pattern.
    """

    @pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
    @pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, None])
    @pytest.mark.parametrize("dist_type", ["Normal", "StudentT"])
    def test_loc_scale_moments(self, dist_type, centered, shape):
        loc = np.random.uniform(-1.0, 1.0, shape)
        scale = np.random.uniform(0.5, 1.5, shape)

        def model(loc, scale):
            with numpyro.plate_stack("plates", shape), numpyro.plate("particles", 100_000):
                if dist_type == "Normal":
                    numpyro.sample("x", dist.Normal(loc, scale))
                else:
                    numpyro.sample("x", dist.StudentT(10.0, loc, scale))

        def get_expected(loc, scale):
            with handlers.trace() as tr:
                handlers.seed(model, 0)(loc, scale)
            return get_moments(tr["x"]["value"])

        shape_params = ["df"] if dist_type == "StudentT" else []
        reparam_config = {"x": LocScaleReparam(centered, shape_params=shape_params)}

        def get_actual(loc, scale):
            with handlers.trace() as tr, handlers.reparam(config=reparam_config):
                handlers.seed(model, 0)(loc, scale)
            return get_moments(tr["x"]["value"])

        expected = get_expected(loc, scale)
        actual = get_actual(loc, scale)
        # StudentT has heavier tails → higher-variance moment estimates.
        tol = 0.3 if dist_type == "StudentT" else 0.1
        assert_allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("shape", [(), (4,)], ids=str)
    @pytest.mark.parametrize("centered", [0.0, 1.0])
    def test_loc_scale_gradients(self, centered, shape):
        """Gradients through reparameterized model should match original."""
        loc = np.random.uniform(-1.0, 1.0, shape)
        scale = np.random.uniform(0.5, 1.5, shape)

        def model(loc, scale):
            with numpyro.plate_stack("plates", shape), numpyro.plate("particles", 100_000):
                numpyro.sample("x", dist.Normal(loc, scale))

        def get_expected(loc, scale):
            with handlers.trace() as tr:
                handlers.seed(model, 0)(loc, scale)
            return get_moments(tr["x"]["value"])

        def get_actual(loc, scale):
            with handlers.trace() as tr, handlers.reparam(config={"x": LocScaleReparam(centered)}):
                handlers.seed(model, 0)(loc, scale)
            return get_moments(tr["x"]["value"])

        expected_grad = jax.jacobian(get_expected, argnums=(0, 1))(loc, scale)
        actual_grad = jax.jacobian(get_actual, argnums=(0, 1))(loc, scale)
        assert_allclose(actual_grad[0], expected_grad[0], atol=0.05)
        assert_allclose(actual_grad[1], expected_grad[1], atol=0.05)


# ---------------------------------------------------------------------------
# IV. Syntax patterns (NumPyro-style)
# ---------------------------------------------------------------------------


class TestSyntax:
    """Verify handler composition patterns all produce the same trace.

    Ported from NumPyro's test_reparam.py::test_syntax.
    """

    def test_three_syntax_patterns(self):
        # Use a plain dict config so all patterns share the exact same config.
        config = {"x": LocScaleReparam(0.0), "y": LocScaleReparam(0.0)}

        # 1. Eager function syntax
        with handlers.seed(rng_seed=0):
            tr1 = handlers.trace(handlers.reparam(simple_normal_model, config=config)).get_trace()

        # 2. Context manager syntax
        with handlers.reparam(config=config), handlers.trace() as tr2, handlers.seed(rng_seed=0):
            simple_normal_model()

        # 3. Decorator syntax (Strategy.__call__)
        strategy = AutoReparam(centered=0.0)
        decorated = strategy(simple_normal_model)
        with handlers.seed(rng_seed=0):
            tr3 = handlers.trace(decorated).get_trace()

        assert tr1.keys() == tr2.keys() == tr3.keys()


# ---------------------------------------------------------------------------
# V. End-to-end inference
# ---------------------------------------------------------------------------


class TestEndToEndSVI:
    """End-to-end SVI: train, then Predictive.

    Ported from Pyro's test_strategies.py::test_end_to_end.
    """

    def test_svi_then_predictive(self):
        strategy = AutoReparam(centered=0.0)
        model = strategy(simple_normal_model)
        guide = AutoNormal(model)
        svi = SVI(model, guide, Adam(1e-3), Trace_ELBO())

        svi_state = svi.init(jax.random.PRNGKey(0))
        for _ in range(3):
            svi_state, _loss = svi.update(svi_state)

        params = svi.get_params(svi_state)
        predictive = Predictive(model, guide=guide, params=params, num_samples=5)
        samples = predictive(jax.random.PRNGKey(1))
        assert "x" in samples
        assert "y" in samples

    def test_learnable_centering(self):
        """centered=None creates numpyro.param sites that SVI can optimize."""
        strategy = AutoReparam(centered=None)
        model = strategy(simple_normal_model)
        guide = AutoNormal(model)
        svi = SVI(model, guide, Adam(1e-2), Trace_ELBO())

        svi_state = svi.init(jax.random.PRNGKey(0))
        for _ in range(20):
            svi_state, _loss = svi.update(svi_state)

        params = svi.get_params(svi_state)
        # Should have learned centering params
        centering_params = [k for k in params if "_centered" in k]
        assert len(centering_params) > 0


class TestEndToEndNUTS:
    """End-to-end NUTS with AutoReparam(centered=0.0)."""

    def test_neals_funnel(self):
        """Neal's funnel: NCP is essential for NUTS to sample correctly.

        Without NCP, NUTS produces many divergences due to the extreme
        correlation between y and x scale. With AutoReparam(centered=0.0),
        the decentered parameterization breaks this correlation.
        """
        from numpyro.infer import MCMC, NUTS

        strategy = AutoReparam(centered=0.0)
        model = strategy(neals_funnel)

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=200, num_samples=200, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), dim=5)
        samples = mcmc.get_samples()

        assert "y" in samples
        assert "x" in samples
        # y ~ N(0, 9): posterior mean should be near 0
        assert abs(float(jnp.mean(samples["y"]))) < 2.0
        # Check no extreme values (would indicate bad mixing)
        assert jnp.all(jnp.isfinite(samples["y"]))
        assert jnp.all(jnp.isfinite(samples["x"]))


# ---------------------------------------------------------------------------
# VI. SSM-specific integration
# ---------------------------------------------------------------------------


class TestAutoReparamSSM:
    """Test AutoReparam with the actual SSM model."""

    def _make_simple_ssm(self):
        from causal_ssm_agent.models.ssm.model import SSMModel, SSMSpec

        spec = SSMSpec(n_latent=2, n_manifest=2)
        return SSMModel(spec=spec, likelihood="kalman")

    def test_ssm_site_classification(self):
        """Verify which SSM sites get reparameterized and which don't."""
        model = self._make_simple_ssm()
        strategy = AutoReparam(centered=0.0)

        model_fn = functools.partial(
            model.model, likelihood_backend=model.make_likelihood_backend()
        )
        reparam_model = handlers.reparam(model_fn, config=strategy)

        T = 10
        observations = jnp.zeros((T, 2))
        times = jnp.linspace(0, 1, T)

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(reparam_model).get_trace(observations, times)

        # Normal sites (loc-scale, real support) → LocScaleReparam
        for site in ["drift_diag_pop", "drift_offdiag_pop", "t0_means_pop"]:
            if site in strategy.config:
                assert isinstance(strategy.config[site], LocScaleReparam), (
                    f"{site} should be LocScaleReparam"
                )

        # HalfNormal sites (positive support) → None
        for site in ["diffusion_diag_pop", "manifest_var_diag", "t0_var_diag"]:
            assert strategy.config.get(site) is None, f"{site} should NOT be reparameterized"

        # All values finite
        for name, site in trace.items():
            if site["type"] in ("sample", "deterministic"):
                assert jnp.all(jnp.isfinite(site["value"])), f"Non-finite at {name}"

    def test_fit_svi_with_reparam(self):
        """fit() + SVI + AutoReparam produces valid posterior samples."""
        from causal_ssm_agent.models.ssm.inference import fit

        model = self._make_simple_ssm()
        T = 10
        observations = jnp.zeros((T, 2))
        times = jnp.linspace(0, 1, T)

        result = fit(
            model,
            observations,
            times,
            method="svi",
            reparam=AutoReparam(centered=0.0),
            num_steps=50,
            num_samples=10,
            seed=42,
        )

        assert result.method == "svi"
        samples = result.get_samples()
        assert len(samples) > 0
        for v in samples.values():
            assert jnp.all(jnp.isfinite(v))

    def test_fit_nuts_filters_auxiliary_sites(self):
        """NUTS results and diagnostics should expose original sites only."""
        from causal_ssm_agent.models.ssm.inference import fit

        model = self._make_simple_ssm()
        observations = jnp.zeros((8, 2))
        times = jnp.linspace(0, 1, 8)

        result = fit(
            model,
            observations,
            times,
            method="nuts",
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            seed=0,
        )

        sample_names = set(result.get_samples())
        assert "drift_diag_pop" in sample_names
        assert "diffusion_diag_pop" in sample_names
        assert all("_decentered" not in name for name in sample_names)

        diag = result.get_mcmc_diagnostics()
        assert diag is not None
        diag_names = {entry["parameter"] for entry in diag["per_parameter"]}
        assert all("_decentered" not in name for name in diag_names)

    def test_extract_constrained_samples_filters_auxiliary_sites(self):
        """Replay-based extraction should drop internal reparam auxiliaries."""
        from jax.flatten_util import ravel_pytree

        from causal_ssm_agent.models.ssm.utils import _discover_sites, extract_constrained_samples

        model = self._make_simple_ssm()
        strategy = AutoReparam(centered=0.0)
        observations = jnp.zeros((5, 2))
        times = jnp.linspace(0, 1, 5)
        backend = model.make_likelihood_backend()
        site_info = _discover_sites(
            model,
            observations,
            times,
            jax.random.PRNGKey(0),
            backend,
            reparam=strategy,
        )
        example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
        flat, unravel_fn = ravel_pytree(example_unc)

        samples = extract_constrained_samples(
            flat[None, :],
            site_info,
            unravel_fn,
            model.spec,
            reparam=strategy,
            model=model,
            observations=observations,
            times=times,
        )

        assert "drift_diag_pop" in samples
        assert "diffusion_diag_pop" in samples
        assert all("_decentered" not in name for name in samples)

    def test_fit_nuts_da_noncentered_with_reparam(self):
        """Default reparam should not break the non-centered DA state path."""
        from causal_ssm_agent.models.ssm.inference import fit

        model = self._make_simple_ssm()
        observations = jnp.zeros((4, 2))
        times = jnp.linspace(0, 1, 4)

        result = fit(
            model,
            observations,
            times,
            method="nuts_da",
            centered=False,
            num_warmup=1,
            num_samples=1,
            num_chains=1,
            svi_warmstart=False,
            seed=0,
        )

        sample_names = set(result.get_samples())
        assert "drift_diag_pop" in sample_names
        assert "eps" not in sample_names
        assert "eps_0" not in sample_names
        assert all("_decentered" not in name for name in sample_names)

    def test_fit_pgas_rejects_reparam(self):
        """PGAS should fail explicitly rather than silently ignore reparam."""
        from causal_ssm_agent.models.ssm.inference import fit

        model = self._make_simple_ssm()
        observations = jnp.zeros((4, 2))
        times = jnp.linspace(0, 1, 4)

        with pytest.raises(ValueError, match="PGAS does not support reparameterization"):
            fit(
                model,
                observations,
                times,
                method="pgas",
                reparam=AutoReparam(centered=0.0),
                n_outer=1,
                n_csmc_particles=4,
                n_mh_steps=1,
                svi_warmstart=False,
                seed=0,
            )
