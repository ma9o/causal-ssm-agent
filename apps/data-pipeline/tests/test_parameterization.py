"""Tests for the canonical site registry and compile-stable prior evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
import pytest
from jax.flatten_util import ravel_pytree
from numpyro import handlers

from causal_ssm_agent.models.ssm.model import SSMModel, SSMPriors, SSMSpec
from causal_ssm_agent.models.ssm.parameterization import (
    SupportClass,
    assemble_deterministics_from_registry,
    build_prior_runtime_state,
    build_site_registry,
    build_transforms,
    build_unravel_fn,
    compile_prior_semantics,
    deserialize_prior_runtime_state,
    deserialize_site_registry,
    group_sites_by_assembly_role,
    log_prior_unconstrained,
    reconstruct_ssm_priors,
    sample_prior_unconstrained,
    serialize_prior_runtime_state,
    serialize_site_registry,
    verify_registry_matches_trace,
)
from causal_ssm_agent.models.ssm.utils import _discover_sites
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_spec():
    """Minimal 2-latent, 2-manifest Gaussian SSM."""
    return SSMSpec(n_latent=2, n_manifest=2)


@pytest.fixture
def simple_model(simple_spec):
    return SSMModel(simple_spec, likelihood="kalman")


@pytest.fixture
def dag_spec():
    """DAG-constrained spec with drift mask and lambda mask."""
    import numpy as np

    drift_mask = np.array([[True, False], [True, True]])  # (0,1) edge blocked
    lambda_mask = np.array([[True, False], [False, True]])
    lambda_template = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    return SSMSpec(
        n_latent=2,
        n_manifest=2,
        drift_mask=drift_mask,
        lambda_mat=lambda_template,
        lambda_mask=lambda_mask,
        cint="free",
    )


@pytest.fixture
def dag_model(dag_spec):
    return SSMModel(dag_spec, likelihood="kalman")


# ---------------------------------------------------------------------------
# Site registry tests
# ---------------------------------------------------------------------------


class TestSiteRegistry:
    def test_registry_names_match_trace(self, simple_model):
        """Registry produces the same site names as model tracing."""
        spec = simple_model.spec
        registry = build_site_registry(spec)
        backend = simple_model.make_likelihood_backend()
        T = 10
        obs = jnp.zeros((T, spec.n_manifest))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(simple_model, obs, times, random.PRNGKey(0), backend)
        verify_registry_matches_trace(registry, site_info)

    def test_registry_names_match_trace_dag(self, dag_model):
        """Registry matches trace for DAG-constrained model with cint."""
        spec = dag_model.spec
        registry = build_site_registry(spec)
        backend = dag_model.make_likelihood_backend()
        T = 10
        obs = jnp.zeros((T, spec.n_manifest))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(dag_model, obs, times, random.PRNGKey(0), backend)
        verify_registry_matches_trace(registry, site_info)

    def test_registry_shapes_match_trace(self, simple_model):
        """Registry shapes match traced shapes."""
        spec = simple_model.spec
        registry = build_site_registry(spec)
        backend = simple_model.make_likelihood_backend()
        T = 10
        obs = jnp.zeros((T, spec.n_manifest))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(simple_model, obs, times, random.PRNGKey(0), backend)
        for site in registry:
            assert site.shape == site_info[site.name]["shape"], (
                f"Shape mismatch for {site.name}: "
                f"registry={site.shape}, trace={site_info[site.name]['shape']}"
            )

    def test_registry_sorted_by_name(self, simple_spec):
        """Registry is sorted by site name (JAX pytree convention)."""
        registry = build_site_registry(simple_spec)
        names = [s.name for s in registry]
        assert names == sorted(names)

    def test_fixed_drift_excludes_drift_sites(self):
        """When drift is a fixed array, no drift sites appear."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.array([[-0.5, 0.0], [0.0, -0.5]]),
        )
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert "drift_diag_pop" not in names
        assert "drift_offdiag_pop" not in names

    def test_diag_diffusion_excludes_lower(self):
        """Diagonal diffusion has no lower-triangle sites."""
        spec = SSMSpec(n_latent=2, n_manifest=2, diffusion="diag")
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert "diffusion_diag_pop" in names
        assert "diffusion_lower" not in names

    def test_free_diffusion_includes_lower(self):
        """Free diffusion includes lower-triangle sites."""
        spec = SSMSpec(n_latent=2, n_manifest=2, diffusion="free")
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert "diffusion_diag_pop" in names
        assert "diffusion_lower" in names

    def test_support_classes(self, simple_spec):
        """Check that support classes are correctly assigned."""
        registry = build_site_registry(simple_spec)
        support_map = {s.name: s.support for s in registry}
        # REAL support sites (Normal priors)
        assert support_map["drift_diag_pop"] == SupportClass.REAL
        # POSITIVE support sites (HalfNormal priors)
        assert support_map["diffusion_diag_pop"] == SupportClass.POSITIVE
        assert support_map["manifest_var_diag"] == SupportClass.POSITIVE
        assert support_map["t0_var_diag"] == SupportClass.POSITIVE

    def test_mixed_diffusion_includes_proc_df_site(self):
        """Any student-t latent in diffusion_dists should expose proc_df."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        registry = build_site_registry(spec)
        assert "proc_df" in {site.name for site in registry}

    def test_mixed_diffusion_sampling_emits_proc_df(self):
        """The traced model should sample proc_df when diffusion_dists include student_t."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        model = SSMModel(spec, likelihood="particle")

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(lambda: model._sample_likelihood_extra_params(spec)).get_trace()

        assert "proc_df" in trace


# ---------------------------------------------------------------------------
# Transforms and unravel
# ---------------------------------------------------------------------------


class TestTransformsAndUnravel:
    def test_unravel_dimension(self, simple_spec):
        """Unravel function produces correct flat dimension."""
        registry = build_site_registry(simple_spec)
        D, unravel_fn = build_unravel_fn(registry)
        z = jnp.zeros(D)
        unc = unravel_fn(z)
        total = sum(jnp.prod(jnp.array(v.shape)) for v in unc.values())
        assert int(total) == D

    def test_unravel_matches_trace(self, simple_model):
        """Registry-based unravel gives same structure as trace-based."""
        spec = simple_model.spec
        registry = build_site_registry(spec)
        D, _unravel_fn = build_unravel_fn(registry)

        backend = simple_model.make_likelihood_backend()
        obs = jnp.zeros((10, spec.n_manifest))
        times = jnp.linspace(0, 1, 10)
        site_info = _discover_sites(simple_model, obs, times, random.PRNGKey(0), backend)
        example_unc = {
            name: info["transform"].inv(info["value"]) for name, info in site_info.items()
        }
        flat_trace, _unravel_trace = ravel_pytree(example_unc)

        assert flat_trace.shape[0] == D

    def test_transforms_exp_for_positive(self, simple_spec):
        """POSITIVE sites get ExpTransform."""
        registry = build_site_registry(simple_spec)
        transforms = build_transforms(registry)
        # diffusion_diag_pop is POSITIVE
        z = jnp.array([0.5, -0.3])
        x = transforms["diffusion_diag_pop"](z)
        assert jnp.allclose(x, jnp.exp(z))

    def test_transforms_identity_for_real(self, simple_spec):
        """REAL sites get IdentityTransform."""
        registry = build_site_registry(simple_spec)
        transforms = build_transforms(registry)
        z = jnp.array([0.5, -0.3])
        x = transforms["drift_diag_pop"](z)
        assert jnp.allclose(x, z)


# ---------------------------------------------------------------------------
# Registry-driven deterministic assembly
# ---------------------------------------------------------------------------


class TestDeterministicAssembly:
    def test_group_sites_by_assembly_role(self, simple_spec):
        """Assembly grouping is driven by registry metadata."""
        registry = build_site_registry(simple_spec)
        grouped = group_sites_by_assembly_role(registry)
        assert "drift" in grouped
        assert "diffusion" in grouped
        assert {site.name for site in grouped["drift"]} == {
            "drift_diag_pop",
            "drift_offdiag_pop",
        }

    def test_assemble_deterministics_from_registry_free_spec(self, simple_spec):
        """Registry-driven assembly builds the expected matrices."""
        registry = build_site_registry(simple_spec)
        samples = {
            "drift_diag_pop": jnp.array([[0.5, 0.3]], dtype=jnp.float32),
            "drift_offdiag_pop": jnp.array([[0.1, -0.2]], dtype=jnp.float32),
            "diffusion_diag_pop": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_pop": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, simple_spec, registry)
        assert det["drift"].shape == (1, 2, 2)
        assert jnp.allclose(jnp.diag(det["drift"][0]), jnp.array([-0.5, -0.3]))
        assert jnp.allclose(det["diffusion"][0], jnp.array([[0.4, 0.0], [0.25, 0.6]]))
        assert det["lambda"].shape == (1, 2, 2)
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.49, 0.64])))
        assert jnp.allclose(det["t0_means"][0], jnp.array([1.0, -1.0]))
        assert jnp.allclose(det["t0_cov"][0], jnp.diag(jnp.array([0.81, 1.21])))

    def test_assemble_deterministics_from_registry_fixed_fallbacks(self):
        """Fixed spec matrices are broadcast without any sampled sites."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.array([[-0.4, 0.1], [0.0, -0.2]], dtype=jnp.float32),
            diffusion=jnp.array([[0.3, 0.0], [0.1, 0.5]], dtype=jnp.float32),
            lambda_mat=jnp.array([[1.0, 0.0], [0.2, 1.0]], dtype=jnp.float32),
            manifest_var=jnp.array([[0.4, 0.0], [0.0, 0.6]], dtype=jnp.float32),
            t0_means=jnp.array([0.5, -0.5], dtype=jnp.float32),
            t0_var=jnp.array([[0.7, 0.0], [0.0, 0.8]], dtype=jnp.float32),
        )
        registry = build_site_registry(spec)

        det = assemble_deterministics_from_registry({}, spec, registry, n_draws=3)
        assert jnp.allclose(det["drift"], jnp.broadcast_to(spec.drift, (3, 2, 2)))
        assert jnp.allclose(det["diffusion"], jnp.broadcast_to(spec.diffusion, (3, 2, 2)))
        assert jnp.allclose(det["lambda"], jnp.broadcast_to(spec.lambda_mat, (3, 2, 2)))
        expected_manifest_cov = spec.manifest_var @ spec.manifest_var.T
        assert jnp.allclose(det["manifest_cov"], jnp.broadcast_to(expected_manifest_cov, (3, 2, 2)))
        assert jnp.allclose(det["t0_means"], jnp.broadcast_to(spec.t0_means, (3, 2)))
        expected_t0_cov = spec.t0_var @ spec.t0_var.T
        assert jnp.allclose(det["t0_cov"], jnp.broadcast_to(expected_t0_cov, (3, 2, 2)))


# ---------------------------------------------------------------------------
# Prior runtime state
# ---------------------------------------------------------------------------


class TestPriorRuntimeState:
    def test_default_state_has_all_sites(self, simple_spec):
        """Default prior state covers all registry sites."""
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry)
        for site in registry:
            assert site.name in state, f"Missing site {site.name}"

    def test_state_has_correct_keys(self, simple_spec):
        """Each site's params have correct keys for its support class."""
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry)
        for site in registry:
            params = state[site.name]
            assert "family" in params
            if site.support == SupportClass.REAL:
                assert "loc" in params
                assert "scale" in params
            elif site.support == SupportClass.POSITIVE:
                assert "scale" in params
                assert "concentration" in params
                assert "rate" in params

    def test_custom_priors_reflected(self, simple_spec):
        """Custom SSMPriors values appear in the state."""
        priors = SSMPriors(drift_diag={"mu": -2.0, "sigma": 0.5})
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry, priors)
        assert jnp.allclose(state["drift_diag_pop"]["loc"], jnp.full(2, -2.0))
        assert jnp.allclose(state["drift_diag_pop"]["scale"], jnp.full(2, 0.5))

    def test_state_is_valid_pytree(self, simple_spec):
        """Prior state can be flattened/unflattened as a JAX pytree."""
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        state2 = jax.tree_util.tree_unflatten(treedef, leaves)
        for site in registry:
            for key in state[site.name]:
                assert jnp.allclose(state[site.name][key], state2[site.name][key])


# ---------------------------------------------------------------------------
# Prior log-probability correctness
# ---------------------------------------------------------------------------


class TestLogPriorCorrectness:
    def test_normal_log_prob_matches_numpyro(self):
        """Pure-JAX Normal log_prob matches NumPyro."""
        from causal_ssm_agent.models.ssm.parameterization import _normal_log_prob

        x = jnp.array([0.5, -1.0, 2.0])
        loc = jnp.array([0.0, 0.0, 1.0])
        scale = jnp.array([1.0, 2.0, 0.5])
        expected = jnp.sum(dist.Normal(loc, scale).log_prob(x))
        actual = _normal_log_prob(x, loc, scale)
        assert jnp.allclose(actual, expected, atol=1e-5)

    def test_half_normal_log_prob_matches_numpyro(self):
        """Pure-JAX HalfNormal log_prob matches NumPyro."""
        from causal_ssm_agent.models.ssm.parameterization import _half_normal_log_prob

        x = jnp.array([0.5, 1.0, 2.0])
        scale = jnp.array([1.0, 2.0, 0.5])
        expected = jnp.sum(dist.HalfNormal(scale).log_prob(x))
        actual = _half_normal_log_prob(x, scale)
        assert jnp.allclose(actual, expected, atol=1e-5)

    def test_gamma_log_prob_matches_numpyro(self):
        """Pure-JAX Gamma log_prob matches NumPyro."""
        from causal_ssm_agent.models.ssm.parameterization import _gamma_log_prob

        x = jnp.array([0.5, 1.0, 2.0])
        concentration = jnp.array([2.0, 5.0, 1.0])
        rate = jnp.array([1.0, 0.5, 2.0])
        expected = jnp.sum(dist.Gamma(concentration, rate).log_prob(x))
        actual = _gamma_log_prob(x, concentration, rate)
        assert jnp.allclose(actual, expected, atol=1e-5)

    def test_log_prior_unc_matches_trace_based(self, simple_model):
        """Registry-based log prior matches trace-based log prior."""
        spec = simple_model.spec
        registry = build_site_registry(spec)
        D, unravel_fn = build_unravel_fn(registry)
        prior_state = build_prior_runtime_state(registry, simple_model.priors)

        # Trace-based reference
        backend = simple_model.make_likelihood_backend()
        obs = jnp.zeros((10, spec.n_manifest))
        times = jnp.linspace(0, 1, 10)
        site_info = _discover_sites(simple_model, obs, times, random.PRNGKey(0), backend)
        trace_transforms = {name: info["transform"] for name, info in site_info.items()}
        trace_distributions = {name: info["distribution"] for name, info in site_info.items()}
        example_unc = {
            name: info["transform"].inv(info["value"]) for name, info in site_info.items()
        }
        _, trace_unravel = ravel_pytree(example_unc)

        # Evaluate at a random point
        rng_key = random.PRNGKey(42)
        z = random.normal(rng_key, (D,)) * 0.5

        # Registry-based
        lp_registry = log_prior_unconstrained(z, unravel_fn, registry, prior_state)

        # Trace-based
        unc = trace_unravel(z)
        con = {name: trace_transforms[name](unc[name]) for name in unc}
        lp_trace = sum(jnp.sum(trace_distributions[name].log_prob(con[name])) for name in unc)
        lj_trace = sum(
            jnp.sum(trace_transforms[name].log_abs_det_jacobian(unc[name], con[name]))
            for name in unc
        )
        lp_trace_total = lp_trace + lj_trace

        assert jnp.allclose(lp_registry, lp_trace_total, atol=1e-4), (
            f"Registry: {lp_registry}, Trace: {lp_trace_total}"
        )

    def test_log_prior_unc_gradients_flow(self, simple_spec):
        """log_prior_unconstrained is differentiable."""
        registry = build_site_registry(simple_spec)
        D, unravel_fn = build_unravel_fn(registry)
        prior_state = build_prior_runtime_state(registry)

        z = jnp.ones(D) * 0.1
        grad_fn = jax.grad(lambda z: log_prior_unconstrained(z, unravel_fn, registry, prior_state))
        g = grad_fn(z)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == (D,)


# ---------------------------------------------------------------------------
# Compile stability (no recompilation on prior changes)
# ---------------------------------------------------------------------------


class TestCompileStability:
    def test_no_retrace_on_prior_value_change(self, simple_spec):
        """Changing prior values does not trigger JAX retracing."""
        registry = build_site_registry(simple_spec)
        D, unravel_fn = build_unravel_fn(registry)

        state1 = build_prior_runtime_state(
            registry, SSMPriors(drift_diag={"mu": -0.5, "sigma": 1.0})
        )
        state2 = build_prior_runtime_state(
            registry, SSMPriors(drift_diag={"mu": -2.0, "sigma": 0.3})
        )

        trace_count = 0

        @jax.jit
        def _eval(z, ps):
            nonlocal trace_count
            trace_count += 1
            return log_prior_unconstrained(z, unravel_fn, registry, ps)

        z = jnp.zeros(D)

        # First call: traces and compiles
        _ = _eval(z, state1)
        traces_after_first = trace_count

        # Second call with different prior values: should NOT retrace
        _ = _eval(z, state2)
        assert trace_count == traces_after_first, (
            f"Retraced on prior value change: {trace_count} > {traces_after_first}"
        )

    def test_no_retrace_on_family_switch(self, simple_spec):
        """Changing family index does not trigger JAX retracing."""
        from causal_ssm_agent.models.ssm.parameterization import _make_positive_params

        registry = build_site_registry(simple_spec)
        D, unravel_fn = build_unravel_fn(registry)

        state1 = build_prior_runtime_state(registry)
        # Switch diffusion_diag_pop from HalfNormal (0) to Gamma (1)
        # Use _make_positive_params to ensure consistent weak_type/dtype.
        state2 = build_prior_runtime_state(registry)
        state2["diffusion_diag_pop"] = _make_positive_params(
            (simple_spec.n_latent,),
            family=1,
            scale=1.0,
            concentration=2.0,
            rate=1.0,
        )

        trace_count = 0

        @jax.jit
        def _eval(z, ps):
            nonlocal trace_count
            trace_count += 1
            return log_prior_unconstrained(z, unravel_fn, registry, ps)

        z = jnp.zeros(D)

        _ = _eval(z, state1)
        traces_after_first = trace_count

        _ = _eval(z, state2)
        assert trace_count == traces_after_first, (
            f"Retraced on family switch: {trace_count} > {traces_after_first}"
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_shape(self, simple_spec):
        """Sampled array has correct shape."""
        registry = build_site_registry(simple_spec)
        D, _ = build_unravel_fn(registry)
        state = build_prior_runtime_state(registry)
        samples, _ = sample_prior_unconstrained(random.PRNGKey(0), registry, state, n_samples=50)
        assert samples.shape == (50, D)

    def test_samples_finite(self, simple_spec):
        """All samples are finite."""
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry)
        samples, _ = sample_prior_unconstrained(random.PRNGKey(0), registry, state, n_samples=100)
        assert jnp.all(jnp.isfinite(samples))

    def test_positive_sites_log_space(self, simple_spec):
        """Samples for POSITIVE sites are in log space (unconstrained)."""
        registry = build_site_registry(simple_spec)
        _D, unravel_fn = build_unravel_fn(registry)
        state = build_prior_runtime_state(registry)
        samples, _ = sample_prior_unconstrained(random.PRNGKey(0), registry, state, n_samples=100)
        # Check one sample: exp(unconstrained) should be positive
        unc = unravel_fn(samples[0])
        for site in registry:
            if site.support == SupportClass.POSITIVE:
                assert jnp.all(jnp.exp(unc[site.name]) > 0)


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_registry_roundtrip(self, simple_spec):
        """Site registry survives serialize → deserialize."""
        registry = build_site_registry(simple_spec)
        payload = serialize_site_registry(registry)
        restored = deserialize_site_registry(payload)
        assert len(restored) == len(registry)
        for orig, rest in zip(registry, restored):
            assert orig.name == rest.name
            assert orig.shape == rest.shape
            assert orig.support == rest.support
            assert orig.assembly_group == rest.assembly_group
            assert orig.site_kind == rest.site_kind
            assert orig.transform_kind == rest.transform_kind
            assert orig.deterministic_name == rest.deterministic_name
            assert orig.fixed_spec_field == rest.fixed_spec_field
            assert orig.priors_field == rest.priors_field
            assert orig.runtime_prior_key == rest.runtime_prior_key
            assert orig.is_runtime_prior_controlled == rest.is_runtime_prior_controlled

    def test_prior_state_roundtrip(self, simple_spec):
        """Prior runtime state survives serialize → deserialize."""
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry)
        payload = serialize_prior_runtime_state(state)
        restored = deserialize_prior_runtime_state(payload, registry)
        for site in registry:
            for key in state[site.name]:
                assert jnp.allclose(
                    state[site.name][key],
                    restored[site.name][key],
                    atol=1e-6,
                ), f"Mismatch for {site.name}.{key}"

    def test_prior_state_roundtrip_custom_priors(self, simple_spec):
        """Custom priors survive the roundtrip."""
        priors = SSMPriors(
            drift_diag={"mu": -2.0, "sigma": 0.3},
            diffusion_diag={"sigma": 0.5},
        )
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry, priors)
        payload = serialize_prior_runtime_state(state)
        restored = deserialize_prior_runtime_state(payload, registry)
        assert jnp.allclose(
            restored["drift_diag_pop"]["loc"],
            jnp.full(2, -2.0),
            atol=1e-6,
        )
        assert jnp.allclose(
            restored["diffusion_diag_pop"]["scale"],
            jnp.full(2, 0.5),
            atol=1e-6,
        )

    def test_compile_prior_semantics_roundtrip(self, simple_spec):
        """compile_prior_semantics → deserialize produces valid state."""
        priors = SSMPriors(drift_diag={"mu": -1.0, "sigma": 0.8})
        semantics = compile_prior_semantics(simple_spec, priors)
        assert semantics["schema_version"] == 3
        registry = deserialize_site_registry(semantics["site_registry"])
        state = deserialize_prior_runtime_state(semantics["prior_state"], registry)
        assert "drift_diag_pop" in state
        assert jnp.allclose(state["drift_diag_pop"]["loc"], jnp.full(2, -1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# SSMPriors reconstruction
# ---------------------------------------------------------------------------


class TestSSMPriorsReconstruction:
    def test_reconstruct_matches_original_scalar(self, simple_spec):
        """Reconstructed SSMPriors matches original for scalar priors."""
        original = SSMPriors()
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry, original)
        reconstructed = reconstruct_ssm_priors(registry, state)
        # Scalar priors should roundtrip as scalars
        assert isinstance(reconstructed.drift_diag["mu"], float)
        assert isinstance(reconstructed.diffusion_diag["sigma"], float)
        assert abs(reconstructed.drift_diag["mu"] - original.drift_diag["mu"]) < 1e-5
        assert abs(reconstructed.drift_diag["sigma"] - original.drift_diag["sigma"]) < 1e-5
        assert abs(reconstructed.diffusion_diag["sigma"] - original.diffusion_diag["sigma"]) < 1e-5

    def test_reconstruct_preserves_per_element_priors(self):
        """Per-element array priors survive reconstruction."""
        spec = SSMSpec(n_latent=3, n_manifest=3)
        priors = SSMPriors(
            drift_diag={"mu": [-0.5, -0.3, -0.7], "sigma": [1.0, 0.5, 0.8]},
        )
        registry = build_site_registry(spec)
        state = build_prior_runtime_state(registry, priors)
        reconstructed = reconstruct_ssm_priors(registry, state)
        # Per-element priors should roundtrip as lists
        assert isinstance(reconstructed.drift_diag["mu"], list)
        assert len(reconstructed.drift_diag["mu"]) == 3
        for i in range(3):
            assert abs(reconstructed.drift_diag["mu"][i] - priors.drift_diag["mu"][i]) < 1e-5

    def test_reconstruct_skips_likelihood_extras(self, simple_spec):
        """Likelihood extra sites don't appear in SSMPriors."""
        from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            manifest_dist=DistributionFamily.STUDENT_T,
        )
        registry = build_site_registry(spec)
        state = build_prior_runtime_state(registry)
        reconstructed = reconstruct_ssm_priors(registry, state)
        # obs_df is a likelihood extra — should not appear in SSMPriors
        assert not hasattr(reconstructed, "obs_df")

    def test_reconstruct_preserves_positive_family_metadata(self, simple_spec):
        """Positive runtime families should roundtrip through compiled semantics."""
        priors = SSMPriors(diffusion_diag={"family": 2, "loc": 0.2, "sigma": 0.7})
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry, priors)
        reconstructed = reconstruct_ssm_priors(registry, state)
        assert reconstructed.diffusion_diag["family"] == 2
        assert reconstructed.diffusion_diag["loc"] == pytest.approx(0.2)
        assert reconstructed.diffusion_diag["sigma"] == pytest.approx(0.7)

    def test_reconstruct_preserves_bounded_real_metadata(self, simple_spec):
        """Bounded executable priors should retain their bounds after reconstruction."""
        priors = SSMPriors(drift_offdiag={"mu": 0.0, "sigma": 0.3, "lower": -1.0, "upper": 1.0})
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry, priors)
        reconstructed = reconstruct_ssm_priors(registry, state)
        assert reconstructed.drift_offdiag["mu"] == pytest.approx(0.0)
        assert reconstructed.drift_offdiag["sigma"] == pytest.approx(0.3)
        assert reconstructed.drift_offdiag["lower"] == pytest.approx(-1.0)
        assert reconstructed.drift_offdiag["upper"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Compiled artifact integration (hard cutover)
# ---------------------------------------------------------------------------


class TestCompiledArtifactIntegration:
    """Test that compiled_prior_semantics is emitted and correctly consumed."""

    @pytest.fixture
    def model_spec_and_priors(self):
        return (
            {
                "likelihoods": [
                    {
                        "variable": "mood_score",
                        "distribution": "gaussian",
                        "link": "identity",
                        "reasoning": "test",
                    }
                ],
                "parameters": [
                    {
                        "name": "rho_mood",
                        "role": "ar_coefficient",
                        "constraint": "unit_interval",
                        "description": "AR mood",
                    },
                    {
                        "name": "sigma_mood",
                        "role": "residual_sd",
                        "constraint": "positive",
                        "description": "SD mood",
                    },
                ],
            },
            {
                "rho_mood": {
                    "parameter": "rho_mood",
                    "distribution": "Normal",
                    "params": {"mu": 0.5, "sigma": 0.2},
                    "sources": [],
                    "reasoning": "r",
                },
                "sigma_mood": {
                    "parameter": "sigma_mood",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 1.0},
                    "sources": [],
                    "reasoning": "r",
                },
            },
        )

    def test_artifact_contains_compiled_prior_semantics(self, model_spec_and_priors):
        """compile_ssm_artifact emits semantics and omits legacy priors."""
        from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)
        assert "priors" not in artifact
        assert "compiled_prior_semantics" in artifact
        sem = artifact["compiled_prior_semantics"]
        assert sem["schema_version"] == 3
        assert "site_registry" in sem
        assert "prior_state" in sem

    def test_builder_from_artifact_uses_semantics(self, model_spec_and_priors):
        """make_builder_from_compiled_artifact reads compiled_prior_semantics."""
        from causal_ssm_agent.models.ssm_compiler import (
            compile_ssm_artifact,
            make_builder_from_compiled_artifact,
        )

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)
        builder = make_builder_from_compiled_artifact(artifact)
        assert builder._ssm_priors is not None

    def test_builder_requires_compiled_prior_semantics(self, model_spec_and_priors):
        """Builder fails clearly when compiled semantics are missing."""
        from causal_ssm_agent.models.ssm_compiler import (
            compile_ssm_artifact,
            make_builder_from_compiled_artifact,
        )

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)
        del artifact["compiled_prior_semantics"]

        with pytest.raises(ValueError, match="compiled_prior_semantics"):
            make_builder_from_compiled_artifact(artifact)

    def test_end_to_end_compile_rebuild_sample(self, model_spec_and_priors):
        """Full roundtrip: compile → rebuild → sample."""
        import numpy as np
        import polars as pl

        from causal_ssm_agent.models.ssm_builder import build_ssm_builder
        from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact
        from causal_ssm_agent.utils.data import pivot_to_wide

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)

        rng = np.random.default_rng(42)
        n = 30
        data_for_model = pl.DataFrame(
            {
                "indicator": ["mood_score"] * n,
                "value": (rng.standard_normal(n) * 1.5 + 5).tolist(),
                "anchor_time": list(range(n)),
            }
        )
        builder = build_ssm_builder(wide_data=pivot_to_wide(data_for_model), compiled_ssm=artifact)
        assert builder._model is not None
        samples = builder.sample_prior_predictive(samples=5)
        assert samples is not None

    def test_compiled_builder_prior_predictive_without_model(self, model_spec_and_priors):
        """Compiled builders can sample prior predictive without building a model."""
        from causal_ssm_agent.models.ssm_compiler import (
            compile_ssm_artifact,
            make_builder_from_compiled_artifact,
        )

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)
        builder = make_builder_from_compiled_artifact(artifact)
        samples = builder.sample_prior_predictive(samples=4)
        assert "drift" in samples
        assert samples["drift"].shape[0] == 4
        assert "observations" in samples
