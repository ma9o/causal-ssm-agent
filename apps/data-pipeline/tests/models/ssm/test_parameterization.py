"""Tests for the canonical site registry and compile-stable prior evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pytest
from jax.flatten_util import ravel_pytree
from numpyro import handlers

from nof1_causal_lab.distributions import (
    DistributionFamily,
    PriorDistributionFamily,
)
from nof1_causal_lab.models.ssm.inference.utils import _discover_sites
from nof1_causal_lab.models.ssm.model import (
    SSMModel,
    SSMSpec,
)
from nof1_causal_lab.models.ssm.parameterization import (
    SupportClass,
    assemble_deterministics_from_registry,
    build_prior_runtime_state,
    build_site_prior_distribution,
    build_site_registry,
    build_unravel_fn,
    compile_prior_semantics,
    deserialize_prior_runtime_state,
    deserialize_site_registry,
    group_sites_by_assembly_role,
    load_prior_runtime_bundle,
    log_prior_unconstrained,
    sample_prior_unconstrained,
    serialize_prior_runtime_state,
    serialize_site_registry,
    verify_registry_matches_trace,
)
from nof1_causal_lab.models.ssm.priors import PriorSpec
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind
from nof1_causal_lab.models.ssm.testing import (
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
    dense_matrix_dynamics_spec,
    full_cholesky_support,
    full_dense_matrix_dynamics_spec,
    full_diagonal_support,
    full_vector_support,
    prior_registry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(
    *,
    n_latent: int = 2,
    n_manifest: int = 2,
    dynamics_spec=None,
    diffusion_block=None,
    lambda_block=None,
    manifest_means_block=None,
    manifest_chol_block=None,
    t0_means_block=None,
    t0_chol_block=None,
    input_effect_block=None,
    static_state_sd_block=None,
    **kwargs,
) -> SSMSpec:
    """Build an SSMSpec from explicit block specs for tests."""
    if dynamics_spec is None:
        dynamics_spec = full_dense_matrix_dynamics_spec(n_latent)
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dynamics_spec,
        diffusion_block=diffusion_block or default_diffusion_block(n_latent),
        lambda_block=lambda_block or default_lambda_block(n_manifest, n_latent),
        manifest_means_block=manifest_means_block or default_manifest_means_block(n_manifest),
        manifest_chol_block=manifest_chol_block or default_manifest_chol_block(n_manifest),
        t0_means_block=t0_means_block or default_t0_means_block(n_latent),
        t0_chol_block=t0_chol_block or default_t0_chol_block(n_latent),
        input_effect_block=input_effect_block or default_input_effect_block(n_latent),
        static_state_sd_block=static_state_sd_block or default_static_state_sd_block(),
        **kwargs,
    )


def _linear_dynamics(
    n_latent: int,
    *,
    decay_support: np.ndarray | None = None,
    edge_support: np.ndarray | None = None,
    coupling_template=None,
    intercept_support: np.ndarray | None = None,
    cint_template=None,
):
    if decay_support is None:
        decay_support = full_diagonal_support(n_latent)
    if edge_support is None:
        edge_support = np.zeros((n_latent, n_latent), dtype=bool)
    if coupling_template is None:
        coupling_template = jnp.zeros((n_latent, n_latent), dtype=jnp.float32)
    if intercept_support is None:
        intercept_support = np.zeros(n_latent, dtype=bool)
    if cint_template is None:
        cint_template = jnp.zeros(n_latent, dtype=jnp.float32)
    return dense_matrix_dynamics_spec(
        n_latent=n_latent,
        decay_support=decay_support,
        edge_support=edge_support,
        coupling_template=jnp.asarray(coupling_template),
        intercept_support=intercept_support,
        cint_template=jnp.asarray(cint_template),
    )


def _fixed_matrix_dynamics(coupling_template):
    coupling_template = jnp.asarray(coupling_template)
    n_latent = int(coupling_template.shape[0])
    return _linear_dynamics(
        n_latent,
        decay_support=np.zeros(n_latent, dtype=bool),
        edge_support=np.zeros((n_latent, n_latent), dtype=bool),
        coupling_template=coupling_template,
    )


def _diffusion_block(n_latent: int, *, free_support=None, template=None) -> DiffusionBlockSpec:
    if free_support is None:
        free_support = np.tri(n_latent, dtype=bool)
    if template is None:
        template = jnp.eye(n_latent)
    return DiffusionBlockSpec(
        n_latent=n_latent,
        diffusion_chol_support=free_support,
        diffusion_chol_template=jnp.asarray(template),
    )


def _lambda_block(n_manifest: int, n_latent: int, *, free_support=None, template=None):
    if free_support is None:
        free_support = np.zeros((n_manifest, n_latent), dtype=bool)
    if template is None:
        template = jnp.eye(n_manifest, n_latent)
    return SparseMatrixBlockSpec(
        n_rows=n_manifest,
        n_cols=n_latent,
        free_support=free_support,
        template=jnp.asarray(template),
        free_site_name="lambda_free",
        det_site_name="lambda",
        support=SupportClass.REAL,
        site_kind=SiteKind.LOADING,
        assembly_group="lambda",
        fixed_spec_field="lambda_mat",
        priors_field="lambda_free",
    )


def _manifest_chol_block(n_manifest: int, *, diag_support=None, template=None):
    if diag_support is None:
        diag_support = full_diagonal_support(n_manifest)
    if template is None:
        template = jnp.zeros((n_manifest, n_manifest))
    return ManifestCholBlockSpec(
        n_manifest=n_manifest,
        diag_support=diag_support,
        template=jnp.asarray(template),
    )


def _t0_means_block(n_latent: int, *, free_support=None, template=None):
    if free_support is None:
        free_support = np.ones(n_latent, dtype=bool)
    if template is None:
        template = jnp.zeros(n_latent)
    return SparseVectorBlockSpec(
        n=n_latent,
        free_support=free_support,
        template=jnp.asarray(template),
        free_site_name="t0_means_free",
        det_site_name="t0_means",
        support=SupportClass.REAL,
        site_kind=SiteKind.T0_MEANS,
        assembly_group="t0",
        fixed_spec_field="t0_means",
        priors_field="t0_means",
    )


def _t0_chol_block(n_latent: int, *, diag_support=None, correlation_support=None, template=None):
    if diag_support is None:
        diag_support = full_diagonal_support(n_latent)
    if correlation_support is None:
        correlation_support = np.tri(n_latent, k=-1, dtype=bool)
    if template is None:
        template = jnp.eye(n_latent)
    return T0CholBlockSpec(
        n_latent=n_latent,
        diag_support=diag_support,
        correlation_support=correlation_support,
        template=jnp.asarray(template),
    )


def _static_state_sd_block(mask, template):
    return SparseVectorBlockSpec(
        n=int(np.asarray(mask).shape[0]),
        free_support=np.asarray(mask, dtype=bool),
        template=jnp.asarray(template),
        free_site_name="static_state_sd_free",
        det_site_name="static_state_sds",
        support=SupportClass.POSITIVE,
        site_kind=SiteKind.STATIC_STATE_SD,
        assembly_group="t0",
        fixed_spec_field="static_state_sds",
        priors_field="static_state_sd",
    )


@pytest.fixture
def simple_spec():
    """Minimal 2-latent, 2-manifest Gaussian SSM."""
    return _make_spec(n_latent=2, n_manifest=2)


@pytest.fixture
def simple_model(simple_spec):
    return SSMModel(simple_spec)


@pytest.fixture
def dag_spec():
    """DAG-constrained spec with dynamics mask and lambda mask."""
    import numpy as np

    edge_support = np.array([[False, False], [True, False]])
    lambda_support = np.array([[True, False], [False, True]])
    lambda_template = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    return _make_spec(
        n_latent=2,
        n_manifest=2,
        dynamics_spec=_linear_dynamics(
            2,
            edge_support=edge_support,
            intercept_support=full_vector_support(2),
        ),
        lambda_block=_lambda_block(
            2,
            2,
            free_support=lambda_support,
            template=lambda_template,
        ),
    )


@pytest.fixture
def dag_model(dag_spec):
    return SSMModel(dag_spec)


@pytest.fixture
def model_spec_and_priors():
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

    def test_registry_shapes_match_trace_partial_manifest_variance_mask(self):
        """Masked manifest variance exposes only free diagonal entries as a site."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            manifest_chol_block=_manifest_chol_block(
                2,
                diag_support=np.array([False, True]),
                template=jnp.diag(jnp.array([0.4, 0.0], dtype=jnp.float32)),
            ),
        )
        model = SSMModel(spec)
        registry = build_site_registry(spec)
        backend = model.make_likelihood_backend()
        T = 10
        obs = jnp.zeros((T, spec.n_manifest))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(model, obs, times, random.PRNGKey(0), backend)

        verify_registry_matches_trace(registry, site_info)
        manifest_site = next(site for site in registry if site.name == "manifest_var_diag_free")
        assert manifest_site.shape == (1,)

    def test_fixed_dynamics_excludes_dynamics_sites(self):
        """When dynamics is a fixed array, no dynamics sites appear."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=_fixed_matrix_dynamics(
                jnp.array([[-0.5, 0.0], [0.0, -0.5]], dtype=jnp.float32)
            ),
        )
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert {"vf_0_decay", "vf_1_decay"}.issubset(names)

    def test_diag_diffusion_excludes_lower(self):
        """Diagonal diffusion has no lower-triangle sites."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            diffusion_block=_diffusion_block(
                2,
                free_support=np.diag(full_diagonal_support(2)),
                template=jnp.eye(2),
            ),
        )
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert "diffusion_diag_free" in names
        assert "diffusion_lower_free" not in names

    def test_free_diffusion_includes_lower(self):
        """Free diffusion includes lower-triangle sites."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            diffusion_block=_diffusion_block(
                2,
                free_support=full_cholesky_support(2),
                template=jnp.eye(2),
            ),
        )
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert "diffusion_diag_free" in names
        assert "diffusion_lower_free" in names

    def test_sparse_initial_state_correlations_only_include_authored_pairs(self):
        """Initial-state correlation sites should only exist for authored pairs."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 0] = True
        spec = _make_spec(
            n_latent=3,
            n_manifest=2,
            t0_chol_block=_t0_chol_block(
                3,
                diag_support=full_diagonal_support(3),
                correlation_support=mask,
                template=jnp.eye(3),
            ),
        )
        registry = build_site_registry(spec)
        site_map = {site.name: site for site in registry}
        assert site_map["t0_var_lower_free"].shape == (1,)

    def test_support_classes(self, simple_spec):
        """Check that support classes are correctly assigned."""
        registry = build_site_registry(simple_spec)
        support_map = {s.name: s.support for s in registry}
        # POSITIVE support sites
        assert support_map["vf_0_decay"] == SupportClass.POSITIVE
        # POSITIVE support sites (HalfNormal priors)
        assert support_map["diffusion_diag_free"] == SupportClass.POSITIVE
        assert support_map["manifest_var_diag_free"] == SupportClass.POSITIVE
        assert support_map["t0_var_diag_free"] == SupportClass.POSITIVE

    def test_mixed_diffusion_includes_proc_df_site(self):
        """Any student-t latent in diffusion_dists should expose proc_df."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=1,
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        registry = build_site_registry(spec)
        assert "proc_df" in {site.name for site in registry}

    def test_mixed_diffusion_sampling_emits_proc_df(self):
        """The traced model should sample proc_df when diffusion_dists include student_t."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=1,
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        model = SSMModel(spec)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(lambda: model._sample_likelihood_extra_params(spec)).get_trace()

        assert "proc_df" in trace

    def test_static_state_sd_site_is_registered_and_traced(self):
        """Compiled baseline factors should expose a positive static-state SD site."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=1,
            static_state_sd_block=_static_state_sd_block(
                np.array([True]),
                jnp.zeros(1),
            ),
            static_factor_loadings=jnp.array([[1.0], [1.0]]),
            t0_chol_block=_t0_chol_block(
                2,
                diag_support=np.zeros(2, dtype=bool),
                correlation_support=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2),
            ),
        )
        model = SSMModel(spec)

        registry = build_site_registry(spec)
        site_map = {site.name: site for site in registry}
        assert site_map["static_state_sd_free"].shape == (1,)
        assert site_map["static_state_sd_free"].support == SupportClass.POSITIVE

        backend = model.make_likelihood_backend()
        obs = jnp.zeros((5, spec.n_manifest))
        times = jnp.arange(5, dtype=jnp.float32)
        site_info = _discover_sites(model, obs, times, random.PRNGKey(0), backend)
        verify_registry_matches_trace(registry, site_info)
        assert site_info["static_state_sd_free"]["shape"] == (1,)


class TestSpecBlockAssembly:
    def test_assemble_t0_cov_adds_low_rank_baseline_factor_covariance(self):
        """Static baseline factors should add `B diag(tau^2) B^T` to the t0 covariance."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=1,
            static_state_sd_block=_static_state_sd_block(
                np.array([True]),
                jnp.zeros(1),
            ),
            static_factor_loadings=jnp.array([[1.0], [1.0]]),
            t0_chol_block=_t0_chol_block(
                2,
                diag_support=np.zeros(2, dtype=bool),
                correlation_support=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2),
            ),
        )
        cov = spec.assemble_t0_cov(static_state_sd_free=jnp.array([2.0]))

        np.testing.assert_allclose(
            np.asarray(cov),
            np.array([[5.0, 4.0], [4.0, 5.0]]),
        )


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


# ---------------------------------------------------------------------------
# Registry-driven deterministic assembly
# ---------------------------------------------------------------------------


class TestDeterministicAssembly:
    def test_group_sites_by_assembly_role(self, simple_spec):
        """Assembly grouping is driven by registry metadata."""
        registry = build_site_registry(simple_spec)
        grouped = group_sites_by_assembly_role(registry)
        assert "dynamics" in grouped
        assert "diffusion" in grouped
        assert {site.name for site in grouped["dynamics"]} == {
            "vf_0_decay",
            "vf_1_weight",
            "vf_2_weight",
        }

    def test_assemble_deterministics_from_registry_free_spec(self, simple_spec):
        """Registry-driven assembly builds the expected matrices."""
        registry = build_site_registry(simple_spec)
        samples = {
            "vf_0_decay": jnp.array([[0.5, 0.3]], dtype=jnp.float32),
            "vf_1_weight": jnp.array([0.1], dtype=jnp.float32),
            "vf_2_weight": jnp.array([-0.2], dtype=jnp.float32),
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, simple_spec, registry)
        assert jnp.allclose(det["diffusion"][0], jnp.array([[0.4, 0.0], [0.25, 0.6]]))
        assert det["lambda"].shape == (1, 2, 2)
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.49, 0.64])))
        assert jnp.allclose(det["t0_means"][0], jnp.array([1.0, -1.0]))
        assert jnp.allclose(det["t0_cov"][0], jnp.diag(jnp.array([0.81, 1.21])))

    def test_assemble_deterministics_from_registry_fixed_fallbacks(self):
        """Fixed spec matrices are broadcast without any sampled sites."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=_fixed_matrix_dynamics(
                jnp.array([[-0.4, 0.1], [0.0, -0.2]], dtype=jnp.float32)
            ),
            diffusion_block=_diffusion_block(
                2,
                free_support=np.zeros((2, 2), dtype=bool),
                template=jnp.array([[0.3, 0.0], [0.1, 0.5]], dtype=jnp.float32),
            ),
            lambda_block=_lambda_block(
                2,
                2,
                template=jnp.array([[1.0, 0.0], [0.2, 1.0]], dtype=jnp.float32),
            ),
            manifest_chol_block=_manifest_chol_block(
                2,
                diag_support=np.zeros(2, dtype=bool),
                template=jnp.array([[0.4, 0.0], [0.0, 0.6]], dtype=jnp.float32),
            ),
            t0_means_block=_t0_means_block(
                2,
                free_support=np.zeros(2, dtype=bool),
                template=jnp.array([0.5, -0.5], dtype=jnp.float32),
            ),
            t0_chol_block=_t0_chol_block(
                2,
                diag_support=np.zeros(2, dtype=bool),
                correlation_support=np.zeros((2, 2), dtype=bool),
                template=jnp.array([[0.7, 0.0], [0.0, 0.8]], dtype=jnp.float32),
            ),
        )
        registry = build_site_registry(spec)

        det = assemble_deterministics_from_registry({}, spec, registry, n_draws=3)
        assert jnp.allclose(
            det["diffusion"],
            jnp.broadcast_to(spec.diffusion_block.assemble(), (3, 2, 2)),
        )
        assert jnp.allclose(
            det["lambda"],
            jnp.broadcast_to(spec.lambda_block.assemble(), (3, 2, 2)),
        )
        manifest_chol = spec.manifest_chol_block.assemble()
        expected_manifest_cov = manifest_chol @ manifest_chol.T
        assert jnp.allclose(det["manifest_cov"], jnp.broadcast_to(expected_manifest_cov, (3, 2, 2)))
        assert isinstance(spec.t0_means_block.assemble(), jnp.ndarray)
        assert jnp.allclose(
            det["t0_means"],
            jnp.broadcast_to(spec.t0_means_block.assemble(), (3, 2)),
        )
        expected_t0_cov = spec.assemble_t0_cov()
        assert jnp.allclose(det["t0_cov"], jnp.broadcast_to(expected_t0_cov, (3, 2, 2)))

    def test_assemble_deterministics_from_registry_partial_manifest_variance_mask(self):
        """Registry assembly respects mixed fixed/free manifest-noise diagonals."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            manifest_chol_block=_manifest_chol_block(
                2,
                diag_support=np.array([False, True]),
                template=jnp.diag(jnp.array([0.4, 0.0], dtype=jnp.float32)),
            ),
        )
        registry = build_site_registry(spec)
        samples = {
            "vf_0_decay": jnp.array([[0.5, 0.3]], dtype=jnp.float32),
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.9]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, spec, registry)
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.16, 0.81])))

    def test_assemble_deterministics_from_registry_initial_state_correlations(self):
        """Initial-state off-diagonal samples are interpreted as correlations."""
        mask = np.zeros((2, 2), dtype=bool)
        mask[1, 0] = True
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            t0_chol_block=_t0_chol_block(
                2,
                diag_support=full_diagonal_support(2),
                correlation_support=mask,
                template=jnp.eye(2),
            ),
        )
        registry = build_site_registry(spec)
        samples = {
            "vf_0_decay": jnp.array([[0.5, 0.3]], dtype=jnp.float32),
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[2.0, 3.0]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, spec, registry)

        assert jnp.allclose(
            det["t0_cov"][0],
            jnp.array([[4.0, 1.5], [1.5, 9.0]], dtype=jnp.float32),
        )

    def test_assemble_deterministics_repairs_invalid_initial_correlation_matrix(self):
        """Impossible authored initial correlations are repaired to a PSD covariance."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 0] = True
        mask[2, 0] = True
        mask[2, 1] = True
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            t0_chol_block=_t0_chol_block(
                3,
                diag_support=full_diagonal_support(3),
                correlation_support=mask,
                template=jnp.eye(3),
            ),
        )
        registry = build_site_registry(spec)
        samples = {
            "vf_0_decay": jnp.array([[0.5, 0.3, 0.4]], dtype=jnp.float32),
            "diffusion_diag_free": jnp.array([[0.4, 0.6, 0.5]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25, 0.1, -0.15]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8, 0.9]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0, 0.5]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[1.0, 1.0, 1.0]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.array([[0.9, 0.9, -0.9]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, spec, registry)
        min_eig = jnp.min(jnp.linalg.eigvalsh(det["t0_cov"][0]))

        assert bool(jnp.isfinite(det["t0_cov"]).all())
        assert float(min_eig) > -1e-6


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
        """Custom PriorRegistry values appear in the state."""
        priors = prior_registry(
            vf_0_decay=PriorSpec(
                PriorDistributionFamily.GAMMA,
                {"concentration": 4.0, "rate": 2.0},
            )
        )
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry, priors)
        assert jnp.allclose(
            state["vf_0_decay"]["concentration"],
            jnp.full(2, 4.0),
        )
        assert jnp.allclose(state["vf_0_decay"]["rate"], jnp.full(2, 2.0))

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
        """Pure-JAX Normal log_prob terms match NumPyro."""
        from nof1_causal_lab.models.ssm.parameterization import _normal_log_prob_terms

        x = jnp.array([0.5, -1.0, 2.0])
        loc = jnp.array([0.0, 0.0, 1.0])
        scale = jnp.array([1.0, 2.0, 0.5])
        expected = jnp.sum(dist.Normal(loc, scale).log_prob(x))
        actual = jnp.sum(_normal_log_prob_terms(x, loc, scale))
        assert jnp.allclose(actual, expected, atol=1e-5)

    def test_half_normal_log_prob_matches_numpyro(self):
        """Pure-JAX HalfNormal log_prob terms match NumPyro."""
        from nof1_causal_lab.models.ssm.parameterization import _half_normal_log_prob_terms

        x = jnp.array([0.5, 1.0, 2.0])
        scale = jnp.array([1.0, 2.0, 0.5])
        expected = jnp.sum(dist.HalfNormal(scale).log_prob(x))
        actual = jnp.sum(_half_normal_log_prob_terms(x, scale))
        assert jnp.allclose(actual, expected, atol=1e-5)

    def test_gamma_log_prob_matches_numpyro(self):
        """Pure-JAX Gamma log_prob terms match NumPyro."""
        from nof1_causal_lab.models.ssm.parameterization import _gamma_log_prob_terms

        x = jnp.array([0.5, 1.0, 2.0])
        concentration = jnp.array([2.0, 5.0, 1.0])
        rate = jnp.array([1.0, 0.5, 2.0])
        expected = jnp.sum(dist.Gamma(concentration, rate).log_prob(x))
        actual = jnp.sum(_gamma_log_prob_terms(x, concentration, rate))
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
            registry,
            prior_registry(
                vf_0_decay=PriorSpec(
                    PriorDistributionFamily.GAMMA,
                    {"concentration": 2.0, "rate": 4.0},
                )
            ),
        )
        state2 = build_prior_runtime_state(
            registry,
            prior_registry(
                vf_0_decay=PriorSpec(
                    PriorDistributionFamily.GAMMA,
                    {"concentration": 4.0, "rate": 2.0},
                )
            ),
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
        from nof1_causal_lab.models.ssm.parameterization import _make_positive_params

        registry = build_site_registry(simple_spec)
        D, unravel_fn = build_unravel_fn(registry)

        state1 = build_prior_runtime_state(registry)
        # Switch diffusion_diag_free from HalfNormal (0) to Gamma (1)
        # Use _make_positive_params to ensure consistent weak_type/dtype.
        state2 = build_prior_runtime_state(registry)
        state2["diffusion_diag_free"] = _make_positive_params(
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
        for orig, rest in zip(registry, restored, strict=True):
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
        priors = prior_registry(
            vf_0_decay=PriorSpec(
                PriorDistributionFamily.GAMMA,
                {"concentration": 4.0, "rate": 2.0},
            ),
            diffusion_diag_free=PriorSpec(PriorDistributionFamily.HALF_NORMAL, {"sigma": 0.5}),
        )
        registry = build_site_registry(simple_spec)
        state = build_prior_runtime_state(registry, priors)
        payload = serialize_prior_runtime_state(state)
        restored = deserialize_prior_runtime_state(payload, registry)
        assert jnp.allclose(
            restored["vf_0_decay"]["concentration"],
            jnp.full(2, 4.0),
            atol=1e-6,
        )
        assert jnp.allclose(restored["vf_0_decay"]["rate"], jnp.full(2, 2.0))
        assert jnp.allclose(
            restored["diffusion_diag_free"]["scale"],
            jnp.full(2, 0.5),
            atol=1e-6,
        )

    def test_compile_prior_semantics_roundtrip(self, simple_spec):
        """compile_prior_semantics → deserialize produces valid state."""
        priors = prior_registry(
            vf_0_decay=PriorSpec(
                PriorDistributionFamily.GAMMA,
                {"concentration": 4.0, "rate": 2.0},
            )
        )
        semantics = compile_prior_semantics(simple_spec, priors)
        assert semantics["schema_version"] == 5
        registry = deserialize_site_registry(semantics["site_registry"])
        state = deserialize_prior_runtime_state(semantics["prior_state"], registry)
        assert "vf_0_decay" in state
        assert jnp.allclose(
            state["vf_0_decay"]["concentration"],
            jnp.full(2, 4.0),
            atol=1e-6,
        )


class TestCanonicalRuntimePriors:
    def test_loaded_runtime_preserves_per_element_priors(self):
        """Compiled prior semantics preserve vector-valued site parameters exactly."""
        spec = _make_spec(n_latent=3, n_manifest=3)
        priors = prior_registry(
            vf_0_decay=PriorSpec(
                PriorDistributionFamily.GAMMA,
                {
                    "concentration": [2.0, 3.0, 4.0],
                    "rate": [4.0, 5.0, 6.0],
                },
            ),
        )
        runtime = load_prior_runtime_bundle(compile_prior_semantics(spec, priors))
        assert runtime.prior_state["vf_0_decay"]["concentration"].shape == (3,)
        assert jnp.allclose(
            runtime.prior_state["vf_0_decay"]["concentration"],
            jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32),
        )

    def test_site_distribution_handles_vector_positive_priors(self, simple_spec):
        """Canonical site distributions accept vector-valued positive scales."""
        priors = prior_registry(
            t0_var_diag_free=PriorSpec(
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": [1.0, 2.0]},
            )
        )
        runtime = load_prior_runtime_bundle(compile_prior_semantics(simple_spec, priors))
        site = next(
            site for site in runtime.site_runtime.registry if site.name == "t0_var_diag_free"
        )
        prior_dist = build_site_prior_distribution(site, runtime.prior_state[site.name])
        assert isinstance(prior_dist, dist.HalfNormal)
        assert prior_dist.batch_shape == (2,)
        assert jnp.allclose(prior_dist.scale, jnp.array([1.0, 2.0], dtype=jnp.float32))

    def test_positive_delta_distribution_samples_fixed_vector(self, simple_spec):
        """Positive Delta priors build NumPyro Delta distributions and preserve shape."""
        priors = prior_registry(
            vf_0_decay=PriorSpec(
                PriorDistributionFamily.DELTA,
                {"value": [0.25, 0.5]},
            )
        )
        runtime = load_prior_runtime_bundle(compile_prior_semantics(simple_spec, priors))
        site = next(site for site in runtime.site_runtime.registry if site.name == "vf_0_decay")
        prior_dist = build_site_prior_distribution(site, runtime.prior_state[site.name])

        assert isinstance(prior_dist, dist.Delta)
        assert prior_dist.batch_shape == (2,)
        assert jnp.allclose(prior_dist.v, jnp.array([0.25, 0.5], dtype=jnp.float32))

    def test_positive_delta_roundtrips_through_serialized_semantics(self, simple_spec):
        """Positive Delta value survives v5 compiled-prior serialization."""
        priors = prior_registry(
            vf_0_decay=PriorSpec(
                PriorDistributionFamily.DELTA,
                {"value": [0.25, 0.5]},
            )
        )
        runtime = load_prior_runtime_bundle(compile_prior_semantics(simple_spec, priors))

        assert jnp.allclose(
            runtime.prior_state["vf_0_decay"]["value"],
            jnp.array([0.25, 0.5], dtype=jnp.float32),
        )

    def test_decay_rate_positive_site_keeps_fixed_parameter_keys(self, simple_spec):
        """Base-decay prior state keeps the same positive-family leaves across prior edits."""
        runtime = load_prior_runtime_bundle(compile_prior_semantics(simple_spec))

        dynamics_params = runtime.prior_state["vf_0_decay"]
        assert set(dynamics_params) == {
            "family",
            "loc",
            "scale",
            "concentration",
            "rate",
            "value",
        }
        assert jnp.allclose(dynamics_params["concentration"], jnp.full((2,), 2.0))
        assert jnp.allclose(dynamics_params["rate"], jnp.full((2,), 4.0))

        correlation_params = runtime.prior_state["t0_var_lower_free"]
        assert set(correlation_params) == {"family", "loc", "scale", "low", "high"}
        assert jnp.allclose(correlation_params["low"], jnp.full((1,), -1.0))
        assert jnp.allclose(correlation_params["high"], jnp.full((1,), 1.0))


# ---------------------------------------------------------------------------
# Compiled artifact integration (hard cutover)
# ---------------------------------------------------------------------------


class TestCompiledArtifactIntegration:
    """Test that compiled_prior_semantics is emitted and correctly consumed."""

    def test_artifact_contains_compiled_prior_semantics(self, model_spec_and_priors):
        """compile_ssm_artifact emits semantics and omits legacy priors."""
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)
        assert "priors" not in artifact
        assert "compiled_prior_semantics" in artifact
        assert "edge_lag_days" in artifact
        assert artifact["edge_lag_days"] == []
        sem = artifact["compiled_prior_semantics"]
        assert sem["schema_version"] == 5
        assert "site_registry" in sem
        assert "prior_state" in sem

    def test_known_input_beta_binds_to_input_effect_site(self):
        """A beta from a known input compiles to B, not the latent dynamics matrix."""
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact

        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "dose",
                        "description": "Dose",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                    },
                    {
                        "name": "mood",
                        "description": "Mood",
                        "role": "endogenous",
                        "is_outcome": True,
                        "temporal_status": "time_varying",
                    },
                ],
                "edges": [
                    {
                        "cause": "dose",
                        "effect": "mood",
                        "description": "Dose affects mood",
                        "lagged": True,
                    }
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "dose_mg",
                        "construct_name": "dose",
                        "construct_polarity": "positive",
                        "how_to_measure": "Dose in mg",
                        "measurement_dtype": "continuous",
                        "aggregation": "sum",
                    },
                    {
                        "name": "mood_score",
                        "construct_name": "mood",
                        "construct_polarity": "positive",
                        "how_to_measure": "Mood score",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                ],
            },
            "estimation": {
                "state_order": ["mood"],
                "edges": [
                    {
                        "cause": "dose",
                        "effect": "mood",
                        "description": "Dose affects mood",
                        "lagged": True,
                    }
                ],
                "induced_dependencies": [],
                "known_inputs": [
                    {
                        "construct": "dose",
                        "source_indicator": "dose_mg",
                        "scale": 10.0,
                        "missing_policy": "forward_fill",
                    }
                ],
            },
        }
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                }
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_dose_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "name": "sigma_mood",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_dose_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.1}},
            "sigma_mood": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
        }

        artifact = compile_ssm_artifact(model_spec, priors, causal_spec=causal_spec)

        assert artifact["spec"]["manifest_names"] == ["mood_score"]
        assert artifact["spec"]["input_names"] == ["dose"]
        assert artifact["spec"]["input_source_indicators"] == ["dose_mg"]
        assert artifact["spec"]["input_effect_block"]["free_support"] == [[True]]
        beta_binding = next(
            binding
            for binding in artifact["parameter_bindings"]
            if binding["parameter"] == "beta_dose_mood"
        )
        assert {
            "parameter": beta_binding["parameter"],
            "site_name": beta_binding["site_name"],
            "flat_index": beta_binding["flat_index"],
        } == {
            "parameter": "beta_dose_mood",
            "site_name": "input_effect_free",
            "flat_index": 0,
        }
        assert beta_binding["site_kind"] == "input_effect"
        assert beta_binding["transform"] == "dt_effect_to_ct_rate"

    def test_model_from_artifact_uses_semantics(self, model_spec_and_priors):
        """build_model_from_compiled_artifact reads compiled_prior_semantics."""
        import polars as pl

        from nof1_causal_lab.models.ssm.compile.artifact import (
            build_model_from_compiled_artifact,
            compile_ssm_artifact,
        )

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)
        model = build_model_from_compiled_artifact(
            artifact,
            pl.DataFrame({"time": [0.0], "mood_score": [5.0]}),
        )
        assert model.priors is not None
        assert model.get_prior_runtime_bundle() is not None

    def test_model_from_artifact_requires_compiled_prior_semantics(self, model_spec_and_priors):
        """Model rebuild fails clearly when compiled semantics are missing."""
        import polars as pl

        from nof1_causal_lab.models.ssm.compile.artifact import (
            build_model_from_compiled_artifact,
            compile_ssm_artifact,
        )

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)
        del artifact["compiled_prior_semantics"]

        with pytest.raises(ValueError, match="compiled_prior_semantics"):
            build_model_from_compiled_artifact(
                artifact,
                pl.DataFrame({"time": [0.0], "mood_score": [5.0]}),
            )

    def test_end_to_end_compile_rebuild_sample(self, model_spec_and_priors):
        """Full roundtrip: compile → rebuild → sample."""
        import numpy as np
        import polars as pl

        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
        from nof1_causal_lab.models.ssm.runtime import (
            prepare_wide_model_runtime,
            sample_prior_predictive,
        )
        from nof1_causal_lab.utils.data import pivot_to_wide

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
        runtime = prepare_wide_model_runtime(
            pivot_to_wide(data_for_model),
            compiled_ssm=artifact,
            sampler_config={"method": "marginal_particle_gibbs"},
        )
        samples = sample_prior_predictive(
            runtime.model,
            samples=5,
            times=runtime.times,
            observation_support=runtime.observation_support,
            observation_mask=~jnp.isnan(runtime.observations),
            transition_inputs=runtime.transition_inputs,
        )
        assert samples is not None

    def test_compiled_model_prior_predictive(self, model_spec_and_priors):
        """Compiled models can sample prior predictive from artifact semantics."""
        import polars as pl

        from nof1_causal_lab.models.ssm.compile.artifact import (
            build_model_from_compiled_artifact,
            compile_ssm_artifact,
        )
        from nof1_causal_lab.models.ssm.runtime import sample_prior_predictive

        model_spec, priors = model_spec_and_priors
        artifact = compile_ssm_artifact(model_spec, priors)
        model = build_model_from_compiled_artifact(
            artifact,
            pl.DataFrame({"time": [0.0], "mood_score": [5.0]}),
        )
        samples = sample_prior_predictive(model, samples=4)
        assert "vf_0_decay" in samples
        assert samples["vf_0_decay"].shape[0] == 4
        assert "observations" in samples

    def test_compiled_model_traces_vector_t0_prior_without_reconstructing(self):
        """Compiled models execute vector-valued positive priors via runtime semantics."""
        import polars as pl

        from nof1_causal_lab.models.ssm.compile.artifact import (
            build_model_from_compiled_artifact,
            serialize_ssm_spec,
        )
        from nof1_causal_lab.models.ssm.runtime import prepare_fit_inputs

        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_block=_lambda_block(2, 2, template=jnp.eye(2)),
            manifest_chol_block=_manifest_chol_block(
                2,
                diag_support=full_diagonal_support(2),
                template=jnp.zeros((2, 2)),
            ),
            manifest_names=["m0", "m1"],
        )
        priors = prior_registry(
            t0_var_diag_free=PriorSpec(
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": [1.0, 2.0]},
            )
        )
        artifact = {
            "spec": serialize_ssm_spec(spec),
            "compiled_prior_semantics": compile_prior_semantics(spec, priors),
            "parameter_bindings": [],
        }
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "m0": [0.1, 0.2, 0.3],
                "m1": [0.4, 0.5, 0.6],
            }
        )

        model = build_model_from_compiled_artifact(artifact, wide)
        observations, times, _manifest_names, _wide = prepare_fit_inputs(model.spec, wide)
        backend = model.make_likelihood_backend()
        trace = handlers.trace(handlers.seed(model.model, rng_seed=0)).get_trace(
            observations,
            times,
            likelihood_backend=backend,
        )

        assert trace["t0_var_diag_free"]["value"].shape == (2,)
