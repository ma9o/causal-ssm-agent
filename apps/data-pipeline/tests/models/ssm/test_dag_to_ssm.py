"""Tests for DAG-to-SSM constraint propagation (Fixes 1-3).

Tests that:
1. dynamics_support constrains off-diagonal sampling to causal edges only
2. lambda_support + template constrains factor loadings to measurement model
3. Per-element priors align with mask positions
4. Builder constructs structural support from CausalSpec
5. Pipeline threading passes causal_spec through
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.handlers as handlers
import polars as pl
import pytest

from nof1_causal_lab.artifacts import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily, PriorDistributionFamily
from nof1_causal_lab.models.ssm.model import SSMModel, SSMSpec
from nof1_causal_lab.models.ssm.parameterization import (
    SiteDescriptor,
    TransformKind,
    build_prior_runtime_state,
    build_site_prior_distribution,
)
from nof1_causal_lab.models.ssm.priors import PriorSpec
from nof1_causal_lab.models.ssm.structure import SparseMatrixBlockSpec
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from nof1_causal_lab.models.ssm.testing import (
    block_ssm_spec,
    dense_matrix_dynamics_spec,
    full_vector_support,
    prior_registry,
    zero_loading_support,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


def _make_3latent_spec(
    edge_support: np.ndarray | None = None,
    lambda_block: SparseMatrixBlockSpec | None = None,
) -> SSMSpec:
    """3 latent, 4 manifest spec with optional masks."""
    n_l, n_m = 3, 4
    if edge_support is None:
        edge_support = np.ones((n_l, n_l), dtype=bool)
        np.fill_diagonal(edge_support, False)
    if lambda_block is None:
        lambda_block = SparseMatrixBlockSpec(
            n_rows=n_m,
            n_cols=n_l,
            free_support=zero_loading_support(n_m, n_l),
            template=jnp.eye(n_m, n_l),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        )
    return block_ssm_spec(
        n_latent=n_l,
        n_manifest=n_m,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=n_l,
            decay_support=np.ones(n_l, dtype=bool),
            edge_support=edge_support,
            coupling_template=jnp.zeros((n_l, n_l)),
            intercept_support=np.zeros(n_l, dtype=bool),
            cint_template=jnp.zeros(n_l),
        ),
        lambda_block=lambda_block,
        latent_names=["X", "Y", "Z"],
        manifest_names=["x1", "x2", "y1", "z1"],
    )


def _make_causal_spec_dict() -> dict:
    """Minimal CausalSpec dict: X→Y, Y→Z, 4 indicators."""
    return {
        "latent": {
            "constructs": [
                {
                    "name": "X",
                    "description": "Cause",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "Y",
                    "description": "Mediator",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                },
                {
                    "name": "Z",
                    "description": "Downstream",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
            ],
            "edges": [
                {
                    "cause": "X",
                    "effect": "Y",
                    "description": "X causes Y",
                    "lagged": True,
                },
                {
                    "cause": "Y",
                    "effect": "Z",
                    "description": "Y causes Z",
                    "lagged": True,
                },
            ],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "x1",
                    "construct_name": "X",
                    "construct_polarity": "positive",
                    "how_to_measure": "measure x",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "x2",
                    "construct_name": "X",
                    "construct_polarity": "positive",
                    "how_to_measure": "measure x alt",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "y1",
                    "construct_name": "Y",
                    "construct_polarity": "positive",
                    "how_to_measure": "measure y",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "z1",
                    "construct_name": "Z",
                    "construct_polarity": "positive",
                    "how_to_measure": "measure z",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
        },
        "estimation": {
            "state_order": ["X", "Y", "Z"],
            "edges": [
                {
                    "cause": "X",
                    "effect": "Y",
                    "description": "X causes Y",
                    "lagged": True,
                },
                {
                    "cause": "Y",
                    "effect": "Z",
                    "description": "Y causes Z",
                    "lagged": True,
                },
            ],
            "induced_dependencies": [],
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Fix 1: DAG-constrained dynamics
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicsMask:
    """Test that component translation constrains linear edge sampling."""

    def test_dynamics_support_zeros_non_edges(self):
        """Dynamics entries where mask is False should be zero."""
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y
        offdiag_support[2, 1] = True  # Y→Z

        spec = _make_3latent_spec(edge_support=offdiag_support)
        model = SSMModel(spec)

        rng = random.PRNGKey(42)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        weight_sites = sorted(
            name for name in trace if name.startswith("vf_") and name.endswith("_weight")
        )
        assert weight_sites == ["vf_1_weight", "vf_2_weight"]

    def test_no_mask_fully_free(self):
        """Default dynamics mask expands to a fully free dynamics structure."""
        spec = _make_3latent_spec()
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        weight_sites = [
            name for name in trace if name.startswith("vf_") and name.endswith("_weight")
        ]
        assert len(weight_sites) == 6

    def test_dynamics_support_single_latent(self):
        """Single latent: no off-diagonal, mask should be identity."""
        spec = block_ssm_spec(
            n_latent=1,
            n_manifest=1,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=1,
                decay_support=np.ones(1, dtype=bool),
                edge_support=np.zeros((1, 1), dtype=bool),
                coupling_template=jnp.zeros((1, 1)),
                intercept_support=np.zeros(1, dtype=bool),
                cint_template=jnp.zeros(1),
            ),
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((5, 1)),
            times=jnp.arange(5, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        assert not any(name.startswith("vf_") and name.endswith("_weight") for name in trace)


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Structured lambda
# ═══════════════════════════════════════════════════════════════════════


class TestLambdaMask:
    """Test that lambda_support constrains factor loadings."""

    def test_lambda_template_plus_mask(self):
        """Template+mask mode: fixed reference + free additional loadings."""
        # X has 2 indicators (x1 ref, x2 free), Y has 1, Z has 1
        lambda_mat = jnp.zeros((4, 3))
        lambda_mat = lambda_mat.at[0, 0].set(1.0)  # x1→X (ref)
        lambda_mat = lambda_mat.at[2, 1].set(1.0)  # y1→Y (ref)
        lambda_mat = lambda_mat.at[3, 2].set(1.0)  # z1→Z (ref)

        lambda_support = np.zeros((4, 3), dtype=bool)
        lambda_support[1, 0] = True  # x2→X (free)

        spec = _make_3latent_spec(
            lambda_block=SparseMatrixBlockSpec(
                n_rows=4,
                n_cols=3,
                free_support=lambda_support,
                template=lambda_mat,
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            )
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # Only 1 free loading sampled
        assert trace["lambda_free"]["value"].shape == (1,)

        # Check the assembled lambda
        lam = trace["lambda"]["value"]
        assert float(lam[0, 0]) == 1.0  # Fixed reference
        assert float(lam[2, 1]) == 1.0  # Fixed reference
        assert float(lam[3, 2]) == 1.0  # Fixed reference
        assert float(lam[1, 0]) != 0.0  # Free loading was sampled

    def test_lambda_no_mask_returns_fixed(self):
        """Array lambda_mat with default zero free-mask is returned as-is."""
        lambda_mat = jnp.eye(4, 3)
        spec = _make_3latent_spec(
            lambda_block=SparseMatrixBlockSpec(
                n_rows=4,
                n_cols=3,
                free_support=zero_loading_support(4, 3),
                template=lambda_mat,
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            )
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # No lambda_free sampled
        assert "lambda_free" not in trace
        # Lambda deterministic IS emitted (the block always emits its
        # assembled output, fixed or sampled); the value equals the template.
        assert "lambda" in trace
        np.testing.assert_allclose(np.asarray(trace["lambda"]["value"]), np.asarray(lambda_mat))


# ═══════════════════════════════════════════════════════════════════════
# Fix 3: Per-element priors
# ═══════════════════════════════════════════════════════════════════════


class TestPerElementPriors:
    """Test array-valued priors through canonical site runtime state."""

    @staticmethod
    def _real_site(shape: tuple[int, ...]) -> SiteDescriptor:
        return SiteDescriptor(
            name="test_site",
            shape=shape,
            support=SupportClass.REAL,
            assembly_group="test",
            site_kind=SiteKind.DYNAMICS_WEIGHT,
            transform_kind=TransformKind.IDENTITY,
        )

    def test_make_prior_dist_scalar(self):
        """Scalar mu/sigma produces scalar Normal."""
        site = self._real_site(())
        priors = prior_registry(
            test_site=PriorSpec(PriorDistributionFamily.NORMAL, {"mu": 0.0, "sigma": 1.0})
        )
        state = build_prior_runtime_state([site], priors)
        d = build_site_prior_distribution(site, state[site.name])
        assert d.batch_shape == ()

    def test_make_prior_dist_array(self):
        """Array mu/sigma produces batched Normal."""
        site = self._real_site((3,))
        priors = prior_registry(
            test_site=PriorSpec(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2, 0.3], "sigma": [1.0, 0.5, 0.3]},
            )
        )
        state = build_prior_runtime_state([site], priors)
        d = build_site_prior_distribution(site, state[site.name])
        assert d.batch_shape == (3,)

    def test_make_prior_batch_scalar_expand(self):
        """Scalar prior expanded to batch shape."""
        site = self._real_site((5,))
        priors = prior_registry(
            test_site=PriorSpec(PriorDistributionFamily.NORMAL, {"mu": 0.0, "sigma": 1.0})
        )
        state = build_prior_runtime_state([site], priors)
        d = build_site_prior_distribution(site, state[site.name])
        assert d.batch_shape == (5,)

    def test_make_prior_batch_array_passthrough(self):
        """Array prior with correct shape passes through."""
        site = self._real_site((2,))
        priors = prior_registry(
            test_site=PriorSpec(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2], "sigma": [1.0, 0.5]},
            )
        )
        state = build_prior_runtime_state([site], priors)
        d = build_site_prior_distribution(site, state[site.name])
        assert d.batch_shape == (2,)

    def test_make_prior_batch_mismatch_raises(self):
        """Array prior with wrong shape raises."""
        site = self._real_site((3,))
        priors = prior_registry(
            test_site=PriorSpec(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2], "sigma": [1.0, 0.5]},
            )
        )
        with pytest.raises(ValueError, match="broadcast"):
            build_prior_runtime_state([site], priors)

    def test_per_element_prior_in_model(self):
        """Per-element dynamics priors are used in sampling."""
        offdiag_support = np.zeros((2, 2), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y

        spec = block_ssm_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=2,
                decay_support=np.ones(2, dtype=bool),
                edge_support=offdiag_support,
                coupling_template=jnp.zeros((2, 2)),
                intercept_support=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
            latent_names=["X", "Y"],
            manifest_names=["x1", "y1"],
        )

        # Per-element prior: single off-diagonal has mu=2.0
        priors = prior_registry(
            vf_1_weight=PriorSpec(
                PriorDistributionFamily.NORMAL,
                {"mu": 2.0, "sigma": 0.1},
            )
        )
        model = SSMModel(spec, priors)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((5, 2)),
            times=jnp.arange(5, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # The off-diagonal value should be near 2.0 (tight prior)
        weight = float(trace["vf_1_weight"]["value"])
        assert abs(weight - 2.0) < 1.0, f"Expected ~2.0, got {weight}"


# ═══════════════════════════════════════════════════════════════════════
# Runtime structural-support construction
# ═══════════════════════════════════════════════════════════════════════


class TestRuntimeStructuralSupport:
    """Test that compilation constructs correct block support from CausalSpec."""

    def test_build_structural_support_from_causal_spec(self):
        """Compilation constructs dynamics/lambda support from CausalSpec."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_causal_spec,
        )

        causal_spec = _make_causal_spec_dict()

        latent_names = ["X", "Y", "Z"]
        manifest_cols = ["x1", "x2", "y1", "z1"]

        dynamics_support, _input_effect_support, lambda_mat, lambda_support, _edge_lag_days = (
            build_structural_support_from_causal_spec(
                latent_names, manifest_cols, 3, 4, causal_spec=causal_spec
            )
        )

        # Dynamics mask: baseline persistence diagonals + X→Y + Y→Z
        assert dynamics_support is not None
        assert dynamics_support[0, 0]  # X baseline persistence
        assert dynamics_support[1, 1]  # Y self
        assert dynamics_support[2, 2]  # Z self
        assert dynamics_support[1, 0]  # X→Y (effect=Y row, cause=X col)
        assert dynamics_support[2, 1]  # Y→Z (effect=Z row, cause=Y col)
        assert not dynamics_support[0, 1]  # No Y→X edge
        assert not dynamics_support[0, 2]  # No Z→X edge
        assert not dynamics_support[1, 2]  # No Z→Y edge
        assert not dynamics_support[2, 0]  # No X→Z edge

        # Lambda: x1 fixed ref for X, x2 free for X, y1 fixed ref for Y, z1 fixed ref for Z
        assert float(lambda_mat[0, 0]) == 1.0  # x1→X
        assert float(lambda_mat[2, 1]) == 1.0  # y1→Y
        assert float(lambda_mat[3, 2]) == 1.0  # z1→Z

        assert lambda_support is not None
        assert lambda_support[1, 0]  # x2→X is free
        assert not lambda_support[0, 0]  # x1→X is fixed
        assert not lambda_support[2, 1]  # y1→Y is fixed

    def test_no_causal_spec_materializes_explicit_default_masks(self):
        """Without causal_spec, structural defaults are still explicit."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_causal_spec,
        )

        dynamics_support, input_effect_support, _lambda_mat, lambda_support, _edge_lag_days = (
            build_structural_support_from_causal_spec(None, ["x1"], 1, 1, causal_spec=None)
        )
        np.testing.assert_array_equal(dynamics_support, np.array([[True]]))
        np.testing.assert_array_equal(input_effect_support, np.zeros((1, 0), dtype=bool))
        np.testing.assert_array_equal(lambda_support, np.array([[False]]))

    def test_known_input_edge_compiles_to_input_effect_support(self):
        """Known inputs are transition drivers, not latent dynamics columns."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_causal_spec,
        )

        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "dose",
                        "description": "Medication dose",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                    },
                    {
                        "name": "mood",
                        "description": "Mood state",
                        "role": "endogenous",
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
                        "how_to_measure": "record dose",
                        "measurement_dtype": "continuous",
                        "aggregation": "sum",
                    },
                    {
                        "name": "mood_rating",
                        "construct_name": "mood",
                        "construct_polarity": "positive",
                        "how_to_measure": "record mood",
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
                        "missing_policy": "zero",
                    }
                ],
            },
        }

        dynamics_support, input_effect_support, lambda_mat, lambda_support, edge_lag_days = (
            build_structural_support_from_causal_spec(
                ["mood"],
                ["mood_rating"],
                1,
                1,
                causal_spec=causal_spec,
            )
        )

        np.testing.assert_array_equal(dynamics_support, np.array([[True]]))
        np.testing.assert_array_equal(input_effect_support, np.array([[True]]))
        np.testing.assert_array_equal(lambda_mat, np.array([[1.0]]))
        np.testing.assert_array_equal(lambda_support, np.array([[False]]))
        assert edge_lag_days == {}

    @pytest.mark.parametrize(
        ("override", "error_pattern"),
        [
            pytest.param(
                {
                    "lambda_block": SparseMatrixBlockSpec(
                        n_rows=4,
                        n_cols=3,
                        free_support=None,
                        template=jnp.eye(4, 3),
                        free_site_name="lambda_free",
                        det_site_name="lambda",
                        support=SupportClass.REAL,
                        site_kind=SiteKind.LOADING,
                        assembly_group="lambda",
                        fixed_spec_field="lambda_mat",
                        priors_field="lambda_free",
                    )
                },
                "lambda_support must have shape",
                id="lambda_support_none",
            ),
            pytest.param(
                {
                    "lambda_block": SparseMatrixBlockSpec(
                        n_rows=4,
                        n_cols=3,
                        free_support=np.ones((4, 4), dtype=bool),
                        template=jnp.eye(4, 3),
                        free_site_name="lambda_free",
                        det_site_name="lambda",
                        support=SupportClass.REAL,
                        site_kind=SiteKind.LOADING,
                        assembly_group="lambda",
                        fixed_spec_field="lambda_mat",
                        priors_field="lambda_free",
                    )
                },
                r"lambda_support must have shape \(4, 3\)",
                id="lambda_support_wrong_shape",
            ),
            pytest.param(
                {"diffusion_dists": [DistributionFamily.GAUSSIAN] * 2},
                "diffusion_dists length must match n_latent",
                id="diffusion_dists_short",
            ),
            pytest.param(
                {"manifest_dists": [DistributionFamily.GAUSSIAN] * 3},
                "manifest_dists length must match n_manifest",
                id="manifest_dists_short",
            ),
            pytest.param(
                {
                    "manifest_dists": [DistributionFamily.GAUSSIAN] * 4,
                    "manifest_links": [LinkFunction.IDENTITY] * 3,
                },
                "manifest_links length must match n_manifest",
                id="manifest_links_short",
            ),
            pytest.param(
                {"manifest_level_counts": [0, 0, 0]},
                "manifest_level_counts length must match n_manifest",
                id="manifest_level_counts_short",
            ),
        ],
    )
    def test_ssm_spec_rejects_invalid_structural_metadata(self, override, error_pattern):
        """SSMSpec rejects invalid mask shapes, missing masks, and mismatched lengths."""
        base = {
            "n_latent": 3,
            "n_manifest": 4,
            "dynamics_spec": dense_matrix_dynamics_spec(
                n_latent=3,
                decay_support=np.ones(3, dtype=bool),
                edge_support=np.ones((3, 3), dtype=bool),
                coupling_template=jnp.zeros((3, 3)),
                intercept_support=np.zeros(3, dtype=bool),
                cint_template=jnp.zeros(3),
            ),
            "lambda_block": SparseMatrixBlockSpec(
                n_rows=4,
                n_cols=3,
                free_support=zero_loading_support(4, 3),
                template=jnp.eye(4, 3),
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
            "latent_names": ["X", "Y", "Z"],
            "manifest_names": ["x1", "x2", "y1", "z1"],
        }
        with pytest.raises(ValueError, match=error_pattern):
            block_ssm_spec(**{**base, **override})

    def test_ssm_spec_rejects_string_lambda_mat(self):
        """Loading structure must be expressed as template + mask, not a string mode."""
        with pytest.raises(ValueError, match="lambda_mat must have shape"):
            block_ssm_spec(
                n_latent=2,
                n_manifest=2,
                dynamics_spec=dense_matrix_dynamics_spec(
                    n_latent=2,
                    decay_support=np.ones(2, dtype=bool),
                    edge_support=np.ones((2, 2), dtype=bool),
                    coupling_template=jnp.zeros((2, 2)),
                    intercept_support=np.zeros(2, dtype=bool),
                    cint_template=jnp.zeros(2),
                ),
                lambda_block=SparseMatrixBlockSpec(
                    n_rows=2,
                    n_cols=2,
                    free_support=zero_loading_support(2, 2),
                    template="free",
                    free_site_name="lambda_free",
                    det_site_name="lambda",
                    support=SupportClass.REAL,
                    site_kind=SiteKind.LOADING,
                    assembly_group="lambda",
                    fixed_spec_field="lambda_mat",
                    priors_field="lambda_free",
                ),
            )

    def test_model_build_rejects_direct_ssm_spec_plus_causal_spec(self):
        """Direct specs may not carry a causal graph unless already translated."""
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model

        causal_spec = _make_causal_spec_dict()
        X = pl.DataFrame(
            {
                "time": list(range(5)),
                "x1": [1.0] * 5,
                "x2": [2.0] * 5,
                "y1": [3.0] * 5,
                "z1": [4.0] * 5,
            }
        )

        with pytest.raises(ValueError, match="Do not pass causal_spec alongside a direct SSMSpec"):
            build_ssm_model(X, ssm_spec=_make_3latent_spec(), causal_spec=causal_spec)

    def test_model_build_rejects_autodetect_when_causal_spec_present(self):
        """Auto-detected specs may not bypass causal-structure translation."""
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model

        causal_spec = _make_causal_spec_dict()
        X = pl.DataFrame(
            {
                "time": list(range(5)),
                "x1": [1.0] * 5,
                "x2": [2.0] * 5,
                "y1": [3.0] * 5,
                "z1": [4.0] * 5,
            }
        )

        with pytest.raises(ValueError, match="requires either model_spec or ssm_spec"):
            build_ssm_model(X, causal_spec=causal_spec)

    def test_translate_spec_compiles_static_baseline_factor_from_induced_dependency(self):
        """Initial-state confounders should compile to low-rank baseline factors."""
        from nof1_causal_lab.artifacts import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "u_shared",
                        "description": "Shared static confounder",
                        "role": "exogenous",
                        "temporal_status": "time_invariant",
                    },
                    {
                        "name": "stress",
                        "description": "Stress",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                    },
                    {
                        "name": "sleep",
                        "description": "Sleep",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "is_outcome": True,
                    },
                ],
                "edges": [
                    {"cause": "u_shared", "effect": "stress"},
                    {"cause": "u_shared", "effect": "sleep"},
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "stress_score",
                        "construct_name": "stress",
                        "construct_polarity": "positive",
                        "how_to_measure": "measure stress",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "name": "sleep_score",
                        "construct_name": "sleep",
                        "construct_polarity": "positive",
                        "how_to_measure": "measure sleep",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                ],
            },
            "estimation": {
                "state_order": ["stress", "sleep"],
                "edges": [],
                "induced_dependencies": [
                    {
                        "between": ["stress", "sleep"],
                        "kind": "initial_state_correlation",
                        "source_confounders": ["u_shared"],
                    }
                ],
            },
        }
        model_spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="stress_score",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="sleep_score",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[
                ParameterSpec(
                    name="tau_u_shared",
                    role=ParameterRole.STATIC_STATE_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="baseline confounder sd",
                ),
            ],
        )

        spec, _edge_lag_days = translate_spec(model_spec, causal_spec=causal_spec)

        np.testing.assert_array_equal(spec.static_state_sd_block.free_support, np.array([True]))
        np.testing.assert_allclose(np.asarray(spec.static_state_sd_block.template), np.zeros(1))
        np.testing.assert_allclose(
            np.asarray(spec.static_factor_loadings),
            np.array([[1.0], [1.0]]),
        )
        assert spec.static_factor_names == ["tau_u_shared"]
        np.testing.assert_array_equal(
            spec.t0_chol_block.correlation_support,
            np.zeros((2, 2), dtype=bool),
        )
        np.testing.assert_array_equal(spec.t0_means_block.free_support, np.array([False, False]))
        np.testing.assert_array_equal(spec.t0_chol_block.diag_support, np.array([False, False]))

    def test_translate_spec_marks_centerable_gaussian_mean_indicators(self):
        """Gaussian identity indicators with interval means should be auto-centered."""
        from nof1_causal_lab.artifacts import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        causal_spec = _make_causal_spec_dict()
        model_spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="x1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="x2",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="y1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="z1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[],
        )

        spec, _edge_lag_days = translate_spec(model_spec, causal_spec=causal_spec)

        assert spec.manifest_centered == [True, True, True, True]

    def test_translate_spec_fixes_manifest_noise_for_single_indicator_constructs(self):
        """Single-indicator constructs get fixed zero manifest noise in the compiled spec."""
        from nof1_causal_lab.artifacts import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        causal_spec = _make_causal_spec_dict()
        model_spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="x1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="x2",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="y1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="z1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[
                ParameterSpec(
                    name="rho_X",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for X",
                ),
                ParameterSpec(
                    name="rho_Y",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for Y",
                ),
                ParameterSpec(
                    name="rho_Z",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for Z",
                ),
                ParameterSpec(
                    name="beta_X_Y",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="X causes Y",
                ),
                ParameterSpec(
                    name="beta_Y_Z",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="Y causes Z",
                ),
                ParameterSpec(
                    name="sigma_X",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="residual sd X",
                ),
                ParameterSpec(
                    name="sigma_Y",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="residual sd Y",
                ),
                ParameterSpec(
                    name="sigma_Z",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="residual sd Z",
                ),
                ParameterSpec(
                    name="lambda_x2_X",
                    role=ParameterRole.LOADING,
                    constraint=ParameterConstraint.POSITIVE,
                    description="loading",
                ),
            ],
        )

        spec, _edge_lag_days = translate_spec(model_spec, causal_spec=causal_spec)

        assert isinstance(spec.manifest_chol_block.template, jnp.ndarray)
        np.testing.assert_array_equal(
            spec.manifest_chol_block.diag_support,
            np.array([True, True, False, False]),
        )
        np.testing.assert_allclose(np.asarray(spec.manifest_chol_block.template), np.zeros((4, 4)))

    def test_translate_spec_rejects_initial_state_correlation_parameters_with_causal_spec(self):
        """Causal-spec compilation no longer accepts pairwise cor0 parameters."""
        from nof1_causal_lab.artifacts import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        causal_spec = _make_causal_spec_dict()
        model_spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="x1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="x2",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="y1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="z1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[
                ParameterSpec(
                    name="cor0_X_Z",
                    role=ParameterRole.INITIAL_STATE_CORRELATION,
                    constraint=ParameterConstraint.CORRELATION,
                    description="initial correlation",
                ),
            ],
        )

        with pytest.raises(
            ValueError,
            match="no longer accepts INITIAL_STATE_CORRELATION parameters",
        ):
            translate_spec(model_spec, causal_spec=causal_spec)

    def test_translate_spec_rejects_self_initial_state_correlation_with_causal_spec(self):
        """Even self-pairs are rejected once causal-spec compilation is active."""
        from nof1_causal_lab.artifacts import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        causal_spec = _make_causal_spec_dict()
        model_spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="x1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="x2",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="y1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    variable="z1",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[
                ParameterSpec(
                    name="cor0_X_X",
                    role=ParameterRole.INITIAL_STATE_CORRELATION,
                    constraint=ParameterConstraint.CORRELATION,
                    description="invalid self correlation",
                ),
            ],
        )

        with pytest.raises(
            ValueError,
            match="no longer accepts INITIAL_STATE_CORRELATION parameters",
        ):
            translate_spec(model_spec, causal_spec=causal_spec)

    def test_model_build_end_to_end(self):
        """Model construction with causal_spec produces masked spec."""

        from nof1_causal_lab.artifacts import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
        )
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model

        def _lik(var: str) -> LikelihoodSpec:
            return LikelihoodSpec(
                variable=var,
                distribution=DistributionFamily.GAUSSIAN,
                link=LinkFunction.IDENTITY,
                reasoning="test",
            )

        model_spec = ModelSpec(
            likelihoods=[_lik("x1"), _lik("x2"), _lik("y1"), _lik("z1")],
            parameters=[
                ParameterSpec(
                    name="rho_X",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for X",
                ),
                ParameterSpec(
                    name="rho_Y",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for Y",
                ),
                ParameterSpec(
                    name="rho_Z",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for Z",
                ),
                ParameterSpec(
                    name="beta_X_Y",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="X→Y effect",
                ),
                ParameterSpec(
                    name="beta_Y_Z",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="Y→Z effect",
                ),
                ParameterSpec(
                    name="sigma_X",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Residual SD for X",
                ),
                ParameterSpec(
                    name="sigma_Y",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Residual SD for Y",
                ),
                ParameterSpec(
                    name="sigma_Z",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Residual SD for Z",
                ),
                ParameterSpec(
                    name="lambda_x2_X",
                    role=ParameterRole.LOADING,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Loading for x2 on X",
                ),
                ParameterSpec(
                    name="obs_sd_x1",
                    role=ParameterRole.MEASUREMENT_ERROR_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Measurement-error SD for x1",
                ),
                ParameterSpec(
                    name="obs_sd_x2",
                    role=ParameterRole.MEASUREMENT_ERROR_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Measurement-error SD for x2",
                ),
            ],
        )

        causal_spec = _make_causal_spec_dict()

        # Minimal wide data
        X = pl.DataFrame(
            {
                "time": list(range(10)),
                "x1": [1.0] * 10,
                "x2": [2.0] * 10,
                "y1": [3.0] * 10,
                "z1": [4.0] * 10,
            }
        )

        model = build_ssm_model(X, model_spec=model_spec, priors={}, causal_spec=causal_spec)
        spec = model.spec

        dynamics_sites = [
            site for site in spec.iter_sample_sites() if site.assembly_group == "dynamics"
        ]
        assert sum(site.site_kind == SiteKind.DYNAMICS_DECAY for site in dynamics_sites) == 3
        assert spec.lambda_block.free_support is not None
        assert spec.n_latent == 3
        assert spec.n_manifest == 4


# ═══════════════════════════════════════════════════════════════════════
# Site-registry mask awareness
# ═══════════════════════════════════════════════════════════════════════


class TestSiteRegistryMasks:
    """Test that the canonical site registry respects SSM masks."""

    def test_site_registry_with_dynamics_support(self):
        """Site registry should size masked dynamics entries correctly."""
        from nof1_causal_lab.models.ssm.parameterization import build_site_registry

        # 3 latent, X→Y and Y→Z = 2 off-diagonal entries
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True
        offdiag_support[2, 1] = True

        spec = block_ssm_spec(
            n_latent=3,
            n_manifest=3,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=3,
                decay_support=np.ones(3, dtype=bool),
                edge_support=offdiag_support,
                coupling_template=jnp.zeros((3, 3)),
                intercept_support=full_vector_support(3),
                cint_template=jnp.zeros(3),
            ),
        )

        registry = {site.name: site for site in build_site_registry(spec)}

        weight_sites = sorted(name for name in registry if name.endswith("_weight"))
        assert weight_sites == ["vf_1_weight", "vf_2_weight"]
        assert registry["vf_0_decay"].shape == (3,)

    def test_site_registry_with_lambda_support(self):
        """Site registry should size masked loading entries correctly."""
        from nof1_causal_lab.models.ssm.parameterization import build_site_registry

        lambda_mat = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        lambda_support = np.array([[False, False], [False, False], [True, False]])

        spec = block_ssm_spec(
            n_latent=2,
            n_manifest=3,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=2,
                decay_support=np.ones(2, dtype=bool),
                edge_support=np.ones((2, 2), dtype=bool),
                coupling_template=jnp.zeros((2, 2)),
                intercept_support=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=3,
                n_cols=2,
                free_support=lambda_support,
                template=lambda_mat,
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
        )

        registry = {site.name: site for site in build_site_registry(spec)}
        assert registry["lambda_free"].shape == (1,)


# ═══════════════════════════════════════════════════════════════════════
# Integration: trace verification
# ═══════════════════════════════════════════════════════════════════════


class TestTraceVerification:
    """Verify parameter shapes via numpyro.handlers.trace."""

    def test_masked_model_trace(self):
        """Full model trace with component edge sites."""
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y
        offdiag_support[2, 1] = True  # Y→Z

        lambda_mat = jnp.zeros((4, 3))
        lambda_mat = lambda_mat.at[0, 0].set(1.0)
        lambda_mat = lambda_mat.at[2, 1].set(1.0)
        lambda_mat = lambda_mat.at[3, 2].set(1.0)

        lambda_support = np.zeros((4, 3), dtype=bool)
        lambda_support[1, 0] = True

        spec = _make_3latent_spec(
            edge_support=offdiag_support,
            lambda_block=SparseMatrixBlockSpec(
                n_rows=4,
                n_cols=3,
                free_support=lambda_support,
                template=lambda_mat,
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(123)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        assert trace["vf_0_decay"]["value"].shape == (3,)
        weight_sites = [
            name for name in trace if name.startswith("vf_") and name.endswith("_weight")
        ]
        assert len(weight_sites) == 2

        # Lambda: 1 free loading
        assert trace["lambda_free"]["value"].shape == (1,)

        # Deterministic lambda should be 4x3
        assert trace["lambda"]["value"].shape == (4, 3)
