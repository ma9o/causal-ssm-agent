"""Tests for DAG-to-SSM constraint propagation (Fixes 1-3).

Tests that:
1. drift_mask constrains off-diagonal sampling to causal edges only
2. lambda_mask + template constrains factor loadings to measurement model
3. Per-element priors align with mask positions
4. Builder constructs masks from CausalSpec
5. graph_analysis and parametric_id respect masks
6. Pipeline threading passes causal_spec through
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.handlers as handlers
import pytest

from causal_ssm_agent.models.ssm.model import (
    SSMModel,
    SSMPriors,
    SSMSpec,
    _make_prior_batch,
    _make_prior_dist,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


def _make_3latent_spec(
    drift_mask: np.ndarray | None = None,
    lambda_mask: np.ndarray | None = None,
    lambda_mat: jnp.ndarray | None = None,
) -> SSMSpec:
    """3 latent, 4 manifest spec with optional masks."""
    n_l, n_m = 3, 4
    if lambda_mat is None:
        lambda_mat = jnp.eye(n_m, n_l)
    return SSMSpec(
        n_latent=n_l,
        n_manifest=n_m,
        drift="free",
        diffusion="diag",
        lambda_mat=lambda_mat,
        drift_mask=drift_mask,
        lambda_mask=lambda_mask,
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
                    "causal_granularity": "daily",
                },
                {
                    "name": "Y",
                    "description": "Mediator",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
                {
                    "name": "Z",
                    "description": "Downstream",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
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
            "indicators": [
                {
                    "name": "x1",
                    "construct_name": "X",
                    "how_to_measure": "measure x",
                    "measurement_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "x2",
                    "construct_name": "X",
                    "how_to_measure": "measure x alt",
                    "measurement_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "y1",
                    "construct_name": "Y",
                    "how_to_measure": "measure y",
                    "measurement_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "z1",
                    "construct_name": "Z",
                    "how_to_measure": "measure z",
                    "measurement_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Fix 1: DAG-constrained drift
# ═══════════════════════════════════════════════════════════════════════


class TestDriftMask:
    """Test that drift_mask constrains off-diagonal sampling."""

    def test_drift_mask_reduces_param_count(self):
        """Masked drift should sample fewer off-diagonal params."""
        # X→Y, Y→Z: only 2 off-diagonal edges
        mask = np.eye(3, dtype=bool)
        mask[1, 0] = True  # Y row, X col (X→Y)
        mask[2, 1] = True  # Z row, Y col (Y→Z)

        spec = _make_3latent_spec(drift_mask=mask)
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # drift_offdiag_pop should have shape (2,) — only 2 edges
        assert trace["drift_offdiag_pop"]["value"].shape == (2,)

    def test_drift_mask_zeros_non_edges(self):
        """Drift entries where mask is False should be zero."""
        mask = np.eye(3, dtype=bool)
        mask[1, 0] = True  # X→Y
        mask[2, 1] = True  # Y→Z

        spec = _make_3latent_spec(drift_mask=mask)
        model = SSMModel(spec)

        rng = random.PRNGKey(42)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        drift = trace["drift"]["value"]
        # Off-diagonal zeros: positions NOT in mask
        assert float(drift[0, 1]) == 0.0  # Y→X: no edge
        assert float(drift[0, 2]) == 0.0  # Z→X: no edge
        assert float(drift[1, 2]) == 0.0  # Z→Y: no edge
        assert float(drift[2, 0]) == 0.0  # X→Z: no edge

        # Non-zero where edges exist
        assert float(drift[1, 0]) != 0.0  # X→Y edge
        assert float(drift[2, 1]) != 0.0  # Y→Z edge

    def test_no_mask_fully_free(self):
        """Without mask, all off-diagonal entries are free."""
        spec = _make_3latent_spec(drift_mask=None)
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # 3x3 - 3 diagonal = 6 off-diagonal
        assert trace["drift_offdiag_pop"]["value"].shape == (6,)

    def test_drift_mask_single_latent(self):
        """Single latent: no off-diagonal, mask should be identity."""
        mask = np.eye(1, dtype=bool)
        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            drift="free",
            drift_mask=mask,
            lambda_mat=jnp.eye(1),
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((5, 1)),
            times=jnp.arange(5, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # No off-diagonal params sampled
        assert "drift_offdiag_pop" not in trace


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Structured lambda
# ═══════════════════════════════════════════════════════════════════════


class TestLambdaMask:
    """Test that lambda_mask constrains factor loadings."""

    def test_lambda_template_plus_mask(self):
        """Template+mask mode: fixed reference + free additional loadings."""
        # X has 2 indicators (x1 ref, x2 free), Y has 1, Z has 1
        lambda_mat = jnp.zeros((4, 3))
        lambda_mat = lambda_mat.at[0, 0].set(1.0)  # x1→X (ref)
        lambda_mat = lambda_mat.at[2, 1].set(1.0)  # y1→Y (ref)
        lambda_mat = lambda_mat.at[3, 2].set(1.0)  # z1→Z (ref)

        lambda_mask = np.zeros((4, 3), dtype=bool)
        lambda_mask[1, 0] = True  # x2→X (free)

        spec = _make_3latent_spec(lambda_mat=lambda_mat, lambda_mask=lambda_mask)
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
        """Array lambda_mat without mask is returned as-is."""
        lambda_mat = jnp.eye(4, 3)
        spec = _make_3latent_spec(lambda_mat=lambda_mat, lambda_mask=None)
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # No lambda_free sampled
        assert "lambda_free" not in trace
        # Lambda should not appear as deterministic since it's fully fixed
        assert "lambda" not in trace


# ═══════════════════════════════════════════════════════════════════════
# Fix 3: Per-element priors
# ═══════════════════════════════════════════════════════════════════════


class TestPerElementPriors:
    """Test array-valued priors via _make_prior_dist and _make_prior_batch."""

    def test_make_prior_dist_scalar(self):
        """Scalar mu/sigma produces scalar Normal."""
        d = _make_prior_dist({"mu": 0.0, "sigma": 1.0})
        assert d.batch_shape == ()

    def test_make_prior_dist_array(self):
        """Array mu/sigma produces batched Normal."""
        d = _make_prior_dist({"mu": [0.1, 0.2, 0.3], "sigma": [1.0, 0.5, 0.3]})
        assert d.batch_shape == (3,)

    def test_make_prior_batch_scalar_expand(self):
        """Scalar prior expanded to batch shape."""
        d = _make_prior_batch({"mu": 0.0, "sigma": 1.0}, 5)
        assert d.batch_shape == (5,)

    def test_make_prior_batch_array_passthrough(self):
        """Array prior with correct shape passes through."""
        d = _make_prior_batch({"mu": [0.1, 0.2], "sigma": [1.0, 0.5]}, 2)
        assert d.batch_shape == (2,)

    def test_make_prior_batch_mismatch_raises(self):
        """Array prior with wrong shape raises."""
        with pytest.raises(ValueError, match="does not match"):
            _make_prior_batch({"mu": [0.1, 0.2], "sigma": [1.0, 0.5]}, 3)

    def test_per_element_prior_in_model(self):
        """Per-element drift priors are used in sampling."""
        mask = np.eye(2, dtype=bool)
        mask[1, 0] = True  # X→Y

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift="free",
            drift_mask=mask,
            lambda_mat=jnp.eye(2),
            latent_names=["X", "Y"],
            manifest_names=["x1", "y1"],
        )

        # Per-element prior: single off-diagonal has mu=2.0
        priors = SSMPriors(
            drift_offdiag={"mu": [2.0], "sigma": [0.1]},
        )
        model = SSMModel(spec, priors)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((5, 2)),
            times=jnp.arange(5, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # The off-diagonal value should be near 2.0 (tight prior)
        offdiag = float(trace["drift_offdiag_pop"]["value"][0])
        assert abs(offdiag - 2.0) < 1.0, f"Expected ~2.0, got {offdiag}"


# ═══════════════════════════════════════════════════════════════════════
# Builder mask construction
# ═══════════════════════════════════════════════════════════════════════


class TestBuilderMasks:
    """Test that SSMModelBuilder constructs correct masks from CausalSpec."""

    def test_build_masks_from_causal_spec(self):
        """Builder constructs drift_mask and lambda_mask from CausalSpec."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        causal_spec = _make_causal_spec_dict()
        builder = SSMModelBuilder(causal_spec=causal_spec)

        latent_names = ["X", "Y", "Z"]
        manifest_cols = ["x1", "x2", "y1", "z1"]

        drift_mask, lambda_mat, lambda_mask = builder._build_masks_from_causal_spec(
            latent_names, manifest_cols, 3, 4
        )

        # Drift mask: diagonal + X→Y + Y→Z
        assert drift_mask is not None
        assert drift_mask[0, 0]  # X self
        assert drift_mask[1, 1]  # Y self
        assert drift_mask[2, 2]  # Z self
        assert drift_mask[1, 0]  # X→Y (effect=Y row, cause=X col)
        assert drift_mask[2, 1]  # Y→Z (effect=Z row, cause=Y col)
        assert not drift_mask[0, 1]  # No Y→X edge
        assert not drift_mask[0, 2]  # No Z→X edge
        assert not drift_mask[1, 2]  # No Z→Y edge
        assert not drift_mask[2, 0]  # No X→Z edge

        # Lambda: x1 fixed ref for X, x2 free for X, y1 fixed ref for Y, z1 fixed ref for Z
        assert float(lambda_mat[0, 0]) == 1.0  # x1→X
        assert float(lambda_mat[2, 1]) == 1.0  # y1→Y
        assert float(lambda_mat[3, 2]) == 1.0  # z1→Z

        assert lambda_mask is not None
        assert lambda_mask[1, 0]  # x2→X is free
        assert not lambda_mask[0, 0]  # x1→X is fixed
        assert not lambda_mask[2, 1]  # y1→Y is fixed

    def test_no_causal_spec_no_masks(self):
        """Without causal_spec, masks are None."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder()
        drift_mask, _lambda_mat, lambda_mask = builder._build_masks_from_causal_spec(
            ["X"], ["x1"], 1, 1
        )
        assert drift_mask is None
        assert lambda_mask is None

    def test_builder_end_to_end(self):
        """Full builder pipeline with causal_spec produces masked spec."""
        import polars as pl

        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
        from causal_ssm_agent.orchestrator.schemas_model import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
        )

        def _lik(var: str) -> LikelihoodSpec:
            return LikelihoodSpec(
                variable=var,
                distribution=DistributionFamily.NORMAL,
                link=LinkFunction.IDENTITY,
                reasoning="test",
            )

        model_spec = ModelSpec(
            likelihoods=[_lik("x1"), _lik("x2"), _lik("y1"), _lik("z1")],
            parameters=[
                ParameterSpec(
                    name="rho_X",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.NONE,
                    description="AR for X",
                    search_context="autoregressive coefficient",
                ),
                ParameterSpec(
                    name="rho_Y",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.NONE,
                    description="AR for Y",
                    search_context="autoregressive coefficient",
                ),
                ParameterSpec(
                    name="rho_Z",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.NONE,
                    description="AR for Z",
                    search_context="autoregressive coefficient",
                ),
                ParameterSpec(
                    name="beta_X_Y",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="X→Y effect",
                    search_context="causal effect",
                ),
                ParameterSpec(
                    name="beta_Y_Z",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="Y→Z effect",
                    search_context="causal effect",
                ),
            ],
            random_effects=[],
            model_clock="daily",
            reasoning="Test model",
        )

        causal_spec = _make_causal_spec_dict()

        builder = SSMModelBuilder(
            model_spec=model_spec,
            priors={},
            causal_spec=causal_spec,
        )

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

        builder.build_model(X)
        spec = builder._spec

        # Verify masks were built
        assert spec.drift_mask is not None
        assert spec.lambda_mask is not None
        assert spec.n_latent == 3
        assert spec.n_manifest == 4


# ═══════════════════════════════════════════════════════════════════════
# Graph analysis mask awareness
# ═══════════════════════════════════════════════════════════════════════


class TestGraphAnalysisMasks:
    """Test that graph_analysis functions respect masks."""

    def test_drift_sparsity_uses_mask(self):
        """compute_drift_sparsity returns drift_mask when set."""
        from causal_ssm_agent.models.likelihoods.graph_analysis import compute_drift_sparsity

        mask = np.array([[True, False], [True, True]])
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift="free",
            drift_mask=mask,
            lambda_mat=jnp.eye(2),
        )

        result = compute_drift_sparsity(spec)
        np.testing.assert_array_equal(result, mask)

    def test_drift_sparsity_no_mask_all_true(self):
        """Without mask, drift_sparsity is all True for free drift."""
        from causal_ssm_agent.models.likelihoods.graph_analysis import compute_drift_sparsity

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift="free",
            lambda_mat=jnp.eye(2),
        )

        result = compute_drift_sparsity(spec)
        assert np.all(result)

    def test_obs_dependency_with_lambda_mask(self):
        """compute_obs_dependency combines fixed nonzeros with lambda_mask."""
        from causal_ssm_agent.models.likelihoods.graph_analysis import compute_obs_dependency

        lambda_mat = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        lambda_mask = np.array([[False, False], [False, False], [True, False]])

        spec = SSMSpec(
            n_latent=2,
            n_manifest=3,
            lambda_mat=lambda_mat,
            lambda_mask=lambda_mask,
        )

        result = compute_obs_dependency(spec)
        # Row 0: [True, False] (from lambda_mat)
        # Row 1: [False, True] (from lambda_mat)
        # Row 2: [True, False] (from lambda_mask)
        expected = np.array([[True, False], [False, True], [True, False]])
        np.testing.assert_array_equal(result, expected)


# ═══════════════════════════════════════════════════════════════════════
# Parametric ID mask awareness
# ═══════════════════════════════════════════════════════════════════════


class TestParametricIdMasks:
    """Test that count_free_params uses masks."""

    def test_count_free_params_with_drift_mask(self):
        """count_free_params should count masked off-diagonal entries."""
        from causal_ssm_agent.utils.parametric_id import count_free_params

        # 3 latent, X→Y and Y→Z = 2 off-diagonal entries
        mask = np.eye(3, dtype=bool)
        mask[1, 0] = True
        mask[2, 1] = True

        spec = SSMSpec(
            n_latent=3,
            n_manifest=3,
            drift="free",
            drift_mask=mask,
            lambda_mat=jnp.eye(3),
            diffusion="diag",
            cint="free",
            t0_means="free",
            t0_var="diag",
            manifest_var="diag",
        )

        counts = count_free_params(spec)

        # Should have 2 off-diagonal, not 6
        assert counts["drift_offdiag_pop"] == 2
        assert counts["drift_diag_pop"] == 3

    def test_count_free_params_with_lambda_mask(self):
        """count_free_params counts masked lambda entries."""
        from causal_ssm_agent.utils.parametric_id import count_free_params

        lambda_mat = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        lambda_mask = np.array([[False, False], [False, False], [True, False]])

        spec = SSMSpec(
            n_latent=2,
            n_manifest=3,
            lambda_mat=lambda_mat,
            lambda_mask=lambda_mask,
            drift="free",
            diffusion="diag",
        )

        counts = count_free_params(spec)
        assert counts.get("lambda_free", 0) == 1

    def test_count_free_params_no_mask(self):
        """Without mask, count_free_params gives full off-diagonal count."""
        from causal_ssm_agent.utils.parametric_id import count_free_params

        spec = SSMSpec(
            n_latent=3,
            n_manifest=3,
            drift="free",
            lambda_mat=jnp.eye(3),
            diffusion="diag",
        )

        counts = count_free_params(spec)
        assert counts.get("drift_offdiag_pop", 0) == 6  # 3*3 - 3


# ═══════════════════════════════════════════════════════════════════════
# Integration: trace verification
# ═══════════════════════════════════════════════════════════════════════


class TestTraceVerification:
    """Verify parameter shapes via numpyro.handlers.trace."""

    def test_masked_model_trace(self):
        """Full model trace with masks: verify drift_offdiag_pop shape."""
        mask = np.eye(3, dtype=bool)
        mask[1, 0] = True  # X→Y
        mask[2, 1] = True  # Y→Z

        lambda_mat = jnp.zeros((4, 3))
        lambda_mat = lambda_mat.at[0, 0].set(1.0)
        lambda_mat = lambda_mat.at[2, 1].set(1.0)
        lambda_mat = lambda_mat.at[3, 2].set(1.0)

        lambda_mask = np.zeros((4, 3), dtype=bool)
        lambda_mask[1, 0] = True

        spec = _make_3latent_spec(
            drift_mask=mask,
            lambda_mat=lambda_mat,
            lambda_mask=lambda_mask,
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(123)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=model.make_likelihood_backend(),
        )

        # Drift: 3 diagonal + 2 off-diagonal
        assert trace["drift_diag_pop"]["value"].shape == (3,)
        assert trace["drift_offdiag_pop"]["value"].shape == (2,)

        # Lambda: 1 free loading
        assert trace["lambda_free"]["value"].shape == (1,)

        # Deterministic drift should be 3x3
        assert trace["drift"]["value"].shape == (3, 3)

        # Deterministic lambda should be 4x3
        assert trace["lambda"]["value"].shape == (4, 3)

    def test_hierarchical_masked_drift(self):
        """Hierarchical model with mask: vmap produces correct shapes."""
        mask = np.eye(2, dtype=bool)
        mask[1, 0] = True  # X→Y

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift="free",
            drift_mask=mask,
            lambda_mat=jnp.eye(2),
            hierarchical=True,
            n_subjects=3,
            indvarying=["t0_means"],
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        subject_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((9, 2)),
            times=jnp.arange(9, dtype=jnp.float32),
            subject_ids=subject_ids,
            likelihood_backend=model.make_likelihood_backend(),
        )

        # 1 off-diagonal edge
        assert trace["drift_offdiag_pop"]["value"].shape == (1,)

        # Drift is broadcast to (n_subjects, n_latent, n_latent) in hierarchical mode
        drift = trace["drift"]["value"]
        assert drift.shape == (3, 2, 2)

        # Each subject's drift should have the edge at [1, 0] and zero at [0, 1]
        assert float(drift[0, 0, 1]) == 0.0
        assert float(drift[1, 0, 1]) == 0.0
        assert float(drift[2, 0, 1]) == 0.0
