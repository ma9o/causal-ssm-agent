"""Guard tests for the per-construct identification-anchor invariant.

Every retained construct must have exactly one location anchor and one scale
anchor (docs/reference/statistical-model-spec/identification.md). These tests
enumerate the family/policy combinations so that any future eligibility change
that reopens an exact likelihood ridge fails loudly at compile time.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts import DistributionFamily, LikelihoodSpec, LinkFunction
from nof1_causal_lab.artifacts.statistical_model_spec import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
    StatisticalModelSpec,
)
from nof1_causal_lab.flows.transitions.model_spec.agentic.parameter_surfaces import (
    parameter_is_active_for_statistical_model_spec,
)
from nof1_causal_lab.models.ssm.compile.spec_translation import (
    SpecTranslationError,
    translate_spec,
)
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.likelihood_extra_params import assemble_sampled_extra_params
from tests.ssm_spec_fixtures import block_ssm_spec

# ═══════════════════════════════════════════════════════════════════════
# Fixture builders
# ═══════════════════════════════════════════════════════════════════════


def _indicator(
    name: str,
    construct_name: str,
    dtype: str,
    *,
    polarity: str = "positive",
) -> dict:
    indicator = {
        "name": name,
        "construct_name": construct_name,
        "construct_polarity": polarity,
        "how_to_measure": f"measure {name}",
        "measurement_dtype": dtype,
        "aggregation": "mean" if dtype == "continuous" else "last",
    }
    if dtype == "ordinal":
        indicator["ordinal_levels"] = ["low", "medium", "high"]
    if dtype == "categorical":
        indicator["categorical_levels"] = ["a", "b", "c"]
    return indicator


def _causal_design(
    construct_names: list[str],
    indicators: list[dict],
    *,
    time_invariant: set[str] | None = None,
) -> dict:
    time_invariant = time_invariant or set()
    return {
        "latent": {
            "constructs": [
                {
                    "name": name,
                    "description": name,
                    "temporal_status": (
                        "time_invariant" if name in time_invariant else "time_varying"
                    ),
                }
                for name in construct_names
            ],
            "edges": [],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": indicators,
        },
        "estimation": {
            "state_order": list(construct_names),
            "edges": [],
            "induced_dependencies": [],
        },
    }


_LIKELIHOOD_BY_DTYPE = {
    "continuous": (DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY),
    "binary": (DistributionFamily.BERNOULLI, LinkFunction.LOGIT),
    "ordinal": (DistributionFamily.ORDERED_LOGISTIC, LinkFunction.CUMULATIVE_LOGIT),
    "categorical": (DistributionFamily.CATEGORICAL, LinkFunction.SOFTMAX),
}


def _likelihood(variable: str, dtype: str) -> LikelihoodSpec:
    distribution, link = _LIKELIHOOD_BY_DTYPE[dtype]
    return LikelihoodSpec(
        variable=variable,
        distribution=distribution,
        link=link,
        reasoning="test",
    )


def _model_spec(
    likelihoods: list[LikelihoodSpec],
    parameters: list[ParameterSpec] | None = None,
    *,
    equilibrium_forcing: bool = False,
) -> StatisticalModelSpec:
    return StatisticalModelSpec(
        likelihoods=likelihoods,
        parameters=parameters or [],
        equilibrium_forcing=equilibrium_forcing,
    )


# ═══════════════════════════════════════════════════════════════════════
# Ordered-logistic: free threshold base, no centering
# ═══════════════════════════════════════════════════════════════════════


class TestOrderedThresholds:
    def test_cutpoints_keep_free_base(self):
        """The threshold base shifts the cutpoints instead of cancelling out."""
        spec = block_ssm_spec(
            n_latent=1,
            n_manifest=1,
            dynamics_spec=DynamicsSpec(n_latent=1, components=()),
            manifest_dists=[DistributionFamily.ORDERED_LOGISTIC],
            manifest_level_counts=[3],
        )
        extra = assemble_sampled_extra_params(
            spec,
            {
                "obs_ordered_base": jnp.array([0.7]),
                "obs_ordered_gaps": jnp.array([[0.5]]),
            },
        )
        np.testing.assert_allclose(
            np.asarray(extra["obs_ordered_cutpoints"]), np.array([[0.7, 1.2]])
        )

    def test_ordinal_only_construct_compiles(self):
        """Well-at-zero anchors location; the fixed logistic link anchors scale."""
        causal_design = _causal_design(["mood"], [_indicator("mood_level", "mood", "ordinal")])
        spec, _ = translate_spec(
            _model_spec([_likelihood("mood_level", "ordinal")]),
            causal_design=causal_design,
        )
        assert not any(spec.manifest_cat_anchor)
        assert float(spec.lambda_block.template[0, 0]) == 1.0
        assert not spec.lambda_block.free_support[0, 0]


# ═══════════════════════════════════════════════════════════════════════
# Location anchors: equilibrium center and static t0 mean
# ═══════════════════════════════════════════════════════════════════════


class TestLocationAnchors:
    def test_free_center_without_standardized_channel_fails(self):
        causal_design = _causal_design(["mood"], [_indicator("mood_flag", "mood", "binary")])
        spec = _model_spec(
            [_likelihood("mood_flag", "binary")],
            [
                ParameterSpec(
                    name="cint_mood",
                    role=ParameterRole.STATE_INTERCEPT,
                    constraint=ParameterConstraint.NONE,
                    description="equilibrium center",
                )
            ],
            equilibrium_forcing=True,
        )
        with pytest.raises(SpecTranslationError, match="no location anchor"):
            translate_spec(spec, causal_design=causal_design)

    def test_free_center_with_standardized_channel_compiles(self):
        causal_design = _causal_design(
            ["mood"],
            [
                _indicator("mood_rating", "mood", "continuous"),
                _indicator("mood_flag", "mood", "binary"),
            ],
        )
        spec, _ = translate_spec(
            _model_spec(
                [
                    _likelihood("mood_rating", "continuous"),
                    _likelihood("mood_flag", "binary"),
                ],
                [
                    ParameterSpec(
                        name="cint_mood",
                        role=ParameterRole.STATE_INTERCEPT,
                        constraint=ParameterConstraint.NONE,
                        description="equilibrium center",
                    )
                ],
                equilibrium_forcing=True,
            ),
            causal_design=causal_design,
        )
        assert spec.manifest_standardized[0]

    def test_static_t0_mean_gated_without_standardized_channel(self):
        causal_design = _causal_design(
            ["mood", "trait"],
            [
                _indicator("mood_rating", "mood", "continuous"),
                _indicator("trait_flag", "trait", "binary"),
            ],
            time_invariant={"trait"},
        )
        spec, _ = translate_spec(
            _model_spec(
                [
                    _likelihood("mood_rating", "continuous"),
                    _likelihood("trait_flag", "binary"),
                ]
            ),
            causal_design=causal_design,
        )
        trait_index = spec.latent_names.index("trait")
        assert not spec.t0_means_block.free_support[trait_index]

    def test_static_t0_mean_free_with_standardized_channel(self):
        causal_design = _causal_design(
            ["mood", "trait"],
            [
                _indicator("mood_rating", "mood", "continuous"),
                _indicator("trait_score", "trait", "continuous"),
            ],
            time_invariant={"trait"},
        )
        spec, _ = translate_spec(
            _model_spec(
                [
                    _likelihood("mood_rating", "continuous"),
                    _likelihood("trait_score", "continuous"),
                ]
            ),
            causal_design=causal_design,
        )
        trait_index = spec.latent_names.index("trait")
        assert spec.t0_means_block.free_support[trait_index]

    def test_construct_without_indicators_fails(self):
        causal_design = _causal_design(
            ["mood", "ghost"], [_indicator("mood_rating", "mood", "continuous")]
        )
        with pytest.raises(SpecTranslationError, match="retains no indicators"):
            translate_spec(
                _model_spec([_likelihood("mood_rating", "continuous")]),
                causal_design=causal_design,
            )


# ═══════════════════════════════════════════════════════════════════════
# Categorical: pinned loadings and anchor slopes
# ═══════════════════════════════════════════════════════════════════════


class TestCategoricalAnchors:
    def test_categorical_loading_pinned_in_mixed_construct(self):
        causal_design = _causal_design(
            ["mood"],
            [
                _indicator("mood_rating", "mood", "continuous"),
                _indicator("mood_kind", "mood", "categorical"),
            ],
        )
        spec, _ = translate_spec(
            _model_spec(
                [
                    _likelihood("mood_rating", "continuous"),
                    _likelihood("mood_kind", "categorical"),
                ]
            ),
            causal_design=causal_design,
        )
        cat_row = spec.manifest_names.index("mood_kind")
        assert float(spec.lambda_block.template[cat_row, 0]) == 1.0
        assert not spec.lambda_block.free_support[cat_row, 0]
        assert not spec.manifest_cat_anchor[cat_row]

    def test_all_categorical_construct_gets_anchor_slope(self):
        causal_design = _causal_design(["mood"], [_indicator("mood_kind", "mood", "categorical")])
        spec, _ = translate_spec(
            _model_spec([_likelihood("mood_kind", "categorical")]),
            causal_design=causal_design,
        )
        assert spec.manifest_cat_anchor == [True]

        extra = assemble_sampled_extra_params(
            block_ssm_spec(
                n_latent=1,
                n_manifest=1,
                dynamics_spec=DynamicsSpec(n_latent=1, components=()),
                manifest_dists=[DistributionFamily.CATEGORICAL],
                manifest_level_counts=[3],
                manifest_cat_anchor=[True],
            ),
            {
                "obs_cat_intercepts": jnp.array([[0.3, -0.4]]),
                "obs_cat_slopes": jnp.array([[9.9, 2.0]]),
            },
        )
        np.testing.assert_allclose(np.asarray(extra["obs_cat_slopes"]), np.array([[1.0, 2.0]]))


# ═══════════════════════════════════════════════════════════════════════
# Reference indicator preference and prior-surface activation
# ═══════════════════════════════════════════════════════════════════════


class TestAnchorSurfaces:
    def test_reference_prefers_continuous_over_ordinal(self):
        causal_design = _causal_design(
            ["mood"],
            [
                _indicator("mood_level", "mood", "ordinal"),
                _indicator("mood_rating", "mood", "continuous"),
            ],
        )
        spec, _ = translate_spec(
            _model_spec(
                [
                    _likelihood("mood_level", "ordinal"),
                    _likelihood("mood_rating", "continuous"),
                ]
            ),
            causal_design=causal_design,
        )
        continuous_row = spec.manifest_names.index("mood_rating")
        ordinal_row = spec.manifest_names.index("mood_level")
        assert float(spec.lambda_block.template[continuous_row, 0]) == 1.0
        assert spec.lambda_block.free_support[ordinal_row, 0]

    def test_loading_surface_inactive_for_categorical_choice(self):
        parameter = {"name": "lambda_mood_kind_mood", "role": "loading", "indicator": "mood_kind"}
        chosen = {
            "mood_kind": {
                "variable": "mood_kind",
                "distribution": "categorical",
                "link": "softmax",
                "construct_name": "mood",
            }
        }
        assert not parameter_is_active_for_statistical_model_spec(
            parameter,
            chosen,
            initialization_policy="stationary",
            observation_intercept_policy="free",
            equilibrium_forcing=False,
        )
        chosen["mood_kind"]["distribution"] = "ordered_logistic"
        chosen["mood_kind"]["link"] = "cumulative_logit"
        assert parameter_is_active_for_statistical_model_spec(
            parameter,
            chosen,
            initialization_policy="stationary",
            observation_intercept_policy="free",
            equilibrium_forcing=False,
        )

    def test_static_t0_mean_surface_requires_standardized_channel(self):
        parameter = {
            "name": "t0_mean_trait",
            "role": "initial_state_mean",
            "construct": "trait",
            "temporal_status": "time_invariant",
        }
        unanchored = {
            "trait_flag": {
                "variable": "trait_flag",
                "distribution": "bernoulli",
                "link": "logit",
                "construct_name": "trait",
                "standardized": False,
            }
        }
        assert not parameter_is_active_for_statistical_model_spec(
            parameter,
            unanchored,
            initialization_policy="free",
            observation_intercept_policy="free",
            equilibrium_forcing=False,
        )
        anchored = {
            "trait_score": {
                "variable": "trait_score",
                "distribution": "gaussian",
                "link": "identity",
                "construct_name": "trait",
                "standardized": True,
            }
        }
        assert parameter_is_active_for_statistical_model_spec(
            parameter,
            anchored,
            initialization_policy="free",
            observation_intercept_policy="free",
            equilibrium_forcing=False,
        )
