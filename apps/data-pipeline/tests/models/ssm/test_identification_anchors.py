"""Guard tests for the per-construct identification-anchor invariant.

Every retained construct must have exactly one location anchor and one scale
anchor (docs/reference/statistical-model-spec/identification.md). These tests
enumerate the family/policy combinations so that any future eligibility change
that reopens an exact likelihood ridge fails loudly at compile time.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts import DistributionFamily, LikelihoodSpec, LinkFunction
from nof1_causal_lab.artifacts.statistical_model_spec import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
    StatisticalModelSpec,
)
from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
from nof1_causal_lab.flows.transitions.model_spec.agentic.parameter_surfaces import (
    parameter_is_active_for_statistical_model_spec,
)
from nof1_causal_lab.models.ssm.compile.spec_translation import (
    SpecTranslationError,
    translate_spec,
)
from nof1_causal_lab.models.ssm.compile.structural import StructuralClosureError
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.likelihood_extra_params import assemble_sampled_extra_params
from nof1_causal_lab.utils.causal_design import build_reference_indicator_lookup
from tests.helpers import make_structural_plan
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
) -> dict[str, Any]:
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


def _structural_plan(
    construct_names: list[str],
    indicators: list[dict[str, Any]],
    *,
    time_invariant: set[str] | None = None,
) -> StructuralPlan:
    time_invariant = time_invariant or set()
    plan = make_structural_plan(construct_names, [])
    for construct in plan["semantics"]["constructs"].values():
        construct["temporal_status"] = (
            "time_invariant" if construct["name"] in time_invariant else "time_varying"
        )

    indicator_items = {
        f"indicator:{index:04d}": indicator for index, indicator in enumerate(indicators)
    }
    plan["semantics"]["indicators"] = indicator_items
    plan["manifest_indicator_order"] = list(indicator_items)
    reference_names = build_reference_indicator_lookup(indicators)
    construct_ids = {
        construct["name"]: source_id
        for source_id, construct in plan["semantics"]["constructs"].items()
    }
    indicator_ids = {
        indicator["name"]: source_id for source_id, indicator in indicator_items.items()
    }
    plan["reference_indicator_ids"] = {
        construct_ids[construct_name]: indicator_ids[indicator_name]
        for construct_name, indicator_name in reference_names.items()
    }
    plan["dispositions"] = [
        disposition
        for disposition in plan["dispositions"]
        if disposition["source_kind"] != "indicator"
    ]
    plan["dispositions"].extend(
        {
            "source_id": source_id,
            "source_kind": "indicator",
            "disposition": "manifest",
            "reason": "test manifest",
        }
        for source_id in indicator_items
    )
    return StructuralPlan.model_validate(plan)


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


def _manifest_mean(variable: str) -> ParameterSpec:
    return ParameterSpec(
        name=f"manifest_mean_{variable}",
        role=ParameterRole.OBSERVATION_INTERCEPT,
        constraint=ParameterConstraint.NONE,
        description=f"Observation intercept for {variable}",
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
        structural_plan = _structural_plan(["mood"], [_indicator("mood_level", "mood", "ordinal")])
        spec, _ = translate_spec(
            _model_spec([_likelihood("mood_level", "ordinal")]),
            structural_plan=structural_plan,
        )
        assert spec.manifest_cat_anchor is not None
        assert not any(spec.manifest_cat_anchor)
        assert float(spec.lambda_block.template[0, 0]) == 1.0
        assert not spec.lambda_block.free_support[0, 0]

    def test_manifest_intercept_is_rejected_for_threshold_channel(self):
        structural_plan = _structural_plan(["mood"], [_indicator("mood_level", "mood", "ordinal")])
        with pytest.raises(SpecTranslationError, match=r"Observation intercept.*is inactive"):
            translate_spec(
                _model_spec(
                    [_likelihood("mood_level", "ordinal")],
                    [_manifest_mean("mood_level")],
                ),
                structural_plan=structural_plan,
            )


# ═══════════════════════════════════════════════════════════════════════
# Location anchors: equilibrium center and static t0 mean
# ═══════════════════════════════════════════════════════════════════════


class TestLocationAnchors:
    def test_manifest_intercept_is_rejected_for_standardized_channel(self):
        structural_plan = _structural_plan(
            ["mood"], [_indicator("mood_rating", "mood", "continuous")]
        )
        with pytest.raises(SpecTranslationError, match=r"Observation intercept.*is inactive"):
            translate_spec(
                _model_spec(
                    [_likelihood("mood_rating", "continuous")],
                    [_manifest_mean("mood_rating")],
                ),
                structural_plan=structural_plan,
            )

    def test_manifest_intercept_remains_free_for_raw_gaussian_sum_channel(self):
        indicator = _indicator("fill_quantity", "dose", "continuous")
        indicator["aggregation"] = "sum"
        structural_plan = _structural_plan(["dose"], [indicator])
        spec, _ = translate_spec(
            _model_spec(
                [_likelihood("fill_quantity", "continuous")],
                [_manifest_mean("fill_quantity")],
            ),
            structural_plan=structural_plan,
        )

        assert spec.manifest_standardized == [False]
        assert spec.manifest_means_block.free_support.tolist() == [True]

    def test_manifest_intercept_remains_free_for_binary_channel(self):
        structural_plan = _structural_plan(["mood"], [_indicator("mood_flag", "mood", "binary")])
        spec, _ = translate_spec(
            _model_spec(
                [_likelihood("mood_flag", "binary")],
                [_manifest_mean("mood_flag")],
            ),
            structural_plan=structural_plan,
        )
        assert spec.manifest_means_block.free_support.tolist() == [True]

    def test_free_center_without_standardized_channel_fails(self):
        structural_plan = _structural_plan(["mood"], [_indicator("mood_flag", "mood", "binary")])
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
        with pytest.raises(StructuralClosureError, match="no location anchor"):
            translate_spec(spec, structural_plan=structural_plan)

    def test_free_center_with_standardized_channel_compiles(self):
        structural_plan = _structural_plan(
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
            structural_plan=structural_plan,
        )
        assert spec.manifest_standardized is not None
        assert spec.manifest_standardized[0]

    def test_static_t0_mean_gated_without_standardized_channel(self):
        structural_plan = _structural_plan(
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
            structural_plan=structural_plan,
        )
        assert spec.latent_names is not None
        trait_index = spec.latent_names.index("trait")
        assert not spec.t0_means_block.free_support[trait_index]

    def test_static_t0_mean_free_with_standardized_channel(self):
        structural_plan = _structural_plan(
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
            structural_plan=structural_plan,
        )
        assert spec.latent_names is not None
        trait_index = spec.latent_names.index("trait")
        assert spec.t0_means_block.free_support[trait_index]

    def test_construct_without_indicators_fails(self):
        with pytest.raises(
            ValidationError,
            match="retained states lack manifest indicators",
        ):
            _structural_plan(
                ["mood", "ghost"],
                [_indicator("mood_rating", "mood", "continuous")],
            )


# ═══════════════════════════════════════════════════════════════════════
# Categorical: pinned loadings and anchor slopes
# ═══════════════════════════════════════════════════════════════════════


class TestCategoricalAnchors:
    def test_categorical_loading_pinned_in_mixed_construct(self):
        structural_plan = _structural_plan(
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
            structural_plan=structural_plan,
        )
        assert spec.manifest_names is not None
        assert spec.manifest_cat_anchor is not None
        cat_row = spec.manifest_names.index("mood_kind")
        assert float(spec.lambda_block.template[cat_row, 0]) == 1.0
        assert not spec.lambda_block.free_support[cat_row, 0]
        assert not spec.manifest_cat_anchor[cat_row]

    def test_all_categorical_construct_gets_anchor_slope(self):
        structural_plan = _structural_plan(
            ["mood"], [_indicator("mood_kind", "mood", "categorical")]
        )
        spec, _ = translate_spec(
            _model_spec([_likelihood("mood_kind", "categorical")]),
            structural_plan=structural_plan,
        )
        assert spec.manifest_cat_anchor == [True]
        assert spec.manifest_level_counts == [3]

        extra = assemble_sampled_extra_params(
            spec,
            {
                "obs_cat_intercepts": jnp.array([[0.3, -0.4]]),
                "obs_cat_slopes": jnp.array([[9.9, 2.0]]),
            },
        )
        np.testing.assert_allclose(np.asarray(extra["obs_cat_slopes"]), np.array([[1.0, 2.0]]))

    def test_manifest_intercept_is_rejected_for_categorical_channel(self):
        structural_plan = _structural_plan(
            ["mood"], [_indicator("mood_kind", "mood", "categorical")]
        )
        with pytest.raises(SpecTranslationError, match=r"Observation intercept.*is inactive"):
            translate_spec(
                _model_spec(
                    [_likelihood("mood_kind", "categorical")],
                    [_manifest_mean("mood_kind")],
                ),
                structural_plan=structural_plan,
            )


# ═══════════════════════════════════════════════════════════════════════
# Reference indicator preference and prior-surface activation
# ═══════════════════════════════════════════════════════════════════════


class TestAnchorSurfaces:
    def test_reference_prefers_continuous_over_ordinal(self):
        structural_plan = _structural_plan(
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
            structural_plan=structural_plan,
        )
        assert spec.manifest_names is not None
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

    def test_raw_gaussian_sum_surface_activates_manifest_intercept(self):
        parameter = {
            "name": "manifest_mean_fill_quantity",
            "role": "observation_intercept",
            "indicator": "fill_quantity",
        }
        chosen = {
            "fill_quantity": {
                "variable": "fill_quantity",
                "distribution": "gaussian",
                "link": "identity",
                "construct_name": "dose",
                "support_kind": "interval",
                "summary_operator": "sum",
                "standardized": False,
            }
        }

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
