"""Stage 5b subset-recompile utilities."""

from __future__ import annotations

import polars as pl
from subset_refit import (
    build_stage5b_subset_runtime,
    subset_causal_spec,
    subset_model_spec,
)

from causal_ssm_agent.artifacts.model_spec import ParameterSpec
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_parameter_surfaces import (
    parameter_is_active_for_model_spec,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_skeleton import (
    derive_deterministic_spec,
)
from causal_ssm_agent.workers.prior_research import get_default_prior


def _causal_spec() -> dict:
    constructs = [
        {
            "name": "X",
            "description": "Treatment-like construct",
            "role": "exogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "Y",
            "description": "Outcome construct",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "is_outcome": True,
        },
        {
            "name": "G",
            "description": "Static genetic predisposition construct",
            "role": "exogenous",
            "temporal_status": "time_invariant",
        },
    ]
    edges = [
        {"cause": "G", "effect": "X", "description": "genetic exposure shift", "lagged": False},
        {"cause": "X", "effect": "Y", "description": "treatment effect", "lagged": True},
        {"cause": "Y", "effect": "X", "description": "reverse indication", "lagged": True},
    ]
    indicators = [
        {
            "name": "x_ref",
            "construct_name": "X",
            "construct_polarity": "positive",
            "how_to_measure": "reference X",
            "measurement_dtype": "continuous",
            "aggregation": "mean",
        },
        {
            "name": "x_aux",
            "construct_name": "X",
            "construct_polarity": "positive",
            "how_to_measure": "auxiliary X",
            "measurement_dtype": "continuous",
            "aggregation": "mean",
        },
        {
            "name": "y_ref",
            "construct_name": "Y",
            "construct_polarity": "positive",
            "how_to_measure": "reference Y",
            "measurement_dtype": "continuous",
            "aggregation": "mean",
        },
        {
            "name": "g_ref",
            "construct_name": "G",
            "construct_polarity": "positive",
            "how_to_measure": "genetic marker",
            "measurement_dtype": "continuous",
            "aggregation": "first",
        },
    ]
    return {
        "latent": {"constructs": constructs, "edges": edges},
        "measurement": {"model_clock": "1d", "indicators": indicators},
        "estimation": {
            "state_order": ["X", "Y", "G"],
            "edges": edges,
            "induced_dependencies": [],
        },
        "identifiability": {"identifiable_treatments": {"G": {}, "X": {}}},
    }


def _model_spec() -> dict:
    likelihoods = [
        {
            "variable": "x_ref",
            "distribution": "gaussian",
            "link": "identity",
            "reasoning": "test",
        },
        {
            "variable": "x_aux",
            "distribution": "gaussian",
            "link": "identity",
            "reasoning": "test",
        },
        {
            "variable": "y_ref",
            "distribution": "gaussian",
            "link": "identity",
            "reasoning": "test",
        },
        {
            "variable": "g_ref",
            "distribution": "gaussian",
            "link": "identity",
            "reasoning": "test",
        },
    ]
    chosen_likelihood_by_variable = {
        likelihood["variable"]: likelihood for likelihood in likelihoods
    }
    parameters = []
    for parameter in derive_deterministic_spec(_causal_spec()).all_params:
        if parameter_is_active_for_model_spec(
            parameter,
            chosen_likelihood_by_variable,
            initialization_policy="stationary",
            observation_intercept_policy="free",
            equilibrium_forcing=False,
        ):
            parameters.append(
                {
                    "name": parameter["name"],
                    "role": parameter["role"],
                    "constraint": parameter["constraint"],
                    "description": parameter["description"],
                }
            )
    return {
        "likelihoods": likelihoods,
        "parameters": parameters,
        "initialization_policy": "stationary",
        "observation_intercept_policy": "free",
        "equilibrium_forcing": False,
    }


def _authored_priors(model_spec: dict) -> dict:
    return {
        parameter["name"]: get_default_prior(
            ParameterSpec.model_validate(parameter),
        ).model_dump(mode="json")
        for parameter in model_spec["parameters"]
    }


def _data_for_model() -> pl.DataFrame:
    rows = []
    for day in range(3):
        for indicator in ("x_ref", "x_aux", "y_ref", "g_ref"):
            rows.append(
                {
                    "indicator": indicator,
                    "value": float(day + len(indicator) / 10),
                    "anchor_time": f"2024-01-0{day + 1}T00:00:00Z",
                }
            )
    return pl.DataFrame(rows)


def test_subset_causal_spec_drops_construct_and_incident_edges() -> None:
    subset, retained, dropped = subset_causal_spec(
        _causal_spec(),
        drop_constructs={"G"},
    )

    assert retained == ("X", "Y")
    assert dropped == ("G",)
    assert subset["estimation"]["state_order"] == ["X", "Y"]
    assert {edge["cause"] for edge in subset["estimation"]["edges"]} == {"X", "Y"}
    assert {edge["effect"] for edge in subset["estimation"]["edges"]} == {"X", "Y"}
    assert {indicator["name"] for indicator in subset["measurement"]["indicators"]} == {
        "x_ref",
        "x_aux",
        "y_ref",
    }


def test_subset_model_spec_rebuilds_active_parameter_surface() -> None:
    subset, _, _ = subset_causal_spec(_causal_spec(), drop_constructs={"G"})
    model_spec = subset_model_spec(_model_spec(), subset)
    parameter_names = {parameter["name"] for parameter in model_spec["parameters"]}

    assert "beta_G_X" not in parameter_names
    assert "t0_mean_G" not in parameter_names
    assert "lambda_g_ref_G" not in parameter_names
    assert "beta_X_Y" in parameter_names
    assert "beta_Y_X" in parameter_names
    assert "lambda_x_aux_X" in parameter_names


def test_build_stage5b_subset_runtime_compiles_subset_masks() -> None:
    model_spec = _model_spec()
    subset = build_stage5b_subset_runtime(
        causal_spec=_causal_spec(),
        model_spec=model_spec,
        authored_priors=_authored_priors(model_spec),
        data_for_model=_data_for_model(),
        drop_constructs={"G"},
        sampler_config={"method": "aux_gibbs", "num_warmup": 1, "num_samples": 1, "num_chains": 1},
    )

    assert subset.runtime.spec.latent_names == ["X", "Y"]
    assert subset.runtime.manifest_names == ["x_ref", "x_aux", "y_ref"]
    assert subset.runtime.observations.shape == (3, 3)
    assert all("G" not in parameter for parameter in subset.retained_parameters)
