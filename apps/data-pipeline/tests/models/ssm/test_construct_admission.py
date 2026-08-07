"""End-to-end tests for the gradual construct-admission engine."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts import (
    DistributionFamily,
    LikelihoodSpec,
    LinkFunction,
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
)
from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.prior import ExecutablePrior
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionState,
    AdmissionTiming,
    ConstructContribution,
    DesignInfo,
    _conditional_variance_for_signal,
    _incoming_edge_off_target,
    _resimulate_edge_off,
    _run_battery,
    admit_construct,
    build_construct_order,
)
from nof1_causal_lab.models.ssm.reachability import CheckResult
from nof1_causal_lab.models.structural import build_structural_plan
from nof1_causal_lab.utils.structural_plan import (
    get_edges,
    get_known_inputs,
    get_manifest_indicators,
    get_state_names,
    restrict_structural_plan,
)
from tests.models.ssm.test_dag_to_ssm import _make_causal_design_dict

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm.model import SSMSpec

_SOFT_CHECKS = {
    "C1b confinement",
    "C2 latent scale",
    "C3 resolvability",
    "C4b edge overwhelm",
    "C4c saturation",
    "C5a location reach",
    "C5b width",
    "C5c transmission",
    "C5d data availability",
}
_TARGETS = {"X", "Y", "Z", "x1", "x2", "y1", "z1", "X->Y", "Y->Z"}
_ALL_SOFT = {(check, target): "t" for check in _SOFT_CHECKS for target in _TARGETS}


def _structural_plan() -> StructuralPlan:
    return build_structural_plan(CausalDesign.model_validate(_make_causal_design_dict()))


def _lik(var: str) -> LikelihoodSpec:
    return LikelihoodSpec(
        variable=var,
        distribution=DistributionFamily.GAUSSIAN,
        link=LinkFunction.IDENTITY,
        reasoning="test",
    )


def _p(name: str, role: ParameterRole, constraint: ParameterConstraint) -> ParameterSpec:
    return ParameterSpec(name=name, role=role, constraint=constraint, description="t")


def _normal(parameter: str, mu: float, sigma: float) -> ExecutablePrior:
    return ExecutablePrior.model_validate(
        {
            "parameter": parameter,
            "distribution": "Normal",
            "params": {"mu": mu, "sigma": sigma},
        }
    )


def _halfnormal(parameter: str, sigma: float) -> ExecutablePrior:
    return ExecutablePrior.model_validate(
        {
            "parameter": parameter,
            "distribution": "HalfNormal",
            "params": {"sigma": sigma},
        }
    )


def _contrib_X() -> ConstructContribution:
    return ConstructContribution(
        name="X",
        likelihoods=(_lik("x1"), _lik("x2")),
        parameters=(
            _p("rho_X", ParameterRole.AR_COEFFICIENT, ParameterConstraint.UNIT_INTERVAL),
            _p("sigma_X", ParameterRole.RESIDUAL_SD, ParameterConstraint.POSITIVE),
            _p("lambda_x2_X", ParameterRole.LOADING, ParameterConstraint.POSITIVE),
            _p("obs_sd_x1", ParameterRole.MEASUREMENT_ERROR_SD, ParameterConstraint.POSITIVE),
            _p("obs_sd_x2", ParameterRole.MEASUREMENT_ERROR_SD, ParameterConstraint.POSITIVE),
        ),
        priors={
            "rho_X": _normal("rho_X", 0.6, 0.1),
            "sigma_X": _halfnormal("sigma_X", 0.5),
            "lambda_x2_X": _normal("lambda_x2_X", 1.0, 0.2),
            "obs_sd_x1": _halfnormal("obs_sd_x1", 0.3),
            "obs_sd_x2": _halfnormal("obs_sd_x2", 0.3),
        },
    )


def _contrib_child(name: str, indicator: str, parent: str) -> ConstructContribution:
    return ConstructContribution(
        name=name,
        likelihoods=(_lik(indicator),),
        parameters=(
            _p(f"rho_{name}", ParameterRole.AR_COEFFICIENT, ParameterConstraint.UNIT_INTERVAL),
            _p(f"sigma_{name}", ParameterRole.RESIDUAL_SD, ParameterConstraint.POSITIVE),
            _p(f"beta_{parent}_{name}", ParameterRole.FIXED_EFFECT, ParameterConstraint.NONE),
        ),
        priors={
            f"rho_{name}": _normal(f"rho_{name}", 0.6, 0.1),
            f"sigma_{name}": _halfnormal(f"sigma_{name}", 0.5),
            f"beta_{parent}_{name}": _normal(f"beta_{parent}_{name}", 0.3, 0.1),
        },
        edge_parents=(parent,),
    )


def _design(seed: int = 0) -> DesignInfo:
    t_grid = jnp.linspace(0.0, 10.0, 201)
    obs_idx = np.arange(1, 201, 2)  # 100 observations, shared across indicators here
    rng = np.random.default_rng(0)
    indicators = ("x1", "x2", "y1", "z1")
    return DesignInfo(
        t_grid=t_grid,
        obs_index_by_indicator=dict.fromkeys(indicators, obs_idx),
        values_by_indicator={v: rng.normal(0.0, 0.9, obs_idx.size) for v in indicators},
        n_draws=64,
        seed=seed,
    )


def test_build_construct_order_is_topological():
    order = build_construct_order(_structural_plan())
    assert order.index("X") < order.index("Y") < order.index("Z")


def test_admit_construct_records_shared_and_diagnostic_timings(monkeypatch):
    from nof1_causal_lab.models.ssm import construct_admission as admission_module

    diagnostic = AdmissionTiming(
        phase="c1_confinement",
        label="C1 confinement",
        duration_ms=4.0,
        checks=("C1a finiteness",),
    )
    monkeypatch.setattr(admission_module, "_compile_partial", lambda *_args: (object(), object()))
    monkeypatch.setattr(admission_module, "_sample_partial", lambda *_args: {})
    monkeypatch.setattr(
        admission_module,
        "_run_battery",
        lambda *_args: (
            [CheckResult("C1a finiteness", "X", "0%", "0%", True, "ok")],
            [diagnostic],
        ),
    )

    _state, report = admission_module.admit_construct(
        AdmissionState(),
        ConstructContribution(name="X"),
        _structural_plan(),
        _design(),
    )

    assert [timing.phase for timing in report.timings] == [
        "model_compilation",
        "prior_predictive",
        "c1_confinement",
        "admission_decision",
    ]
    assert all(timing.duration_ms >= 0 for timing in report.timings)


def test_conditional_variance_uses_observation_family_moments():
    signal = np.array([[0.1, 0.5], [0.2, 0.8]])
    pred = {
        "manifest_cov": np.array([[[0.25]], [[1.0]]]),
        "obs_df": np.array([5.0, 1.5]),
        "obs_shape": np.array([2.0, 4.0]),
        "obs_r": np.array([3.0, 6.0]),
        "obs_concentration": np.array([9.0, 19.0]),
    }

    gaussian = _conditional_variance_for_signal(DistributionFamily.GAUSSIAN, signal, pred, 0)
    np.testing.assert_allclose(gaussian, [[0.25, 0.25], [1.0, 1.0]])
    np.testing.assert_allclose(
        _conditional_variance_for_signal(DistributionFamily.POISSON, signal, pred, 0),
        signal,
    )
    np.testing.assert_allclose(
        _conditional_variance_for_signal(DistributionFamily.BERNOULLI, signal, pred, 0),
        signal * (1.0 - signal),
    )
    student = _conditional_variance_for_signal(DistributionFamily.STUDENT_T, signal, pred, 0)
    np.testing.assert_allclose(student[0], np.full(2, 0.25 * 5.0 / 3.0))
    assert np.isinf(student[1]).all()


def test_time_invariant_construct_omits_temporal_transmission_check():
    draws = 200
    times = 20
    static_values = np.linspace(-1.5, 1.5, draws)
    latent = np.broadcast_to(static_values[:, None, None], (draws, times, 1))
    expected = latent.copy()
    pred = {
        "latents": latent,
        "observations": expected,
        "expected_observations": expected,
        "manifest_cov": np.broadcast_to(np.array([[[0.25]]]), (draws, 1, 1)),
    }
    spec: Any = SimpleNamespace(
        latent_names=["static"],
        manifest_names=["static_indicator"],
        manifest_links=[LinkFunction.IDENTITY],
        manifest_level_counts=None,
        dynamics_spec=SimpleNamespace(components=()),
        diffusion_block=SimpleNamespace(time_invariant_mask=np.array([True])),
    )
    obs_idx = np.arange(times)
    design = DesignInfo(
        t_grid=jnp.arange(times, dtype=float),
        obs_index_by_indicator={"static_indicator": obs_idx},
        values_by_indicator={"static_indicator": np.linspace(-1.0, 1.0, times)},
    )
    target = ConstructContribution(
        name="static",
        likelihoods=(_lik("static_indicator"),),
    )

    results, _timings = _run_battery(spec, pred, design, target)
    checks = {result.check for result in results}
    assert {"C5a location reach", "C5b width"} <= checks
    assert "C5c transmission" not in checks


def test_build_construct_order_covers_only_estimation_states():
    """Constructs marginalized/anchored/dropped out of the estimation
    projection carry no state — nothing to admit for them."""
    causal_design = _make_causal_design_dict()
    causal_design["latent"]["constructs"].append(
        {
            "name": "M",
            "description": "Marginalized confounder",
            "role": "endogenous",
            "temporal_status": "time_varying",
        }
    )
    order = build_construct_order(build_structural_plan(CausalDesign.model_validate(causal_design)))
    assert order == ["X", "Y", "Z"]


def test_build_construct_order_admits_lagged_feedback_cycles():
    """Lagged feedback loops sort as a unit: cycle members adjacent, parents first."""
    causal_design = _make_causal_design_dict()
    feedback = {"cause": "Z", "effect": "Y", "description": "Z feeds back on Y", "lagged": True}
    causal_design["latent"]["edges"].append(dict(feedback))
    order = build_construct_order(build_structural_plan(CausalDesign.model_validate(causal_design)))
    assert order == ["X", "Y", "Z"]


def test_restrict_structural_plan_to_subset():
    restricted = restrict_structural_plan(_structural_plan(), {"X", "Y"})
    assert get_state_names(restricted) == ["X", "Y"]
    assert {indicator["name"] for indicator in get_manifest_indicators(restricted)} == {
        "x1",
        "x2",
        "y1",
    }
    assert all(edge["effect"] != "Z" for edge in get_edges(restricted))


def test_restrict_structural_plan_preserves_known_input_dependency():
    causal_design = _make_causal_design_dict()
    causal_design["known_inputs"] = [
        {
            "construct": "X",
            "source_indicator": "x1",
            "scale": 10.0,
            "missing_policy": "forward_fill",
        }
    ]

    plan = build_structural_plan(CausalDesign.model_validate(causal_design))
    restricted = restrict_structural_plan(plan, {"Y"})

    assert get_state_names(restricted) == ["Y"]
    assert [
        {
            "construct": item["construct"],
            "source_indicator": item["source_indicator"],
            "scale": item["scale"],
            "missing_policy": item["missing_policy"],
        }
        for item in get_known_inputs(restricted)
    ] == [
        {
            "construct": "X",
            "source_indicator": "x1",
            "scale": 10.0,
            "missing_policy": "forward_fill",
        }
    ]
    assert [(edge["cause"], edge["effect"]) for edge in get_edges(restricted)] == [("X", "Y")]
    assert {indicator["name"] for indicator in get_manifest_indicators(restricted)} == {"y1"}


def test_known_input_edge_off_zeroes_only_the_compiled_input_cell(monkeypatch):
    from nof1_causal_lab.models.ssm.predictive import registry_runtime

    captured: dict[str, np.ndarray] = {}

    def _capture_samples(
        _spec,
        samples,
        _times,
        *,
        transition_inputs,
        rng_key,
    ):
        del transition_inputs, rng_key
        captured["input_effect"] = np.asarray(samples["input_effect"])
        return jnp.zeros((2, 3, 2)), jnp.zeros((2, 3, 1))

    monkeypatch.setattr(
        registry_runtime,
        "_simulate_vector_field_predictive_latents",
        _capture_samples,
    )
    spec = cast("SSMSpec", SimpleNamespace(input_names=["dose", "exercise"]))
    contribution = ConstructContribution(name="mood", edge_parents=("dose",))
    edge_target = _incoming_edge_off_target(spec, contribution, ["mood", "sleep"], 0)
    input_effect = jnp.arange(8, dtype=float).reshape(2, 2, 2) + 1.0

    _resimulate_edge_off(
        spec,
        {"input_effect": input_effect},
        jnp.arange(3, dtype=float),
        edge_target,
        seed=1,
    )

    expected = np.asarray(input_effect).copy()
    expected[:, 0, 0] = 0.0
    np.testing.assert_allclose(captured["input_effect"], expected)


@pytest.mark.slow
def test_admit_root_runs_full_battery():
    structural_plan = _structural_plan()
    state, report = admit_construct(
        AdmissionState(), _contrib_X(), structural_plan, _design(), accepted=_ALL_SOFT
    )
    ids = {r.check for r in report.results}
    assert {"C1a finiteness", "C1b confinement", "C2 latent scale", "C3 resolvability"} <= ids
    assert {"C5a location reach", "C5b width", "C5c transmission"} <= ids
    assert "C4b edge overwhelm" not in ids  # root has no incoming edge
    timing_phases = {timing.phase for timing in report.timings}
    assert {
        "model_compilation",
        "prior_predictive",
        "c1_confinement",
        "c2_latent_scale",
        "c3_resolvability",
        "admission_decision",
    } <= timing_phases
    assert all(timing.duration_ms > 0 for timing in report.timings)
    # Hard checks (finite sim + reachable data) hold, so X is admitted.
    assert not report.outcome.startswith("BLOCKED")
    assert report.admitted
    assert state.names == ("X",)


@pytest.mark.slow
def test_admit_child_runs_edge_check_via_edge_off_resim():
    structural_plan = _structural_plan()
    design = _design()
    state, _ = admit_construct(
        AdmissionState(), _contrib_X(), structural_plan, design, accepted=_ALL_SOFT
    )
    state, report = admit_construct(
        state, _contrib_child("Y", "y1", "X"), structural_plan, design, accepted=_ALL_SOFT
    )
    ids = {r.check for r in report.results}
    assert "C4b edge overwhelm" in ids
    c4b = next(r for r in report.results if r.check == "C4b edge overwhelm")
    # The edge-off re-simulation must actually differ from edge-on (the edge moves
    # the child); a zero displacement would mean the resim was a no-op.
    assert c4b.evidence is not None
    assert float(np.median(c4b.evidence["e"])) > 0.0
    assert report.admitted
    assert state.names == ("X", "Y")


@pytest.mark.slow
def test_full_chain_builds_and_compiles_to_ssm_artifact():
    import polars as pl

    from nof1_causal_lab.artifacts.prior import PriorPlan
    from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
    from nof1_causal_lab.models.ssm.runtime import hydrate_compiled_model

    structural_plan = _structural_plan()
    contributions = {
        "X": _contrib_X(),
        "Y": _contrib_child("Y", "y1", "X"),
        "Z": _contrib_child("Z", "z1", "Y"),
    }
    accepted = dict.fromkeys(contributions, _ALL_SOFT)
    state = AdmissionState()
    reports = []
    for name in build_construct_order(structural_plan):
        state, report = admit_construct(
            state,
            contributions[name],
            structural_plan,
            _design(),
            accepted[name],
        )
        reports.append(report)
        assert report.admitted
    assert [r.name for r in reports] == ["X", "Y", "Z"]
    assert state.names == ("X", "Y", "Z")

    # The accumulated StatisticalModelSpec + priors compile to the real compiled_ssm artifact
    # the stage produces, and build a live, fittable 3-latent structure.
    compiled = compile_ssm_artifact(
        state.statistical_model_spec(),
        PriorPlan(priors=dict(state.priors)),
        structural_plan,
    )
    assert compiled.spec is not None
    assert compiled.schema_version == 2
    wide = pl.DataFrame(
        {
            "time": list(range(10)),
            "x1": [0.1] * 10,
            "x2": [0.2] * 10,
            "y1": [0.3] * 10,
            "z1": [0.4] * 10,
        }
    )
    model = hydrate_compiled_model(compiled, wide)
    assert model.spec.n_latent == 3
