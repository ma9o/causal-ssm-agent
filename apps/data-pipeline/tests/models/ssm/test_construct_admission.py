"""End-to-end tests for the gradual construct-admission engine."""

from __future__ import annotations

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
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionState,
    ConstructContribution,
    DesignInfo,
    admit_construct,
    build_construct_order,
    restrict_causal_spec,
    run_construct_build,
)
from tests.models.ssm.test_dag_to_ssm import _make_causal_spec_dict

_ALL_SOFT = {
    "C1b confinement": "t",
    "C2 latent scale": "t",
    "C3 resolvability": "t",
    "C4b edge overwhelm": "t",
    "C4c saturation": "t",
    "C5b width": "t",
    "C5c transmission": "t",
}


def _lik(var: str) -> LikelihoodSpec:
    return LikelihoodSpec(
        variable=var,
        distribution=DistributionFamily.GAUSSIAN,
        link=LinkFunction.IDENTITY,
        reasoning="test",
    )


def _p(name: str, role: ParameterRole, constraint: ParameterConstraint) -> ParameterSpec:
    return ParameterSpec(name=name, role=role, constraint=constraint, description="t")


def _normal(mu: float, sigma: float) -> dict:
    return {"distribution": "Normal", "params": {"mu": mu, "sigma": sigma}}


def _halfnormal(sigma: float) -> dict:
    return {"distribution": "HalfNormal", "params": {"sigma": sigma}}


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
            "rho_X": _normal(0.6, 0.1),
            "sigma_X": _halfnormal(0.5),
            "lambda_x2_X": _normal(1.0, 0.2),
            "obs_sd_x1": _halfnormal(0.3),
            "obs_sd_x2": _halfnormal(0.3),
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
            f"rho_{name}": _normal(0.6, 0.1),
            f"sigma_{name}": _halfnormal(0.5),
            f"beta_{parent}_{name}": _normal(0.3, 0.1),
        },
        edge_parents=(parent,),
    )


def _design(seed: int = 0) -> DesignInfo:
    t_grid = jnp.linspace(0.0, 10.0, 201)
    obs_idx = np.arange(1, 201, 2)  # 100 observations
    obs_times = np.asarray(t_grid)[obs_idx]
    rng = np.random.default_rng(0)
    data = {v: rng.normal(0.0, 0.9, obs_idx.size) for v in ("x1", "x2", "y1", "z1")}
    return DesignInfo(
        t_grid=t_grid,
        obs_idx=obs_idx,
        cadence=float(np.median(np.diff(obs_times))),
        span=float(np.ptp(obs_times)),
        data=data,
        n_draws=64,
        seed=seed,
    )


def test_build_construct_order_is_topological():
    order = build_construct_order(_make_causal_spec_dict())
    assert order.index("X") < order.index("Y") < order.index("Z")


def test_restrict_causal_spec_to_subset():
    restricted = restrict_causal_spec(_make_causal_spec_dict(), {"X", "Y"})
    names = {c["name"] for c in restricted["latent"]["constructs"]}
    assert names == {"X", "Y"}
    assert restricted["estimation"]["state_order"] == ["X", "Y"]
    inds = {i["name"] for i in restricted["measurement"]["indicators"]}
    assert inds == {"x1", "x2", "y1"}  # z1 dropped
    assert all(e["effect"] != "Z" for e in restricted["estimation"]["edges"])


@pytest.mark.slow
def test_admit_root_runs_full_battery():
    causal_spec = _make_causal_spec_dict()
    state, report = admit_construct(
        AdmissionState(), _contrib_X(), causal_spec, _design(), accepted=_ALL_SOFT
    )
    ids = {r.check for r in report.results}
    assert {"C1a finiteness", "C1b confinement", "C2 latent scale", "C3 resolvability"} <= ids
    assert {"C5a location reach", "C5b width", "C5c transmission"} <= ids
    assert "C4b edge overwhelm" not in ids  # root has no incoming edge
    # Hard checks (finite sim + reachable data) hold, so X is admitted.
    assert not report.outcome.startswith("BLOCKED")
    assert report.admitted
    assert state.names == ("X",)


@pytest.mark.slow
def test_admit_child_runs_edge_check_via_edge_off_resim():
    causal_spec = _make_causal_spec_dict()
    design = _design()
    state, _ = admit_construct(
        AdmissionState(), _contrib_X(), causal_spec, design, accepted=_ALL_SOFT
    )
    state, report = admit_construct(
        state, _contrib_child("Y", "y1", "X"), causal_spec, design, accepted=_ALL_SOFT
    )
    ids = {r.check for r in report.results}
    assert "C4b edge overwhelm" in ids
    c4b = next(r for r in report.results if r.check == "C4b edge overwhelm")
    # The edge-off re-simulation must actually differ from edge-on (the edge moves
    # the child); a zero displacement would mean the resim was a no-op.
    assert float(np.median(c4b.evidence["e"])) > 0.0
    assert report.admitted
    assert state.names == ("X", "Y")


@pytest.mark.slow
def test_full_chain_builds_and_compiles_to_ssm_artifact():
    import polars as pl

    from nof1_causal_lab.models.ssm.compile.artifact import (
        build_model_from_compiled_artifact,
        compile_ssm_artifact,
    )

    causal_spec = _make_causal_spec_dict()
    contributions = {
        "X": _contrib_X(),
        "Y": _contrib_child("Y", "y1", "X"),
        "Z": _contrib_child("Z", "z1", "Y"),
    }
    accepted = dict.fromkeys(contributions, _ALL_SOFT)
    state, reports = run_construct_build(causal_spec, contributions, _design(), accepted)
    assert [r.name for r in reports] == ["X", "Y", "Z"]
    assert all(r.admitted for r in reports)
    assert state.names == ("X", "Y", "Z")

    # The accumulated ModelSpec + priors compile to the real compiled_ssm artifact
    # the stage produces, and build a live, fittable 3-latent model.
    compiled = compile_ssm_artifact(state.model_spec(), dict(state.priors), causal_spec=causal_spec)
    assert "spec" in compiled
    assert "schema_version" in compiled
    wide = pl.DataFrame(
        {
            "time": list(range(10)),
            "x1": [0.1] * 10,
            "x2": [0.2] * 10,
            "y1": [0.3] * 10,
            "z1": [0.4] * 10,
        }
    )
    model = build_model_from_compiled_artifact(compiled, wide)
    assert model.spec.n_latent == 3
