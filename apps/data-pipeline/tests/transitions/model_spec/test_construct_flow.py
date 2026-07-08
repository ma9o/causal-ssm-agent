"""Unit tests for the gradual construct-admission model-spec flow (data-free pieces).

The full loop over real ``data_for_model`` (prior-predictive reachability on live
data) is exercised as a Commit-6 integration regression; here we pin the pure
payload → contribution mapping, feedback rendering, prompt assembly, and the
out-of-order submission guard.
"""

from __future__ import annotations

import polars as pl

from nof1_causal_lab.artifacts import (
    DistributionFamily,
    LinkFunction,
    ParameterConstraint,
    ParameterRole,
)
from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
    SUBMIT_CONSTRUCT_SCHEMA,
    ConstructBuildState,
    ParamCatalog,
    _closed_loop_target,
    _closing_edge_effects,
    construct_parents,
    contribution_from_payload,
    deferred_closing_edge_params,
    render_admission_feedback,
)
from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_prompt import (
    build_construct_messages,
)
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionReport,
    AdmissionState,
    ConstructContribution,
)
from nof1_causal_lab.models.ssm.reachability import CheckResult
from tests.models.ssm.test_dag_to_ssm import _make_causal_design_dict


def _normal(mu: float, sigma: float) -> dict:
    return {"distribution": "Normal", "params": {"mu": mu, "sigma": sigma}}


def test_param_catalog_reflects_compiler_free_params():
    catalog = ParamCatalog.from_causal_design(_make_causal_design_dict())
    # X has two indicators → both measurement-noise + the free loading are authorable.
    assert "obs_sd_x1" in catalog.by_construct["X"]
    assert "lambda_x2_X" in catalog.by_construct["X"]
    assert catalog.role_for("lambda_x2_X") == (
        ParameterRole.LOADING,
        ParameterConstraint.POSITIVE,
    )
    # Single-indicator Y: its measurement noise is NOT a free parameter (absorbed
    # into process noise); authoring obs_sd_y1 must be rejected, but the edge is free.
    y_allowed = catalog.allowed_for("Y", ["X"])
    assert "obs_sd_y1" not in y_allowed
    assert "beta_X_Y" in y_allowed
    assert catalog.role_for("beta_X_Y") == (ParameterRole.FIXED_EFFECT, ParameterConstraint.NONE)
    # Structural extensions (self-limiting quartic, Hill edge) are always offerable.
    assert "self_limit_Y" in y_allowed
    assert "hill_emax_X_Y" in y_allowed
    assert catalog.role_for("self_limit_Y")[0] == ParameterRole.DYNAMICS_PARAMETER_POSITIVE
    # Policy-pinned surfaces the admission-time compile never frees (STATIONARY
    # initialization; no equilibrium forcing) are NOT offered — surfacing them is
    # what made the agent author priors the compiler then rejects as "not free".
    # X is a dynamic construct (has rho_X/sigma_X), so its initial state and
    # well-centre are pinned.
    assert "cint_X" not in catalog.allowed_for("X", [])
    assert "t0_mean_X" not in catalog.by_construct["X"]
    assert "t0_sd_X" not in catalog.by_construct["X"]


def test_submit_construct_rejects_non_free_parameter():
    state = ConstructBuildState(
        causal_design=_make_causal_design_dict(),
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
    )
    feedback = state.submit_construct(
        construct="X",
        indicators=[{"variable": "x1", "family": "gaussian", "link": "identity"}],
        priors={"obs_sd_bogus": _normal(0.0, 1.0)},
    )
    assert "not free" in feedback
    assert state.current_construct == "X"  # not admitted


def test_construct_parents_reads_the_dag():
    spec = _make_causal_design_dict()
    assert construct_parents(spec, "Y") == ["X"]
    assert construct_parents(spec, "Z") == ["Y"]
    assert construct_parents(spec, "X") == []


def test_deferred_closing_edge_params_cover_feedback_cycles():
    """The cycle-closing edge's priors become authorable on the second member's turn."""
    spec = _make_causal_design_dict()
    feedback = {"cause": "Z", "effect": "Y", "description": "Z feeds back on Y", "lagged": True}
    spec["latent"]["edges"].append(dict(feedback))
    spec["estimation"]["edges"].append(dict(feedback))

    # Y admitted first (its restricted spec has no Z edge) — nothing deferred for it.
    assert deferred_closing_edge_params(spec, "Y", admitted=set()) == set()
    # Z joins with Y already admitted: the closing edge Z->Y materializes now.
    names = deferred_closing_edge_params(spec, "Z", admitted={"X", "Y"})
    assert "beta_Z_Y" in names
    assert "hill_emax_Z_Y" in names
    # A plain downstream edge (Y->Z with Z being admitted) is not a closing edge.
    assert deferred_closing_edge_params(spec, "Z", admitted=set()) == set()


def test_closing_edge_effects_detects_the_rechecked_member():
    spec = _make_causal_design_dict()
    feedback = {"cause": "Z", "effect": "Y", "description": "Z feeds back on Y", "lagged": True}
    spec["estimation"]["edges"].append(dict(feedback))
    # Admitting Z with Y already admitted closes the Y<->Z loop → Y is the member to recheck.
    assert _closing_edge_effects(spec, "Z", {"X", "Y"}) == ["Y"]
    # Admitting Y (only X admitted) closes no loop — Y->Z's effect isn't admitted yet.
    assert _closing_edge_effects(spec, "Y", {"X"}) == []


def test_closed_loop_target_includes_the_closing_feedback_edge():
    spec = _make_causal_design_dict()
    feedback = {"cause": "Z", "effect": "Y", "description": "Z feeds back on Y", "lagged": True}
    spec["estimation"]["edges"].append(dict(feedback))
    # Y was admitted open-loop with just X->Y; once Z closes the loop the priors also carry
    # beta_Z_Y, so the recheck target must see BOTH parents to re-measure edge overwhelm.
    member_y = ConstructContribution(name="Y", edge_parents=("X",))
    target = _closed_loop_target(
        member_y, spec, {"beta_X_Y": _normal(0.3, 0.1), "beta_Z_Y": _normal(0.2, 0.1)}
    )
    assert target.edge_parents == ("X", "Z")
    assert target.hill_parents == ()
    # A saturating closing edge registers as both an edge parent and a Hill parent.
    hill_target = _closed_loop_target(
        member_y, spec, {"beta_X_Y": _normal(0.3, 0.1), "hill_emax_Z_Y": _normal(1.0, 0.5)}
    )
    assert hill_target.edge_parents == ("X", "Z")
    assert hill_target.hill_parents == ("Z",)


def test_contribution_from_payload_linear_edge():
    spec = _make_causal_design_dict()
    payload = {
        "construct": "Y",
        "indicators": [{"variable": "y1", "family": "gaussian", "link": "identity"}],
        "priors": {
            "rho_Y": _normal(0.6, 0.1),
            "sigma_Y": {"distribution": "HalfNormal", "params": {"sigma": 0.5}},
            "beta_X_Y": _normal(0.3, 0.1),
        },
    }
    contrib = contribution_from_payload(spec, payload, ParamCatalog.from_causal_design(spec))
    assert contrib.name == "Y"
    assert [lik.variable for lik in contrib.likelihoods] == ["y1"]
    assert contrib.likelihoods[0].distribution == DistributionFamily.GAUSSIAN
    assert contrib.likelihoods[0].link == LinkFunction.IDENTITY
    assert {p.name for p in contrib.parameters} == {"rho_Y", "sigma_Y", "beta_X_Y"}
    assert contrib.edge_parents == ("X",)
    assert contrib.hill_parents == ()


def test_contribution_from_payload_hill_edge_and_self_limit():
    spec = _make_causal_design_dict()
    payload = {
        "construct": "Y",
        "indicators": [{"variable": "y1", "family": "gaussian", "link": "identity"}],
        "priors": {
            "rho_Y": _normal(0.6, 0.1),
            "self_limit_Y": {"distribution": "HalfNormal", "params": {"sigma": 0.5}},
            "hill_emax_X_Y": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
            "hill_ec50_X_Y": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
            "hill_n_X_Y": {"distribution": "HalfNormal", "params": {"sigma": 2.0}},
        },
    }
    contrib = contribution_from_payload(spec, payload, ParamCatalog.from_causal_design(spec))
    assert contrib.edge_parents == ("X",)
    assert contrib.hill_parents == ("X",)
    self_limit = next(p for p in contrib.parameters if p.name == "self_limit_Y")
    assert self_limit.role == ParameterRole.DYNAMICS_PARAMETER_POSITIVE


def test_submit_construct_rejects_out_of_order():
    state = ConstructBuildState(
        causal_design=_make_causal_design_dict(),
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
    )
    # Active construct is X; submitting Y is rejected before any compilation.
    feedback = state.submit_construct(
        construct="Y",
        indicators=[{"variable": "y1", "family": "gaussian", "link": "identity"}],
        priors={"rho_Y": _normal(0.6, 0.1)},
    )
    assert "Out-of-order" in feedback
    assert state.current_construct == "X"  # unchanged
    assert state.submission_made is True


def test_render_admission_feedback_lists_failed_checks():
    report = AdmissionReport(
        name="Y",
        results=(
            CheckResult("C1a finiteness", "Y", "0%", "0%", True, "ok"),
            CheckResult("C3 resolvability", "Y", "0.05 d", "[0.3, 2.5]", False, "too fast"),
        ),
        outcome="NEEDS DECISION: C3 resolvability",
        annotations=(),
        admitted=False,
    )
    text = render_admission_feedback(report)
    assert "NEEDS DECISION" in text
    assert "C3 resolvability" in text
    assert "[FAIL]" in text
    assert "[PASS]" in text


def test_build_construct_messages_surfaces_params_and_feedback():
    spec = _make_causal_design_dict()
    state = ConstructBuildState(
        causal_design=spec,
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
        admission=AdmissionState(names=("X",)),
        cursor=1,  # active construct is Y, with X already admitted
    )
    system, user = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        causal_design=spec,
        indicator_audits={},
    )
    assert "continuous-time latent state-space model" in system
    assert "Active construct: `Y`" in user
    assert "`rho_Y`" in user  # own-dynamics param offered
    assert "`beta_X_Y`" in user  # parent edge param offered
    assert "Does X drive Y?" in user

    # On a re-attempt, the last failing report for this construct is injected.
    state.last_report = AdmissionReport(
        name="Y",
        results=(CheckResult("C2 latent scale", "Y", "8.0", "[0.3, 3]", False, "too wide"),),
        outcome="NEEDS DECISION: C2 latent scale",
        annotations=(),
        admitted=False,
    )
    _system2, user2 = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        causal_design=spec,
        indicator_audits={},
    )
    assert "Latest reachability feedback" in user2
    assert "C2 latent scale" in user2


def test_submit_construct_schema_is_well_formed():
    props = SUBMIT_CONSTRUCT_SCHEMA["properties"]
    assert set(SUBMIT_CONSTRUCT_SCHEMA["required"]) == {"construct", "indicators", "priors"}
    assert SUBMIT_CONSTRUCT_SCHEMA["additionalProperties"] is False
    family_enum = props["indicators"]["items"]["properties"]["family"]["enum"]
    assert "gaussian" in family_enum
    assert "beta" in family_enum


def test_public_entrypoints_are_exposed():
    # The build loop + design derivation are wired into the stage flow in Commit 5;
    # here we pin their public surface and that the tool builds with the right name.
    from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
        build_design_info,
        make_submit_construct_tool,
        run_model_spec_construct_build,
    )

    assert callable(run_model_spec_construct_build)
    assert callable(build_design_info)
    state = ConstructBuildState(
        causal_design=_make_causal_design_dict(), data_for_model=pl.DataFrame(), order=["X"]
    )
    tool = make_submit_construct_tool(state)
    assert tool.name == "submit_construct"
