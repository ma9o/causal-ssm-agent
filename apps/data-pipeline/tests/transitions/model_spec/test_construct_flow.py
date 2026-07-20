"""Unit tests for the gradual construct-admission model-spec flow (data-free pieces).

The Temporal workflow owns the live-data orchestration; here we pin the pure
payload → contribution mapping, feedback rendering, prompt assembly, and the
out-of-order submission guard.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
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
    _admission_report_payload,
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
    AdmissionTiming,
    ConstructContribution,
    _signal_from_linear_predictor,
)
from nof1_causal_lab.models.ssm.reachability import CheckResult
from tests.models.ssm.test_dag_to_ssm import _make_causal_design_dict

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec


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


def test_param_catalog_surfaces_static_mean_when_standardization_can_activate_it():
    causal_design = _make_causal_design_dict()
    causal_design["latent"]["constructs"][0]["temporal_status"] = "time_invariant"
    catalog = ParamCatalog.from_causal_design(causal_design)
    assert "t0_mean_X" in catalog.by_construct["X"]


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


def test_submit_construct_rejects_intercept_inactive_for_locked_likelihood():
    causal_design = _make_causal_design_dict()
    x1 = next(
        indicator
        for indicator in causal_design["measurement"]["indicators"]
        if indicator["name"] == "x1"
    )
    x1.update(
        measurement_dtype="ordinal",
        aggregation="last",
        ordinal_levels=["low", "medium", "high"],
    )
    state = ConstructBuildState(
        causal_design=causal_design,
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
    )
    feedback = state.submit_construct(
        construct="X",
        indicators=[
            {"variable": "x1", "family": "ordered_logistic", "link": "cumulative_logit"},
            {"variable": "x2", "family": "gaussian", "link": "identity"},
        ],
        priors={"manifest_mean_x1": _normal(0.0, 1.0)},
    )
    assert "not free" in feedback
    assert "manifest_mean_x1" in feedback
    assert state.current_construct == "X"


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


def test_submit_construct_rejects_mixed_family_in_pooled_site():
    state = ConstructBuildState(
        causal_design=_make_causal_design_dict(),
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
        admission=AdmissionState(
            names=("X",),
            priors={
                "sigma_X": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 0.5, "sigma": 0.1, "lower": 0.1, "upper": 1.0},
                }
            },
        ),
        cursor=1,
    )

    feedback = state.submit_construct(
        construct="Y",
        indicators=[{"variable": "y1", "family": "gaussian", "link": "identity"}],
        priors={
            "sigma_Y": {"distribution": "HalfNormal", "params": {"sigma": 0.5}},
            "beta_X_Y": _normal(0.3, 0.1),
        },
    )

    assert "Prior family mismatch" in feedback
    assert "diffusion_diag_free" in feedback
    assert state.current_construct == "Y"


def test_feedback_closure_hard_recheck_blocks_commit(monkeypatch):
    from nof1_causal_lab.flows.transitions.model_spec.agentic import construct_flow as module

    causal_design = _make_causal_design_dict()
    feedback_edge = {"cause": "Y", "effect": "X", "lagged": True}
    causal_design["latent"]["edges"].append(dict(feedback_edge))
    causal_design["estimation"]["edges"].append(dict(feedback_edge))
    initial = AdmissionState(names=("X",))
    state = ConstructBuildState(
        causal_design=causal_design,
        data_for_model=pl.DataFrame(),
        order=["Y"],
        admission=initial,
    )
    active_pass = CheckResult("C1a finiteness", "Y", "0%", "0%", True, "ok")
    hard_recheck = CheckResult("C1a finiteness", "X", "1%", "0%", False, "bad")
    tentative = AdmissionState(names=("X", "Y"))
    admitted_report = AdmissionReport(
        name="Y",
        results=(active_pass,),
        timings=(),
        outcome="ADMITTED",
        annotations=(),
        admitted=True,
    )
    monkeypatch.setattr(module, "build_design_info", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module,
        "admit_construct",
        lambda *_args, **_kwargs: (tentative, admitted_report),
    )
    monkeypatch.setattr(
        ConstructBuildState,
        "_coupled_recheck",
        lambda *_args, **_kwargs: ([hard_recheck], {"results": [], "timings": []}),
    )

    feedback = state.submit_construct(
        construct="Y",
        indicators=[{"variable": "y1", "family": "gaussian", "link": "identity"}],
        priors={"beta_Y_X": _normal(0.2, 0.1)},
    )

    assert "BLOCKED" in feedback
    assert state.admission is initial
    assert state.current_construct == "Y"


def test_render_admission_feedback_lists_failed_checks():
    report = AdmissionReport(
        name="Y",
        results=(
            CheckResult("C1a finiteness", "Y", "0%", "0%", True, "ok"),
            CheckResult("C3 resolvability", "Y", "0.05 d", "[0.3, 2.5]", False, "too fast"),
        ),
        timings=(),
        outcome="NEEDS DECISION: C3 resolvability",
        annotations=(),
        admitted=False,
    )
    text = render_admission_feedback(report)
    assert "NEEDS DECISION" in text
    assert "C3 resolvability" in text
    assert "[FAIL]" in text
    assert "[PASS]" in text


def test_admission_report_payload_includes_backend_timing_breakdown():
    report = AdmissionReport(
        name="X",
        results=(CheckResult("C1a finiteness", "X", "0%", "0%", True, "ok"),),
        timings=(
            AdmissionTiming("model_compilation", "Model compilation", 12.5),
            AdmissionTiming(
                "c1_confinement",
                "C1 confinement",
                3.25,
                ("C1a finiteness",),
            ),
        ),
        outcome="ADMITTED",
        annotations=(),
        admitted=True,
    )

    payload = _admission_report_payload(report, ConstructContribution(name="X"), attempt=2)

    assert payload["attempt"] == 2
    assert payload["timings"] == [
        {
            "phase": "model_compilation",
            "label": "Model compilation",
            "duration_ms": 12.5,
            "checks": [],
        },
        {
            "phase": "c1_confinement",
            "label": "C1 confinement",
            "duration_ms": 3.25,
            "checks": ["C1a finiteness"],
        },
    ]


def test_ordered_logistic_signal_uses_sampled_cutpoints():
    linear_predictor = np.array([[-2.0, 0.0, 2.0], [-1.0, 0.5, 1.5]])
    predictive = {
        "obs_ordered_base": np.zeros((2, 1)),
        "obs_ordered_gaps": np.ones((2, 1, 2)),
    }

    signal = _signal_from_linear_predictor(
        LinkFunction.CUMULATIVE_LOGIT,
        linear_predictor,
        spec=cast("SSMSpec", SimpleNamespace(manifest_level_counts=[4])),
        pred=predictive,
        manifest_index=0,
    )

    assert signal.shape == (*linear_predictor.shape, 4)
    assert np.allclose(signal.sum(axis=2), 1.0)
    expected_category = np.sum(signal * np.arange(4), axis=2)
    assert np.all(np.diff(expected_category, axis=1) > 0)


def test_build_construct_messages_surfaces_params_and_feedback():
    spec = _make_causal_design_dict()
    state = ConstructBuildState(
        causal_design=spec,
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
        admission=AdmissionState(
            names=("X",),
            priors={
                "sigma_X": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 0.5, "sigma": 0.1, "lower": 0.1, "upper": 1.0},
                }
            },
        ),
        cursor=1,  # active construct is Y, with X already admitted
    )
    system, user = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        causal_design=spec,
        validation_report={"indicators": {}},
    )
    assert "continuous-time latent state-space model" in system
    assert "invoking the registered MCP tool `submit_construct`" in system
    assert "`indicators`, not" in system
    assert "`emissions`" in system
    assert "Active construct: `Y`" in user
    assert "`rho_Y`" in user  # own-dynamics param offered
    assert "`beta_X_Y`" in user  # parent edge param offered
    assert "Does X drive Y?" in user
    assert "MUST use `TruncatedNormal` to match admitted parameters" in user

    # On a re-attempt, the last failing report for this construct is injected.
    state.last_report = AdmissionReport(
        name="Y",
        results=(CheckResult("C2 latent scale", "Y", "8.0", "[0.3, 3]", False, "too wide"),),
        timings=(),
        outcome="NEEDS DECISION: C2 latent scale",
        annotations=(),
        admitted=False,
    )
    _system2, user2 = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        causal_design=spec,
        validation_report={"indicators": {}},
    )
    assert "Latest reachability feedback" in user2
    assert "C2 latent scale" in user2


def test_build_construct_messages_surfaces_conditional_likelihood_parameters():
    spec = _make_causal_design_dict()
    x1 = next(
        indicator for indicator in spec["measurement"]["indicators"] if indicator["name"] == "x1"
    )
    x1.update(
        measurement_dtype="ordinal",
        aggregation="last",
        ordinal_levels=["low", "medium", "high"],
    )
    state = ConstructBuildState(
        causal_design=spec,
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
    )
    _system, user = build_construct_messages(
        state=state,
        construct="X",
        question="How does X behave?",
        causal_design=spec,
        validation_report={"indicators": {}},
    )
    assert "`obs_ordered_base`" in user
    assert "`obs_ordered_gaps`" in user
    assert "`manifest_mean_x1`" in user
    assert "omit for threshold/categorical" in user

    state.admission = AdmissionState(priors={"obs_ordered_base": _normal(0.0, 1.0)})
    _system, user = build_construct_messages(
        state=state,
        construct="X",
        question="How does X behave?",
        causal_design=spec,
        validation_report={"indicators": {}},
    )
    assert "`obs_ordered_base`" not in user


def test_build_construct_messages_keeps_declared_ordinal_support_with_one_observed_level():
    spec = _make_causal_design_dict()
    state = ConstructBuildState(
        causal_design=spec,
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
    )
    x1 = next(ind for ind in spec["measurement"]["indicators"] if ind["name"] == "x1")
    x1["measurement_dtype"] = "ordinal"
    x1["aggregation"] = "last"
    x1["ordinal_levels"] = ["low", "medium", "high"]

    _system, user = build_construct_messages(
        state=state,
        construct="X",
        question="Does X drive Y?",
        causal_design=spec,
        validation_report={
            "indicators": {
                "x1": {
                    "profile": {
                        "n_obs": 1,
                        "min": 0.0,
                        "max": 0.0,
                    }
                }
            },
        },
    )

    assert "SPARSE LEVEL COVERAGE: only one level is observed" in user
    assert "declared ordinal levels define the likelihood support" in user


def test_build_construct_messages_renders_concern_local_semantic_context():
    spec = _make_causal_design_dict()
    spec["measurement"]["model_clock"] = "6h"
    y1 = next(ind for ind in spec["measurement"]["indicators"] if ind["name"] == "y1")
    y1.update(
        {
            "measurement_dtype": "ordinal",
            "aggregation": "last",
            "observation_window": "12h",
            "ordinal_levels": ["none", "mild", "severe"],
        }
    )
    panel = pl.DataFrame(
        {
            "indicator": ["y1", "y1", "y1", "x1"],
            "value": [0.0, 0.0, 0.0, 999.0],
            "anchor_time": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 10),
                datetime(2025, 1, 1),
            ],
        }
    )
    state = ConstructBuildState(
        causal_design=spec,
        data_for_model=panel,
        order=["X", "Y", "Z"],
        admission=AdmissionState(names=("X",)),
        cursor=1,
    )
    profile = {
        "measurement_dtype": "ordinal",
        "n_obs": 3,
        "mean": 0.0,
        "std": 0.0,
        "variance": 0.0,
        "min": 0.0,
        "q25": 0.0,
        "q50": 0.0,
        "q75": 0.0,
        "max": 0.0,
        "zero_fraction": 1.0,
        "variance_to_mean_ratio": None,
        "is_nonnegative": True,
        "is_unit_interval": True,
        "looks_integer_valued": True,
        "time_coverage_ratio": 0.6,
        "max_gap_ratio": 1.4,
        "dtype_violations": 0,
        "duplicate_pct": 0.0,
        "n_unparseable_timestamps": 0,
        "arithmetic_sequence_detected": False,
    }
    validation_report = {
        "is_valid": False,
        "dataset_issues": [
            {
                "severity": "warning",
                "issue_type": "short_panel",
                "message": "Dataset-level sentinel",
            }
        ],
        "indicators": {
            "y1": {
                "profile": profile,
                "validation": {
                    "issues": [
                        {
                            "severity": "error",
                            "issue_type": "no_variance",
                            "message": "Zero variance (constant value = 0.0)",
                        }
                    ],
                    "checks": {"variance": "error"},
                },
            },
            "x1": {
                "profile": {"n_obs": 1, "mean": 999.0},
                "validation": {
                    "issues": [
                        {
                            "severity": "warning",
                            "issue_type": "sibling",
                            "message": "SIBLING_SENTINEL",
                        }
                    ],
                    "checks": {},
                },
            },
        },
    }

    _system, user = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        causal_design=spec,
        validation_report=validation_report,
    )

    assert "Validation report status: **INVALID**" in user
    assert "[WARNING] short_panel: Dataset-level sentinel" in user
    assert "Model clock / authored default effect interval: `6h`" in user
    assert "Estimation role: **retained latent state**" in user
    assert "Theoretical role: `endogenous`" in user
    assert "Temporal status: `time_varying`" in user
    assert "dtype=`ordinal`" in user
    assert "aggregation=`last`" in user
    assert "effective window=`12h`" in user
    assert "0=none, 1=mild, 2=severe" in user
    assert "n=3; mean=0; sd=0; variance=0" in user
    assert "zero fraction=100.0%" in user
    assert "arithmetic sequence detected=false" in user
    assert "coverage/minimum-required-span=60.0%" in user
    assert "largest-gap/allowed-threshold=1.4x" in user
    assert "Observed ordinal occupancy: 0=none (3), 1=mild (0), 2=severe (0)" in user
    assert "span=9 days; median gap=4.5 days; maximum gap=8 days" in user
    assert "[ERROR] no_variance: Zero variance (constant value = 0.0)" in user
    assert "SIBLING_SENTINEL" not in user
    assert "mean=999" not in user


def test_build_construct_messages_handles_null_empirical_profile():
    spec = _make_causal_design_dict()
    state = ConstructBuildState(
        causal_design=spec,
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
    )

    _system, user = build_construct_messages(
        state=state,
        construct="X",
        question="Does X drive Y?",
        causal_design=spec,
        validation_report={
            "indicators": {"x1": {"profile": None, "validation": {"issues": [], "checks": {}}}}
        },
    )

    assert "Raw empirical profile: unavailable (no numeric observations)" in user


def test_build_construct_messages_renders_incoming_known_input_without_hill_option():
    spec = _make_causal_design_dict()
    spec["estimation"]["state_order"] = ["Y", "Z"]
    spec["estimation"]["known_inputs"] = [
        {
            "construct": "X",
            "source_indicator": "x1",
            "scale": 10.0,
            "missing_policy": "forward_fill",
        }
    ]
    state = ConstructBuildState(
        causal_design=spec,
        data_for_model=pl.DataFrame(),
        order=["Y", "Z"],
    )

    _system, user = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        causal_design=spec,
        validation_report={"indicators": {}},
    )

    assert "`X` — **known transition input**, lagged" in user
    assert "source indicator=`x1`" in user
    assert "scale divisor=10" in user
    assert "missing policy=`forward_fill`" in user
    assert "`beta_X_Y`" in user
    assert "Known-input effects are linear-only" in user
    assert "hill_emax_X_Y" not in user


def test_submit_construct_schema_is_well_formed():
    props = SUBMIT_CONSTRUCT_SCHEMA["properties"]
    assert set(SUBMIT_CONSTRUCT_SCHEMA["required"]) == {"construct", "indicators", "priors"}
    assert SUBMIT_CONSTRUCT_SCHEMA["additionalProperties"] is False
    family_enum = props["indicators"]["items"]["properties"]["family"]["enum"]
    assert "gaussian" in family_enum
    assert "beta" in family_enum
    prior_schema = props["priors"]["additionalProperties"]
    assert set(prior_schema["required"]) == {"distribution", "params", "reasoning"}
    assert prior_schema["additionalProperties"] is False
    assert "TruncatedNormal" in prior_schema["properties"]["distribution"]["enum"]
