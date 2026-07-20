"""Gradual construct-by-construct model-spec flow.

Replaces the parameter-block decomposition with construct admission along the
causal DAG's topological order. Each construct is proposed by the LLM through the
``submit_construct`` tool (its emission choice + priors keyed by canonical
parameter name); the cumulative partial model is compiled and gated by the
**exact** prior-predictive reachability battery
(:mod:`nof1_causal_lab.models.ssm.construct_admission`). A construct that fails a
hard check reopens for revision; a soft failure is a decision (revise, or accept
the consequence via ``accept``). When every construct is admitted, the
accumulated :class:`~nof1_causal_lab.artifacts.statistical_model_spec.StatisticalModelSpec` + priors
are materialized by the Temporal model-spec workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts import (
    DistributionFamily,
    LinkFunction,
    ParameterConstraint,
    ParameterRole,
)
from nof1_causal_lab.artifacts.statistical_model_spec import LikelihoodSpec, ParameterSpec
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.flows.runtime_events import emit_model_spec_admission_event
from nof1_causal_lab.models.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.model_semantics import should_auto_standardize_indicator
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionReport,
    AdmissionState,
    AdmissionTiming,
    ConstructContribution,
    DesignInfo,
    admit_construct,
    recheck_member,
    restrict_causal_design,
    trial_admission_state,
)
from nof1_causal_lab.models.ssm.reachability import CHECK_MODES, CheckResult, stage_outcome
from nof1_causal_lab.utils.causal_design import (
    get_estimation_edges,
    get_estimation_state_order,
    get_indicators,
    get_known_input_source_indicators,
)
from nof1_causal_lab.utils.observation_semantics import get_observation_semantics

from .parameter_surfaces import parameter_is_active_for_statistical_model_spec

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import polars as pl

# Attempts per construct before the build fails (each attempt is one fresh
# agent session that must call submit_construct with a revised proposal).
_MAX_ATTEMPTS_PER_CONSTRUCT = 4

# --------------------------------------------------------------------------- #
# Compiler-authoritative parameter catalog
# --------------------------------------------------------------------------- #

# Structural extensions the base (deterministic) skeleton does not enumerate — a
# self-limiting quartic well and Hill (saturating) edge terms. Both are positive
# dynamics parameters; they are admitted per construct on demand.
_STRUCTURAL_ROLE = (ParameterRole.DYNAMICS_PARAMETER_POSITIVE, ParameterConstraint.POSITIVE)

# Roles whose freeness is fixed by the admission-time model policy (not the
# agent's per-construct choices): pinned initial-state means/SDs under STATIONARY
# initialization, and the `cint_` state-intercept / well-centre, which is free
# only under equilibrium forcing (never requested during gradual admission). For
# these the provisional compile is authoritative, so the catalog trusts its
# binding decision; every other role stays offered and is validated at submit.
_POLICY_PINNED_ROLES = frozenset(
    {
        ParameterRole.INITIAL_STATE_MEAN,
        ParameterRole.INITIAL_STATE_SD,
        ParameterRole.STATE_INTERCEPT,
    }
)


@dataclass(frozen=True)
class ParamCatalog:
    """Compiler-authoritative parameter inventory from the deterministic skeleton.

    Identifiability decisions (e.g. a single-indicator construct has no free
    measurement-noise term) live in the compiler, so the free-parameter set —
    with each parameter's role, constraint, and owning construct — is read from
    ``derive_deterministic_spec`` rather than inferred from name prefixes.
    """

    roles: Mapping[str, tuple[ParameterRole, ParameterConstraint]]
    by_construct: Mapping[str, tuple[str, ...]]
    global_params: frozenset[str]
    site_names: Mapping[str, str]
    metadata: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_causal_design(cls, causal_design: dict) -> ParamCatalog:
        from .skeleton import derive_deterministic_spec

        skeleton = derive_deterministic_spec(causal_design)
        roles: dict[str, tuple[ParameterRole, ParameterConstraint]] = {}
        by_construct: dict[str, list[str]] = {}
        global_params: set[str] = set()
        site_names: dict[str, str] = {}
        metadata: dict[str, Mapping[str, Any]] = {}
        for param in (*skeleton.parameters, *skeleton.loading_params):
            role = ParameterRole(param["role"])
            # Initial-state means/SDs and the `cint_` well-centre are pinned by
            # the admission-time model *policy* (STATIONARY init; no equilibrium
            # forcing) — not by the agent's per-construct choices — so the
            # provisional compile authoritatively decides their freeness. Drop
            # them when the compiler did not bind them (no `compiled_site_kind`
            # from `_enrich_parameter_with_binding`), which is what made the
            # agent author priors the compiler then rejects as "not free"; keep
            # the ones it did bind (e.g. a time-invariant construct's free
            # initial state). Family-conditional surfaces (measurement-noise SD,
            # observation intercept) depend on the agent's emission choice —
            # unknown here — so they stay offered and the submit-time compile
            # validates them against the locked family.
            if role in _POLICY_PINNED_ROLES and "compiled_site_kind" not in param:
                is_conditional_static_mean = (
                    role == ParameterRole.INITIAL_STATE_MEAN
                    and param.get("temporal_status") == "time_invariant"
                )
                if not is_conditional_static_mean:
                    continue
            name = param["name"]
            roles[name] = (role, ParameterConstraint(param["constraint"]))
            metadata[name] = dict(param)
            if site_name := param.get("compiled_site_name"):
                site_names[name] = str(site_name)
            owner = param.get("construct")
            if owner is not None:
                by_construct.setdefault(owner, []).append(name)
            else:
                global_params.add(name)
                site_names.setdefault(name, name)
                for construct_name in param.get("construct_names") or ():
                    by_construct.setdefault(str(construct_name), []).append(name)
        return cls(
            roles=roles,
            by_construct={c: tuple(v) for c, v in by_construct.items()},
            global_params=frozenset(global_params),
            site_names=site_names,
            metadata=metadata,
        )

    def structural_names(self, construct: str, parents: Sequence[str]) -> set[str]:
        names = {f"self_limit_{construct}"}
        for parent in parents:
            names.update(
                {
                    f"hill_emax_{parent}_{construct}",
                    f"hill_ec50_{parent}_{construct}",
                    f"hill_n_{parent}_{construct}",
                }
            )
        return names

    def prior_names_for(
        self,
        construct: str,
        *,
        admitted_prior_names: Collection[str] = (),
    ) -> set[str]:
        names = set(self.by_construct.get(construct, ()))
        names -= self.global_params & set(admitted_prior_names)
        return names

    def allowed_for(
        self,
        construct: str,
        parents: Sequence[str],
        *,
        likelihood_by_variable: Mapping[str, Mapping[str, Any]] | None = None,
        admitted_prior_names: Collection[str] = (),
    ) -> set[str]:
        structural = self.structural_names(construct, parents)
        names = self.prior_names_for(
            construct,
            admitted_prior_names=admitted_prior_names,
        )
        if likelihood_by_variable is None:
            return names | structural
        active = {
            name
            for name in names
            if parameter_is_active_for_statistical_model_spec(
                dict(self.metadata[name]),
                {key: dict(value) for key, value in likelihood_by_variable.items()},
                initialization_policy="stationary",
                observation_intercept_policy="free",
                equilibrium_forcing=False,
            )
        }
        return active | structural

    def role_for(self, name: str) -> tuple[ParameterRole, ParameterConstraint]:
        return self.roles.get(name, _STRUCTURAL_ROLE)

    def site_for(self, name: str) -> str | None:
        return self.site_names.get(name)

    def metadata_for(self, name: str) -> Mapping[str, Any]:
        return self.metadata[name]


def construct_parents(causal_design: dict, construct: str) -> list[str]:
    """Direct causal parents of ``construct`` (edge sources into it)."""
    parents: list[str] = []
    for edge in get_estimation_edges(causal_design):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if effect == construct and cause is not None and str(cause) not in parents:
            parents.append(str(cause))
    return parents


def _locked_likelihood_by_variable(
    causal_design: dict,
    indicators: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve submitted emissions into the compiler's likelihood-activation surface."""
    indicator_lookup = {indicator["name"]: indicator for indicator in get_indicators(causal_design)}
    locked: dict[str, dict[str, Any]] = {}
    for submitted in indicators:
        variable = str(submitted["variable"])
        indicator = indicator_lookup[variable]
        distribution = DistributionFamily(submitted["family"])
        link = LinkFunction(submitted["link"])
        semantics = get_observation_semantics(indicator)
        locked[variable] = {
            "variable": variable,
            "distribution": distribution.value,
            "link": link.value,
            "construct_name": indicator.get("construct_name"),
            "support_kind": semantics.support_kind.value,
            "summary_operator": semantics.summary_operator.value,
            "standardized": should_auto_standardize_indicator(
                distribution,
                link,
                semantics.support_kind.value,
                semantics.summary_operator.value,
            ),
        }
    return locked


def _acceptance_map(
    decisions: Sequence[Mapping[str, Any]] | None,
) -> dict[tuple[str, str], str]:
    """Validate structured, target-scoped soft-check acceptance decisions."""
    accepted: dict[tuple[str, str], str] = {}
    for decision in decisions or ():
        check = str(decision.get("check", "")).strip()
        target = str(decision.get("target", "")).strip()
        rationale = str(decision.get("rationale", "")).strip()
        if not check or not target or not rationale:
            raise ValueError("Every acceptance requires non-empty check, target, and rationale.")
        key = (check, target)
        if key in accepted:
            raise ValueError(f"Duplicate acceptance for {check} [{target}].")
        accepted[key] = rationale
    return accepted


def deferred_closing_edge_params(
    causal_design: dict, construct: str, admitted: Collection[str]
) -> set[str]:
    """Params for cycle-closing edges ``construct -> already-admitted effect``.

    ``restrict_causal_design`` keeps an edge only when both endpoints are kept,
    so a feedback edge out of ``construct`` into an earlier-admitted member
    first materializes during THIS construct's admission. Its weight prior
    must be authorable here: the effect construct was admitted without the
    edge, and authoring the prior on its turn would have named a site absent
    from that turn's restricted model.
    """
    names: set[str] = set()
    for edge in get_estimation_edges(causal_design):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if cause == construct and effect in admitted:
            names.add(f"beta_{construct}_{effect}")
            names.update(
                {
                    f"hill_emax_{construct}_{effect}",
                    f"hill_ec50_{construct}_{effect}",
                    f"hill_n_{construct}_{effect}",
                }
            )
    return names


def _closing_edge_effects(
    causal_design: dict, construct: str, prior_admitted: Collection[str]
) -> list[str]:
    """Already-admitted effect(s) of feedback edges out of ``construct``.

    These are the cycle members whose latent dynamics change when admitting ``construct``
    closes the loop — so they warrant a coupled recheck against the closed-loop model.
    """
    effects: list[str] = []
    for edge in get_estimation_edges(causal_design):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if cause == construct and effect in prior_admitted and str(effect) not in effects:
            effects.append(str(effect))
    return effects


def _closed_loop_target(
    member: ConstructContribution, causal_design: dict, priors: Mapping[str, Any]
) -> ConstructContribution:
    """``member``'s contribution with its edge set recomputed on the closed loop.

    C4b/C4c on the member must now see the just-closed feedback edge, whose ``beta_*`` /
    ``hill_*`` prior is already in ``priors`` (authored during the loop-closing submission).
    """
    name = member.name
    parents = construct_parents(causal_design, name)
    latent_parents = set(get_estimation_state_order(causal_design))
    edge_parents = tuple(
        p for p in parents if f"beta_{p}_{name}" in priors or f"hill_emax_{p}_{name}" in priors
    )
    hill_parents = tuple(
        p for p in parents if p in latent_parents and f"hill_emax_{p}_{name}" in priors
    )
    return replace(member, edge_parents=edge_parents, hill_parents=hill_parents)


# --------------------------------------------------------------------------- #
# Tool payload → ConstructContribution
# --------------------------------------------------------------------------- #


def contribution_from_payload(
    causal_design: dict, payload: Mapping[str, Any], catalog: ParamCatalog
) -> ConstructContribution:
    """Parse a ``submit_construct`` payload into a canonical construct contribution.

    Edge/Hill structure is *implied by the authored priors*: a ``beta_<p>_<c>``
    prior declares a linear edge from parent ``p``; a ``hill_emax_<p>_<c>`` prior
    declares a saturating (Hill) edge. The self-limiting quartic is implied by a
    ``self_limit_<c>`` prior. Parents come from the causal DAG, so the compound
    name is split unambiguously. Roles/constraints come from the compiler-
    authoritative ``catalog`` (skeleton), not from the parameter name.
    """
    name = str(payload["construct"])
    likelihoods = tuple(
        LikelihoodSpec(
            variable=str(ind["variable"]),
            distribution=DistributionFamily(ind["family"]),
            link=LinkFunction(ind["link"]),
            reasoning=str(ind.get("reasoning", "")),
        )
        for ind in payload.get("indicators", ())
    )
    raw_priors = payload.get("priors", {})
    if not isinstance(raw_priors, dict) or not all(
        isinstance(v, dict) for v in raw_priors.values()
    ):
        raise ValueError(
            "`priors` must be an object mapping parameter names to distribution objects, "
            'e.g. {"lambda_x": {"distribution": "Normal", "params": {"mu": 0, "sigma": 1}}} '
            "— got a non-object prior value."
        )
    priors = {str(k): dict(v) for k, v in raw_priors.items()}
    parameters = tuple(
        ParameterSpec(
            name=pn,
            role=catalog.role_for(pn)[0],
            constraint=catalog.role_for(pn)[1],
            description=f"authored prior for {pn}",
        )
        for pn in priors
    )
    parents = construct_parents(causal_design, name)
    latent_parents = set(get_estimation_state_order(causal_design))
    edge_parents = tuple(
        p for p in parents if f"beta_{p}_{name}" in priors or f"hill_emax_{p}_{name}" in priors
    )
    hill_parents = tuple(
        p for p in parents if p in latent_parents and f"hill_emax_{p}_{name}" in priors
    )
    return ConstructContribution(
        name=name,
        likelihoods=likelihoods,
        parameters=parameters,
        priors=priors,
        edge_parents=edge_parents,
        hill_parents=hill_parents,
    )


# --------------------------------------------------------------------------- #
# Design derivation from real longitudinal data (fit-consistent)
# --------------------------------------------------------------------------- #


def build_design_info(
    state: AdmissionState,
    contribution: ConstructContribution,
    causal_design: dict,
    data_for_model: pl.DataFrame,
    *,
    n_draws: int,
    seed: int,
) -> DesignInfo:
    """Reachability design for admitting ``contribution`` onto ``state`` (the trial model)."""
    return _design_for_state(
        trial_admission_state(state, contribution),
        causal_design,
        data_for_model,
        n_draws=n_draws,
        seed=seed,
    )


def _design_for_state(
    model_state: AdmissionState,
    causal_design: dict,
    data_for_model: pl.DataFrame,
    *,
    n_draws: int,
    seed: int,
) -> DesignInfo:
    """Derive the reachability design against the compiled ``model_state``.

    Uses the canonical ``prepare_model_runtime`` so the sampling grid, the
    per-indicator observation indices, and the observed values all live in the
    same time + observation space the fit uses — including support-aware handling
    and the emission-space scaling the raw data does not carry. Both admission
    (against a trial state) and the coupled recheck (against the closed-loop state)
    build their design here.
    """
    import polars as pl

    from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
    from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime

    restricted = restrict_causal_design(causal_design, set(model_state.names))
    compiled = compile_ssm_artifact(
        model_state.statistical_model_spec(), dict(model_state.priors), causal_design=restricted
    )

    indicator_names = [lik.variable for lik in model_state.likelihoods]
    indicator_names.extend(sorted(get_known_input_source_indicators(restricted)))
    trial_data = data_for_model.filter(pl.col("indicator").is_in(indicator_names))
    runtime = prepare_model_runtime(trial_data, compiled_ssm=compiled)

    times = np.asarray(runtime.times, dtype=float)
    observations = np.asarray(runtime.observations, dtype=float)
    manifest_names = list(runtime.manifest_names)

    obs_index_by_indicator: dict[str, np.ndarray] = {}
    values_by_indicator: dict[str, np.ndarray] = {}
    for i, manifest in enumerate(manifest_names):
        present = np.where(np.isfinite(observations[:, i]))[0]
        obs_index_by_indicator[manifest] = present
        values_by_indicator[manifest] = observations[present, i]

    return DesignInfo(
        t_grid=jnp.asarray(times),
        obs_index_by_indicator=obs_index_by_indicator,
        values_by_indicator=values_by_indicator,
        n_draws=n_draws,
        seed=seed,
        observation_support=runtime.observation_support,
        transition_inputs=runtime.transition_inputs,
    )


# --------------------------------------------------------------------------- #
# Feedback rendering
# --------------------------------------------------------------------------- #


def render_admission_feedback(report: AdmissionReport) -> str:
    """Render a construct's battery results + verdict as LLM-facing feedback."""
    lines = [f"## Reachability report for `{report.name}`", "", report.outcome, ""]
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"- [{mark}] {r.check} [{r.target}]: {r.value} (target {r.band})")
        if not r.passed:
            if r.note:
                lines.append(f"    {r.note}")
            for d in r.diagnosis:
                lines.append(f"    · {d}")
    if report.annotations:
        lines.append("")
        lines.append("Accepted consequences:")
        lines.extend(f"- {a}" for a in report.annotations)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Admission telemetry payloads (live-view contract)
# --------------------------------------------------------------------------- #
#
# These translate the admission dataclasses into the JSON the web construct-
# admission view reduces. The Temporal activities emit them when a ``workspace_id``
# is present; pure state tests use ``workspace_id=None`` and have no side effects.


def _admission_plan_payload(causal_design: dict, order: Sequence[str]) -> dict[str, Any]:
    """The static admission plan: constructs in admission order + the DAG edges among them."""
    order_set = set(order)
    edges: list[dict[str, str]] = []
    for edge in get_estimation_edges(causal_design):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if cause in order_set and effect in order_set:
            edges.append({"cause": str(cause), "effect": str(effect)})
    constructs = [
        {"name": name, "parents": construct_parents(causal_design, name)} for name in order
    ]
    return {"constructs": constructs, "edges": edges, "max_attempts": _MAX_ATTEMPTS_PER_CONSTRUCT}


def _admission_parameters_payload(contribution: ConstructContribution) -> list[dict[str, Any]]:
    """Authored priors of a submission as ``{name, distribution, params}`` for the UI table."""
    params: list[dict[str, Any]] = []
    for name, dist in contribution.priors.items():
        raw = dist.get("params", {}) if isinstance(dist, dict) else {}
        params.append(
            {
                "name": name,
                "distribution": str(dist.get("distribution", "")) if isinstance(dist, dict) else "",
                "params": {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))},
            }
        )
    return params


def _check_result_payload(result: CheckResult) -> dict[str, Any]:
    """A reachability CheckResult in the admission-view contract (mode from the severity table)."""
    return {
        "check": result.check,
        "target": result.target,
        "value": result.value,
        "band": result.band,
        "passed": result.passed,
        "note": result.note,
        "diagnosis": list(result.diagnosis),
        "mode": CHECK_MODES[result.check],
    }


def _timing_payload(timing: AdmissionTiming) -> dict[str, Any]:
    return {
        "phase": timing.phase,
        "label": timing.label,
        "duration_ms": timing.duration_ms,
        "checks": list(timing.checks),
    }


def _admission_report_payload(
    report: AdmissionReport,
    contribution: ConstructContribution,
    attempt: int,
    coupled_recheck: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One attempt's battery outcome + authored priors, in the admission-view contract.

    ``coupled_recheck`` (present only when admitting this construct closed a feedback loop)
    carries the closed-loop re-evaluation of the already-admitted cycle member(s).
    """
    payload: dict[str, Any] = {
        "name": report.name,
        "attempt": attempt,
        "outcome": report.outcome,
        "admitted": report.admitted,
        "annotations": list(report.annotations),
        "results": [_check_result_payload(r) for r in report.results],
        "timings": [_timing_payload(timing) for timing in report.timings],
        "parameters": _admission_parameters_payload(contribution),
    }
    if coupled_recheck is not None:
        payload["coupled_recheck"] = coupled_recheck
        payload["timings"].append(
            {
                "phase": "coupled_recheck",
                "label": "Coupled subsystem recheck",
                "duration_ms": sum(timing["duration_ms"] for timing in coupled_recheck["timings"]),
                "checks": [],
            }
        )
    return payload


# --------------------------------------------------------------------------- #
# Construct-build session state + tool
# --------------------------------------------------------------------------- #


@dataclass
class ConstructBuildState:
    """Mutable state driving the construct-by-construct admission loop."""

    causal_design: dict
    data_for_model: pl.DataFrame
    order: list[str]
    n_draws: int = 200
    seed: int = 0
    # Live-telemetry seam (mirrors stage 2): production threads the workspace id so the
    # construct-admission view can stream; the batch/test path leaves it None and emits nothing.
    workspace_id: str | None = None
    attempt: int = 0
    catalog: ParamCatalog | None = None
    admission: AdmissionState = field(default_factory=AdmissionState)
    cursor: int = 0
    search_queries: dict[str, str] = field(default_factory=dict)
    search_cache: dict[str, str] = field(default_factory=dict)
    last_report: AdmissionReport | None = None
    last_coupled_results: tuple[CheckResult, ...] = ()
    last_tool_feedback: str | None = None
    submission_made: bool = False
    # Kept so a loop-closing admission can re-run the battery on already-admitted members.
    admitted_contributions: dict[str, ConstructContribution] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.catalog is None:
            self.catalog = ParamCatalog.from_causal_design(self.causal_design)

    @property
    def current_construct(self) -> str | None:
        return self.order[self.cursor] if self.cursor < len(self.order) else None

    def submit_construct(
        self,
        *,
        construct: str,
        indicators: Sequence[Mapping[str, Any]],
        priors: Mapping[str, Any],
        accept: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        self.submission_made = True
        expected = self.current_construct
        if expected is None:
            return "All constructs are already admitted; no further submission is needed."
        if construct != expected:
            return (
                f"Out-of-order submission: the active construct is `{expected}`, not "
                f"`{construct}`. Submit `{expected}` first."
            )
        assert self.catalog is not None  # set in __post_init__
        parents = construct_parents(self.causal_design, construct)
        saturating_parents = [parent for parent in parents if parent in self.admission.names]
        closing = deferred_closing_edge_params(
            self.causal_design, construct, set(self.admission.names)
        )
        locked_likelihoods = _locked_likelihood_by_variable(self.causal_design, indicators)
        allowed = (
            self.catalog.allowed_for(
                construct,
                saturating_parents,
                likelihood_by_variable=locked_likelihoods,
                admitted_prior_names=self.admission.priors,
            )
            | closing
        )
        unknown = [name for name in priors if name not in allowed]
        if unknown:
            return (
                f"These parameters are not free for `{construct}` and cannot take a prior: "
                f"{', '.join(sorted(unknown))}. Author priors only for: "
                f"{', '.join(sorted(allowed))}."
            )
        missing_closing = [
            beta
            for beta in sorted(n for n in closing if n.startswith("beta_"))
            if beta not in priors and beta.replace("beta_", "hill_emax_", 1) not in priors
        ]
        if missing_closing:
            return (
                "Missing cycle-closing edge prior(s): "
                + ", ".join(f"`{n}`" for n in missing_closing)
                + ". This construct closes a feedback loop: the closing edge materializes "
                "in the restricted model NOW, so its weight must be authored in this same "
                "submission (as the `beta_...` prior named above, or its `hill_*` variants) "
                "— otherwise the compiler rejects the unbound edge site."
            )
        pooled_families: dict[str, set[str]] = {}
        for name, prior in {**self.admission.priors, **priors}.items():
            site_name = self.catalog.site_for(name)
            distribution = prior.get("distribution")
            if site_name is not None and isinstance(distribution, str):
                pooled_families.setdefault(site_name, set()).add(distribution)
        mixed_sites = {
            site_name: families
            for site_name, families in pooled_families.items()
            if len(families) > 1
        }
        if mixed_sites:
            details = "; ".join(
                f"`{site_name}` has {', '.join(sorted(families))}"
                for site_name, families in sorted(mixed_sites.items())
            )
            return (
                "Prior family mismatch within pooled compiler sample site(s): "
                f"{details}. Match the distribution family already authored for every "
                "parameter sharing that site."
            )
        payload = {"construct": construct, "indicators": list(indicators), "priors": dict(priors)}
        contribution = contribution_from_payload(self.causal_design, payload, self.catalog)
        if self.workspace_id:
            emit_model_spec_admission_event(
                self.workspace_id,
                "construct_checking",
                {"construct": construct, "attempt": self.attempt},
            )
        design_started = perf_counter_ns()
        try:
            design = build_design_info(
                self.admission,
                contribution,
                self.causal_design,
                self.data_for_model,
                n_draws=self.n_draws,
                seed=self.seed,
            )
        except AggregatedCompileError as exc:
            # Compile-time identification/translation errors are revision-shaped:
            # return them as tool feedback so the admission loop can repair the
            # submission instead of crashing the activity.
            return str(exc)
        design_timing = AdmissionTiming(
            phase="design_preparation",
            label="Design preparation",
            duration_ms=(perf_counter_ns() - design_started) / 1_000_000,
        )
        prior_admitted = set(self.admission.names)
        try:
            accepted = _acceptance_map(accept)
        except ValueError as exc:
            return str(exc)
        new_state, report = admit_construct(
            self.admission, contribution, self.causal_design, design, accepted=accepted
        )
        report = replace(report, timings=(design_timing, *report.timings))
        coupled_recheck: dict[str, Any] | None = None
        coupled_results: list[CheckResult] = []
        if report.admitted:
            coupled_results, coupled_recheck = self._coupled_recheck(
                construct, prior_admitted, new_state
            )
            if coupled_results:
                outcome, annotations = stage_outcome([*report.results, *coupled_results], accepted)
                admitted = outcome.startswith("ADMITTED")
                report = replace(
                    report,
                    outcome=outcome,
                    annotations=annotations,
                    admitted=admitted,
                )
                new_state = replace(
                    new_state,
                    annotations=(*self.admission.annotations, *annotations),
                )
        failed_soft = {
            (result.check, result.target)
            for result in (*report.results, *coupled_results)
            if not result.passed and CHECK_MODES[result.check] == "soft"
        }
        invalid_acceptances = sorted(set(accepted) - failed_soft)
        if invalid_acceptances:
            refs = ", ".join(f"{check} [{target}]" for check, target in invalid_acceptances)
            return f"Acceptance references must name current failing soft checks exactly: {refs}."
        self.last_report = report
        self.last_coupled_results = tuple(coupled_results)
        if report.admitted:
            self.admission = new_state
            self.cursor += 1
            self.admitted_contributions[construct] = contribution
        if self.workspace_id:
            emit_model_spec_admission_event(
                self.workspace_id,
                "construct_report",
                _admission_report_payload(report, contribution, self.attempt, coupled_recheck),
            )
        feedback = render_admission_feedback(report)
        if coupled_results:
            lines = [feedback, "", "Coupled feedback-component checks:"]
            for result in coupled_results:
                mark = "PASS" if result.passed else "FAIL"
                lines.append(
                    f"- [{mark}] {result.check} [{result.target}]: {result.value} "
                    f"(target {result.band})"
                )
                if not result.passed:
                    lines.extend(f"    · {diagnosis}" for diagnosis in result.diagnosis)
            feedback = "\n".join(lines)
        return feedback

    def _coupled_recheck(
        self,
        construct: str,
        prior_admitted: Collection[str],
        tentative_state: AdmissionState,
    ) -> tuple[list[CheckResult], dict[str, Any] | None]:
        """Gate loop closure by rechecking every already-admitted affected member."""
        members = [
            m
            for m in _closing_edge_effects(self.causal_design, construct, prior_admitted)
            if m in self.admitted_contributions
        ]
        if not members:
            return [], None
        design_started = perf_counter_ns()
        design = _design_for_state(
            tentative_state,
            self.causal_design,
            self.data_for_model,
            n_draws=self.n_draws,
            seed=self.seed,
        )
        timings = [
            AdmissionTiming(
                phase="design_preparation",
                label="Design preparation",
                duration_ms=(perf_counter_ns() - design_started) / 1_000_000,
            )
        ]
        raw_results: list[CheckResult] = []
        for member in members:
            target = _closed_loop_target(
                self.admitted_contributions[member], self.causal_design, tentative_state.priors
            )
            member_results, member_timings = recheck_member(
                tentative_state, target, self.causal_design, design
            )
            raw_results.extend(member_results)
            timings.extend(
                replace(
                    timing,
                    phase=f"recheck:{member}:{timing.phase}",
                    label=f"{member}: {timing.label}",
                )
                for timing in member_timings
            )
        if not raw_results:
            return [], None
        return raw_results, {
            "constructs": [*members, construct],
            "closing_edges": [f"{construct}->{m}" for m in members],
            "results": [_check_result_payload(result) for result in raw_results],
            "timings": [_timing_payload(timing) for timing in timings],
        }


SUBMIT_CONSTRUCT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "construct": {
            "type": "string",
            "description": "Name of the construct being admitted (must be the active one).",
        },
        "indicators": {
            "type": "array",
            "description": "Emission choice for each indicator of this construct.",
            "items": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string"},
                    "family": {
                        "type": "string",
                        "enum": [e.value for e in DistributionFamily],
                        "description": "Observation distribution family.",
                    },
                    "link": {
                        "type": "string",
                        "enum": [e.value for e in LinkFunction],
                        "description": "Link mapping the latent linear predictor to the mean.",
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["variable", "family", "link"],
                "additionalProperties": False,
            },
        },
        "priors": {
            "type": "object",
            "description": (
                "Prior proposals keyed by a canonical parameter name listed in the active "
                "construct prompt. A Hill edge is declared by authoring hill_* priors; a "
                "self-limiting well by self_limit_<c>. Conditional likelihood parameters "
                "must be omitted when the submitted family/link does not activate them. "
                "Every value must use the canonical {distribution, params, reasoning} shape."
            ),
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "distribution": {
                        "type": "string",
                        "enum": [
                            family.value
                            for family in PriorDistributionFamily
                            if family != PriorDistributionFamily.DELTA
                        ],
                    },
                    "params": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "reasoning": {"type": "string"},
                    "reference_interval_days": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                },
                "required": ["distribution", "params", "reasoning"],
                "additionalProperties": False,
            },
        },
        "accept": {
            "type": "array",
            "description": (
                "Optional target-scoped decisions accepting current soft-check consequences."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "check": {"type": "string"},
                    "target": {"type": "string"},
                    "rationale": {"type": "string", "minLength": 1},
                },
                "required": ["check", "target", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["construct", "indicators", "priors"],
    "additionalProperties": False,
}


__all__ = [
    "ConstructBuildState",
    "build_design_info",
    "contribution_from_payload",
    "render_admission_feedback",
]
