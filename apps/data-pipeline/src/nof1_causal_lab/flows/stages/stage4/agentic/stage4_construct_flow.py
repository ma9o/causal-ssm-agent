"""Gradual construct-by-construct Stage 4 flow.

Replaces the parameter-block decomposition with construct admission along the
causal DAG's topological order. Each construct is proposed by the LLM through the
``submit_construct`` tool (its emission choice + priors keyed by canonical
parameter name); the cumulative partial model is compiled and gated by the
**exact** prior-predictive reachability battery
(:mod:`nof1_causal_lab.models.ssm.construct_admission`). A construct that fails a
hard check reopens for revision; a soft failure is a decision (revise, or accept
the consequence via ``accept``). When every construct is admitted, the
accumulated :class:`~nof1_causal_lab.artifacts.model_spec.ModelSpec` + priors are
returned as a :class:`Stage4Result`, which the existing materialization turns into
the ``compiled_ssm`` artifact unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts import (
    DistributionFamily,
    LinkFunction,
    ParameterConstraint,
    ParameterRole,
)
from nof1_causal_lab.artifacts.model_spec import LikelihoodSpec, ParameterSpec
from nof1_causal_lab.flows.runtime_events import emit_stage4_admission_event
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionReport,
    AdmissionState,
    ConstructContribution,
    DesignInfo,
    admit_construct,
    build_construct_order,
    recheck_member,
    restrict_causal_spec,
    trial_admission_state,
)
from nof1_causal_lab.models.ssm.reachability import CHECK_MODES, CheckResult
from nof1_causal_lab.utils.causal_spec import get_estimation_edges

from .stage4_types import Stage4Result

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import polars as pl

    from nof1_causal_lab.utils.agent_session import StageSessionFactory

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

    @classmethod
    def from_causal_spec(cls, causal_spec: dict) -> ParamCatalog:
        from .stage4_skeleton import derive_deterministic_spec

        skeleton = derive_deterministic_spec(causal_spec)
        roles: dict[str, tuple[ParameterRole, ParameterConstraint]] = {}
        by_construct: dict[str, list[str]] = {}
        global_params: set[str] = set()
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
                continue
            name = param["name"]
            roles[name] = (role, ParameterConstraint(param["constraint"]))
            owner = param.get("construct")
            if owner is not None:
                by_construct.setdefault(owner, []).append(name)
            else:
                global_params.add(name)
        return cls(
            roles=roles,
            by_construct={c: tuple(v) for c, v in by_construct.items()},
            global_params=frozenset(global_params),
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

    def allowed_for(self, construct: str, parents: Sequence[str]) -> set[str]:
        return (
            set(self.by_construct.get(construct, ()))
            | self.structural_names(construct, parents)
            | set(self.global_params)
        )

    def role_for(self, name: str) -> tuple[ParameterRole, ParameterConstraint]:
        return self.roles.get(name, _STRUCTURAL_ROLE)


def construct_parents(causal_spec: dict, construct: str) -> list[str]:
    """Direct causal parents of ``construct`` (edge sources into it)."""
    parents: list[str] = []
    for edge in get_estimation_edges(causal_spec):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if effect == construct and cause is not None and str(cause) not in parents:
            parents.append(str(cause))
    return parents


def deferred_closing_edge_params(
    causal_spec: dict, construct: str, admitted: Collection[str]
) -> set[str]:
    """Params for cycle-closing edges ``construct -> already-admitted effect``.

    ``restrict_causal_spec`` keeps an edge only when both endpoints are kept,
    so a feedback edge out of ``construct`` into an earlier-admitted member
    first materializes during THIS construct's admission. Its weight prior
    must be authorable here: the effect construct was admitted without the
    edge, and authoring the prior on its turn would have named a site absent
    from that turn's restricted model.
    """
    names: set[str] = set()
    for edge in get_estimation_edges(causal_spec):
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
    causal_spec: dict, construct: str, prior_admitted: Collection[str]
) -> list[str]:
    """Already-admitted effect(s) of feedback edges out of ``construct``.

    These are the cycle members whose latent dynamics change when admitting ``construct``
    closes the loop — so they warrant a coupled recheck against the closed-loop model.
    """
    effects: list[str] = []
    for edge in get_estimation_edges(causal_spec):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if cause == construct and effect in prior_admitted and str(effect) not in effects:
            effects.append(str(effect))
    return effects


def _closed_loop_target(
    member: ConstructContribution, causal_spec: dict, priors: Mapping[str, Any]
) -> ConstructContribution:
    """``member``'s contribution with its edge set recomputed on the closed loop.

    C4b/C4c on the member must now see the just-closed feedback edge, whose ``beta_*`` /
    ``hill_*`` prior is already in ``priors`` (authored during the loop-closing submission).
    """
    name = member.name
    parents = construct_parents(causal_spec, name)
    edge_parents = tuple(
        p for p in parents if f"beta_{p}_{name}" in priors or f"hill_emax_{p}_{name}" in priors
    )
    hill_parents = tuple(p for p in parents if f"hill_emax_{p}_{name}" in priors)
    return replace(member, edge_parents=edge_parents, hill_parents=hill_parents)


# --------------------------------------------------------------------------- #
# Tool payload → ConstructContribution
# --------------------------------------------------------------------------- #


def contribution_from_payload(
    causal_spec: dict, payload: Mapping[str, Any], catalog: ParamCatalog
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
    parents = construct_parents(causal_spec, name)
    edge_parents = tuple(
        p for p in parents if f"beta_{p}_{name}" in priors or f"hill_emax_{p}_{name}" in priors
    )
    hill_parents = tuple(p for p in parents if f"hill_emax_{p}_{name}" in priors)
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
    causal_spec: dict,
    data_for_model: pl.DataFrame,
    *,
    n_draws: int,
    seed: int,
) -> DesignInfo:
    """Reachability design for admitting ``contribution`` onto ``state`` (the trial model)."""
    return _design_for_state(
        trial_admission_state(state, contribution),
        causal_spec,
        data_for_model,
        n_draws=n_draws,
        seed=seed,
    )


def _design_for_state(
    model_state: AdmissionState,
    causal_spec: dict,
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

    restricted = restrict_causal_spec(causal_spec, set(model_state.names))
    compiled = compile_ssm_artifact(
        model_state.model_spec(), dict(model_state.priors), causal_spec=restricted
    )

    indicator_names = [lik.variable for lik in model_state.likelihoods]
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

    diffs = np.diff(times)
    cadence = float(np.median(diffs)) if diffs.size else 1.0
    span = float(np.ptp(times)) if times.size else 1.0
    return DesignInfo(
        t_grid=jnp.asarray(times),
        obs_index_by_indicator=obs_index_by_indicator,
        values_by_indicator=values_by_indicator,
        cadence=cadence,
        span=span,
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
        lines.append(f"- [{mark}] {r.check}: {r.value} (target {r.band})")
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
# admission view reduces. Emission is threaded through the flow (see
# ``run_stage4_construct_build``) only when a ``workspace_id`` is present, so the
# batch/test path (``workspace_id=None``) runs without any telemetry side effect.


def _admission_plan_payload(causal_spec: dict, order: Sequence[str]) -> dict[str, Any]:
    """The static admission plan: constructs in admission order + the DAG edges among them."""
    order_set = set(order)
    edges: list[dict[str, str]] = []
    for edge in get_estimation_edges(causal_spec):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if cause in order_set and effect in order_set:
            edges.append({"cause": str(cause), "effect": str(effect)})
    constructs = [{"name": name, "parents": construct_parents(causal_spec, name)} for name in order]
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
        "parameters": _admission_parameters_payload(contribution),
    }
    if coupled_recheck is not None:
        payload["coupled_recheck"] = coupled_recheck
    return payload


# --------------------------------------------------------------------------- #
# Construct-build session state + tool
# --------------------------------------------------------------------------- #


@dataclass
class ConstructBuildState:
    """Mutable state driving the construct-by-construct admission loop."""

    causal_spec: dict
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
    submission_made: bool = False
    # Kept so a loop-closing admission can re-run the battery on already-admitted members.
    admitted_contributions: dict[str, ConstructContribution] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.catalog is None:
            self.catalog = ParamCatalog.from_causal_spec(self.causal_spec)

    @property
    def current_construct(self) -> str | None:
        return self.order[self.cursor] if self.cursor < len(self.order) else None

    def submit_construct(
        self,
        *,
        construct: str,
        indicators: Sequence[Mapping[str, Any]],
        priors: Mapping[str, Any],
        accept: Mapping[str, str] | None = None,
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
        parents = construct_parents(self.causal_spec, construct)
        closing = deferred_closing_edge_params(
            self.causal_spec, construct, set(self.admission.names)
        )
        allowed = self.catalog.allowed_for(construct, parents) | closing
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
        payload = {"construct": construct, "indicators": list(indicators), "priors": dict(priors)}
        contribution = contribution_from_payload(self.causal_spec, payload, self.catalog)
        if self.workspace_id:
            emit_stage4_admission_event(
                self.workspace_id,
                "construct_checking",
                {"construct": construct, "attempt": self.attempt},
            )
        design = build_design_info(
            self.admission,
            contribution,
            self.causal_spec,
            self.data_for_model,
            n_draws=self.n_draws,
            seed=self.seed,
        )
        prior_admitted = set(self.admission.names)
        new_state, report = admit_construct(
            self.admission, contribution, self.causal_spec, design, accepted=dict(accept or {})
        )
        self.last_report = report
        coupled_recheck: dict[str, Any] | None = None
        if report.admitted:
            self.admission = new_state
            self.cursor += 1
            self.admitted_contributions[construct] = contribution
            # Informational-only, and currently consumed solely via telemetry, so skip its
            # extra closed-loop compile+sim when nobody is streaming (batch/tests).
            if self.workspace_id:
                coupled_recheck = self._coupled_recheck(construct, prior_admitted)
        if self.workspace_id:
            emit_stage4_admission_event(
                self.workspace_id,
                "construct_report",
                _admission_report_payload(report, contribution, self.attempt, coupled_recheck),
            )
        return render_admission_feedback(report)

    def _coupled_recheck(
        self, construct: str, prior_admitted: Collection[str]
    ) -> dict[str, Any] | None:
        """Re-run the battery on already-admitted member(s) if admitting ``construct`` closed a loop.

        Informational: the closed-loop re-evaluation is surfaced on the report event but does not
        gate the admission (the loop stays closed regardless of the recheck outcome).

        The right version, long-term, is to GATE rather than inform: a hard closed-loop failure
        means closing the feedback edge destabilized an already-admitted member, so the correct
        behavior is to *invalidate* that member's admission — re-open the closing construct for
        revision of its closing-edge prior — consistent with the "hard check blocks, no override"
        rule the rest of the battery follows. Kept informational for now to avoid changing
        admission control flow until real closed-loop recheck outcomes are observed.
        """
        members = [
            m
            for m in _closing_edge_effects(self.causal_spec, construct, prior_admitted)
            if m in self.admitted_contributions
        ]
        if not members:
            return None
        design = _design_for_state(
            self.admission,
            self.causal_spec,
            self.data_for_model,
            n_draws=self.n_draws,
            seed=self.seed,
        )
        results: list[dict[str, Any]] = []
        for member in members:
            target = _closed_loop_target(
                self.admitted_contributions[member], self.causal_spec, self.admission.priors
            )
            results.extend(
                _check_result_payload(r)
                for r in recheck_member(self.admission, target, self.causal_spec, design)
            )
        if not results:
            return None
        return {
            "constructs": [*members, construct],
            "closing_edges": [f"{construct}->{m}" for m in members],
            "results": results,
        }


def make_submit_construct_tool(state: ConstructBuildState) -> Any:
    """Build the ``submit_construct`` agent-session tool bound to ``state``."""
    from nof1_causal_lab.utils.openrouter_client import Tool

    async def _execute(
        *,
        construct: str,
        indicators: list[dict[str, Any]],
        priors: dict[str, Any],
        accept: dict[str, str] | None = None,
    ) -> str:
        return state.submit_construct(
            construct=construct,
            indicators=indicators,
            priors=priors,
            accept=accept,
        )

    return Tool(
        name="submit_construct",
        description=(
            "Submit one construct: its indicator emission choices and its priors "
            "(keyed by canonical parameter name). The cumulative model is compiled "
            "and gated by the exact prior-predictive reachability battery; the "
            "returned report says whether the construct is admitted, blocked "
            "(revise), or needs a decision (revise or accept the consequence)."
        ),
        parameters=SUBMIT_CONSTRUCT_SCHEMA,
        execute=_execute,
        stop_on_success=True,
        success_output=None,
    )


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
                "Prior proposals keyed by canonical parameter name (rho_<c>, "
                "sigma_<c>, self_limit_<c>, setpoint_<c>, beta_<p>_<c>, "
                "hill_emax_<p>_<c>/hill_ec50_<p>_<c>/hill_n_<p>_<c>, "
                "lambda_<ind>_<c>, obs_sd_<ind>). A Hill edge is declared by "
                "authoring hill_* priors; a self-limiting well by self_limit_<c>."
            ),
        },
        "accept": {
            "type": "object",
            "description": (
                "Optional: map a soft-check id (e.g. 'C3 resolvability') to a written "
                "rationale to accept its consequence instead of revising."
            ),
        },
    },
    "required": ["construct", "indicators", "priors"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# The build loop
# --------------------------------------------------------------------------- #


async def run_stage4_construct_build(
    *,
    causal_spec: dict,
    question: str,
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]],
    session_factory: StageSessionFactory,
    enable_literature: bool = False,
    n_draws: int = 200,
    seed: int = 0,
    workspace_id: str | None = None,
) -> Stage4Result:
    """Drive construct admission one construct at a time and assemble the result.

    When ``workspace_id`` is given, the loop streams construct-admission telemetry
    (``plan`` → per-attempt ``construct_started``/``construct_checking``/``construct_report``
    → ``done``/``failed``) for the live web view; with ``None`` it runs silently.
    """
    from nof1_causal_lab.flows.stages.stage4.tools import make_search_tool

    from .stage4_construct_prompt import build_construct_messages

    order = build_construct_order(causal_spec)
    if workspace_id:
        emit_stage4_admission_event(
            workspace_id, "plan", _admission_plan_payload(causal_spec, order)
        )
    state = ConstructBuildState(
        causal_spec=causal_spec,
        data_for_model=data_for_model,
        order=order,
        n_draws=n_draws,
        seed=seed,
        workspace_id=workspace_id,
    )

    try:
        for construct in order:
            for _attempt in range(_MAX_ATTEMPTS_PER_CONSTRUCT):
                if state.current_construct != construct:
                    break  # admitted on a previous attempt
                state.attempt = _attempt + 1
                state.submission_made = False
                if workspace_id:
                    emit_stage4_admission_event(
                        workspace_id,
                        "construct_started",
                        {"construct": construct, "attempt": state.attempt},
                    )
                tools = [make_submit_construct_tool(state)]
                if enable_literature:
                    tools.append(make_search_tool(state))
                system_prompt, user_prompt = build_construct_messages(
                    state=state,
                    construct=construct,
                    question=question,
                    causal_spec=causal_spec,
                    indicator_audits=indicator_audits,
                )
                async with session_factory.open(
                    system_prompt=system_prompt,
                    tools=tools,
                    log_label=f"stage-4:construct:{construct}",
                ) as agent_session:
                    await agent_session.turn(user_prompt)
                if not state.submission_made:
                    raise ValueError(
                        f"Stage 4 construct `{construct}` did not call submit_construct before "
                        "the turn ended."
                    )
            if state.current_construct == construct:
                outcome = state.last_report.outcome if state.last_report else "no report"
                raise ValueError(
                    f"Stage 4 construct `{construct}` was not admitted after "
                    f"{_MAX_ATTEMPTS_PER_CONSTRUCT} attempts (last outcome: {outcome})."
                )
    except Exception as exc:
        if workspace_id:
            emit_stage4_admission_event(
                workspace_id,
                "failed",
                {"construct": state.current_construct, "message": str(exc)},
            )
        raise

    if workspace_id:
        emit_stage4_admission_event(workspace_id, "done", {})

    model_spec = state.admission.model_spec().model_dump(mode="json")
    return Stage4Result(
        model_spec=model_spec,
        authored_priors=dict(state.admission.priors),
        search_queries=dict(state.search_queries),
        validation=None,
    )


__all__ = [
    "ConstructBuildState",
    "build_design_info",
    "contribution_from_payload",
    "make_submit_construct_tool",
    "render_admission_feedback",
    "run_stage4_construct_build",
]
