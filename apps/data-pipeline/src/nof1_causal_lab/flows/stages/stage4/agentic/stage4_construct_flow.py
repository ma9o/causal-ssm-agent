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

from dataclasses import dataclass, field
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
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionReport,
    AdmissionState,
    ConstructContribution,
    DesignInfo,
    admit_construct,
    build_construct_order,
    restrict_causal_spec,
    trial_admission_state,
)
from nof1_causal_lab.utils.causal_spec import get_estimation_edges

from .stage4_types import Stage4Result

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
            name = param["name"]
            roles[name] = (ParameterRole(param["role"]), ParameterConstraint(param["constraint"]))
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
    """Derive the reachability design against the compiled *partial* model.

    Uses the canonical ``prepare_model_runtime`` so the sampling grid, the
    per-indicator observation indices, and the observed values all live in the
    same time + observation space the fit uses — including support-aware handling
    and the emission-space scaling the raw data does not carry.
    """
    import polars as pl

    from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
    from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime

    trial = trial_admission_state(state, contribution)
    restricted = restrict_causal_spec(causal_spec, set(trial.names))
    compiled = compile_ssm_artifact(trial.model_spec(), dict(trial.priors), causal_spec=restricted)

    indicator_names = [lik.variable for lik in trial.likelihoods]
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
    catalog: ParamCatalog | None = None
    admission: AdmissionState = field(default_factory=AdmissionState)
    cursor: int = 0
    search_queries: dict[str, str] = field(default_factory=dict)
    search_cache: dict[str, str] = field(default_factory=dict)
    last_report: AdmissionReport | None = None
    submission_made: bool = False

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
        allowed = self.catalog.allowed_for(construct, parents)
        unknown = [name for name in priors if name not in allowed]
        if unknown:
            return (
                f"These parameters are not free for `{construct}` and cannot take a prior: "
                f"{', '.join(sorted(unknown))}. Author priors only for: "
                f"{', '.join(sorted(allowed))}."
            )
        payload = {"construct": construct, "indicators": list(indicators), "priors": dict(priors)}
        contribution = contribution_from_payload(self.causal_spec, payload, self.catalog)
        design = build_design_info(
            self.admission,
            contribution,
            self.causal_spec,
            self.data_for_model,
            n_draws=self.n_draws,
            seed=self.seed,
        )
        new_state, report = admit_construct(
            self.admission, contribution, self.causal_spec, design, accepted=dict(accept or {})
        )
        self.last_report = report
        if report.admitted:
            self.admission = new_state
            self.cursor += 1
        return render_admission_feedback(report)


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
) -> Stage4Result:
    """Drive construct admission one construct at a time and assemble the result."""
    from nof1_causal_lab.flows.stages.stage4.tools import make_search_tool

    from .stage4_construct_prompt import build_construct_messages

    order = build_construct_order(causal_spec)
    state = ConstructBuildState(
        causal_spec=causal_spec,
        data_for_model=data_for_model,
        order=order,
        n_draws=n_draws,
        seed=seed,
    )

    for construct in order:
        for _attempt in range(_MAX_ATTEMPTS_PER_CONSTRUCT):
            if state.current_construct != construct:
                break  # admitted on a previous attempt
            state.submission_made = False
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
