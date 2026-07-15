"""Per-construct prompt for the gradual model-spec flow.

Renders one construct's causal role, its indicators, the canonical parameter
names to author, and — on a re-attempt — the reachability feedback the reviser
must address. Deliberately thin: the reachability battery, not the prompt, is the
source of truth for admission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.distributions import constraint_domain
from nof1_causal_lab.utils.causal_design import choose_reference_indicator, get_indicators

from .construct_flow import (
    construct_parents,
    deferred_closing_edge_params,
    render_admission_feedback,
)
from .prompts.shared_fragments import (
    CONTINUOUS_TIME_DYNAMICS_SECTION,
    LINK_FUNCTION_RULES_SECTION,
    OBSERVATION_DISTRIBUTION_GUIDANCE_SECTION,
    PRIOR_DISTRIBUTION_TYPES_SECTION,
)

if TYPE_CHECKING:
    from .construct_flow import ConstructBuildState

_SYSTEM_TASK = """You are specifying one construct of a continuous-time latent state-space model,
one construct at a time along the causal graph. For the active construct you author:

- its **emission** for each indicator (observation family + link), and
- its **priors**, keyed by canonical parameter name.

The cumulative partial model is then compiled and simulated through the exact
prior-predictive engine and gated by a reachability battery (confinement, latent
scale, design-resolvability, edge overwhelm, saturation, coverage). You will see
the report. If a hard check fails you must revise; if a soft check fails you may
revise or accept its consequence via `accept`, naming the exact `check` and
`target` from the current failure plus a written `rationale`.

Author priors on the natural scale of each parameter. Latents are standardized
(unit scale); loadings map the latent to the indicator's observation scale.
Gaussian/Student-t indicators with an identity link and mean/first/last summaries
are themselves standardized (mean 0, sd 1) before fitting — author their noise,
loading, and manifest-mean priors on that unit scale, not on the raw data scale
shown in the audit profile. All other families keep their natural data scale.
Elicit each dynamic construct's settling time so it is resolvable at the
observation cadence and within the study span.

You must finish by invoking the registered MCP tool `submit_construct`; writing
or describing a `submit_construct(...)` call in text does not execute it and
fails the attempt. Follow the tool schema exactly (`indicators`, not
`emissions`). Do not inspect the filesystem or use shell commands: this prompt
and the registered tool schema contain everything required for the submission."""


def _indicators_for(causal_design: dict, construct: str) -> list[dict]:
    return [i for i in get_indicators(causal_design) if i.get("construct_name") == construct]


def _canonical_parameter_names(state: ConstructBuildState, construct: str) -> list[str]:
    """The compiler-authoritative free parameters this construct may author priors for."""
    assert state.catalog is not None  # set in ConstructBuildState.__post_init__
    names = set(state.catalog.by_construct.get(construct, ()))
    names |= deferred_closing_edge_params(
        state.causal_design, construct, set(state.admission.names)
    )
    return sorted(names)


def build_construct_messages(
    *,
    state: ConstructBuildState,
    construct: str,
    question: str,
    causal_design: dict,
    indicator_audits: dict[str, dict],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for admitting ``construct``."""
    system = "\n\n".join(
        [
            _SYSTEM_TASK,
            OBSERVATION_DISTRIBUTION_GUIDANCE_SECTION,
            LINK_FUNCTION_RULES_SECTION,
            PRIOR_DISTRIBUTION_TYPES_SECTION,
            CONTINUOUS_TIME_DYNAMICS_SECTION,
        ]
    )

    indicators = _indicators_for(causal_design, construct)
    reference = choose_reference_indicator(indicators)
    reference_var = reference["name"] if reference else None
    admitted_parents = [
        p for p in construct_parents(causal_design, construct) if p in state.admission.names
    ]
    param_names = _canonical_parameter_names(state, construct)

    lines: list[str] = [
        f"# Research question\n\n{question}",
        f"# Active construct: `{construct}`",
        "",
        "Already admitted: " + (", ".join(state.admission.names) or "(none yet)"),
        "Direct causal parents (edges into this construct): "
        + (", ".join(admitted_parents) or "(none — this is a root/source)"),
        "",
        "## Indicators of this construct",
    ]
    for ind in indicators:
        var = ind["name"]
        role = "reference (unit loading)" if var == reference_var else "free loading"
        audit = indicator_audits.get(var, {})
        profile = audit.get("profile", {})
        hint = audit.get("recommended_distribution") or audit.get("dtype") or ""
        how = ind.get("how_to_measure", "")
        sparse_declared_levels = (
            ind.get("measurement_dtype") == "ordinal"
            and len(ind.get("ordinal_levels") or []) >= 2
            and profile.get("n_obs", 0) > 0
            and profile.get("min") == profile.get("max")
        )
        usability = (
            " SPARSE LEVEL COVERAGE: only one level is observed, but the declared ordinal "
            "levels define the likelihood support; keep the compatible discrete emission and "
            "treat limited learning as a data limitation."
            if sparse_declared_levels
            else ""
        )
        lines.append(
            f"- `{var}` — {role}. {how} {f'(audit hint: {hint})' if hint else ''}"
            f"{usability}".rstrip()
        )

    closing_betas = sorted(
        n
        for n in deferred_closing_edge_params(causal_design, construct, set(state.admission.names))
        if n.startswith("beta_")
    )
    catalog = state.catalog
    assert catalog is not None  # guaranteed by _canonical_parameter_names above
    lines += [
        "",
        "## Author priors for exactly these canonical parameters",
        "",
        "Author a prior for each parameter below, plus any optional structural "
        "declaration you choose to enable (listed next). Do NOT author a prior for "
        "any name in neither list — it is not a free parameter of this construct "
        "and is rejected. Each prior's support must lie within the stated domain. "
        "Use the exact value shape "
        '`{"distribution": "Normal", "params": {"mu": 0, "sigma": 1}, '
        '"reasoning": "..."}`. Do not use `dist` or place distribution parameters '
        "at the top level.",
        "",
    ]
    for n in param_names:
        role, constraint = catalog.role_for(n)
        site_name = catalog.site_for(n)
        pooled_families = {
            prior.get("distribution")
            for prior_name, prior in state.admission.priors.items()
            if site_name is not None and catalog.site_for(prior_name) == site_name
        }
        pooled_families.discard(None)
        family_requirement = (
            f" — pooled compiler site `{site_name}`: MUST use "
            f"`{next(iter(pooled_families))}` to match admitted parameters"
            if len(pooled_families) == 1
            else ""
        )
        lines.append(
            f"- `{n}` — {role.value.replace('_', ' ')} — support ⊆ "
            f"{constraint_domain(constraint.value)}{family_requirement}"
        )
    lines.append("")
    if closing_betas:
        lines += [
            "This construct closes a feedback loop: the edge(s) "
            + ", ".join(f"`{n}`" for n in closing_betas)
            + " point INTO an already-admitted construct and first materialize now. "
            "Author their priors in THIS submission — they could not be authored earlier.",
            "",
        ]
    lines += [
        "Optional structural declarations (author the prior to enable):",
        f"- `self_limit_{construct}` — a self-limiting (quartic) well for bounded excursions.",
        "- for a *saturating* parent effect, replace `beta_<p>_"
        + construct
        + "` with `hill_emax_<p>_"
        + construct
        + "`, `hill_ec50_<p>_"
        + construct
        + "`, `hill_n_<p>_"
        + construct
        + "`.",
        "",
        f"Call `submit_construct` with construct=`{construct}`, an emission for each "
        "indicator, and the priors object.",
    ]

    if state.last_tool_feedback is not None:
        lines += [
            "",
            "## Latest tool feedback — revise the submission",
            state.last_tool_feedback,
        ]
    elif (
        state.last_report is not None
        and state.last_report.name == construct
        and not state.last_report.admitted
    ):
        lines += [
            "",
            "## Latest reachability feedback — revise the flagged priors or accept the consequence",
            render_admission_feedback(state.last_report),
        ]

    return system, "\n".join(lines)
