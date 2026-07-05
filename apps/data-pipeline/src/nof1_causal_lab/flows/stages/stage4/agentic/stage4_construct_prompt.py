"""Per-construct prompt for the gradual Stage 4 flow.

Renders one construct's causal role, its indicators, the canonical parameter
names to author, and — on a re-attempt — the reachability feedback the reviser
must address. Deliberately thin: the reachability battery, not the prompt, is the
source of truth for admission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.utils.causal_spec import choose_reference_indicator, get_indicators

from .prompts.shared_fragments import (
    CONTINUOUS_TIME_DYNAMICS_SECTION,
    LINK_FUNCTION_RULES_SECTION,
    OBSERVATION_DISTRIBUTION_GUIDANCE_SECTION,
    PRIOR_DISTRIBUTION_TYPES_SECTION,
)
from .stage4_construct_flow import construct_parents, render_admission_feedback

if TYPE_CHECKING:
    from .stage4_construct_flow import ConstructBuildState

_SYSTEM_TASK = """You are specifying one construct of a continuous-time latent state-space model,
one construct at a time along the causal graph. For the active construct you author:

- its **emission** for each indicator (observation family + link), and
- its **priors**, keyed by canonical parameter name.

The cumulative partial model is then compiled and simulated through the exact
prior-predictive engine and gated by a reachability battery (confinement, latent
scale, design-resolvability, edge overwhelm, saturation, coverage). You will see
the report. If a hard check fails you must revise; if a soft check fails you may
revise or accept its consequence with a written rationale via `accept`.

Author priors on the natural scale of each parameter. Latents are standardized
(unit scale); loadings map the latent to the indicator's observation scale.
Elicit each dynamic construct's settling time so it is resolvable at the
observation cadence and within the study span."""


def _indicators_for(causal_spec: dict, construct: str) -> list[dict]:
    return [i for i in get_indicators(causal_spec) if i.get("construct_name") == construct]


def _canonical_parameter_names(
    causal_spec: dict,
    state: ConstructBuildState,
    construct: str,
    indicators: list[dict],
    reference_var: str | None,
) -> list[str]:
    """The canonical parameter names this construct is expected to author priors for."""
    names = [f"rho_{construct}", f"sigma_{construct}"]
    for ind in indicators:
        var = ind["name"]
        names.append(f"obs_sd_{var}")
        if var != reference_var:
            names.append(f"lambda_{var}_{construct}")
    for parent in construct_parents(causal_spec, construct):
        if parent in state.admission.names:
            names.append(f"beta_{parent}_{construct}")
    return names


def build_construct_messages(
    *,
    state: ConstructBuildState,
    construct: str,
    question: str,
    causal_spec: dict,
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

    indicators = _indicators_for(causal_spec, construct)
    reference = choose_reference_indicator(indicators)
    reference_var = reference["name"] if reference else None
    admitted_parents = [
        p for p in construct_parents(causal_spec, construct) if p in state.admission.names
    ]
    param_names = _canonical_parameter_names(
        causal_spec, state, construct, indicators, reference_var
    )

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
        hint = audit.get("recommended_distribution") or audit.get("dtype") or ""
        how = ind.get("how_to_measure", "")
        lines.append(
            f"- `{var}` — {role}. {how} {f'(audit hint: {hint})' if hint else ''}".rstrip()
        )

    lines += [
        "",
        "## Author priors for these canonical parameters",
        ", ".join(f"`{n}`" for n in param_names),
        "",
        "Optional structural declarations (author the prior to enable):",
        f"- `self_limit_{construct}` — a self-limiting (quartic) well for bounded excursions.",
        f"- `setpoint_{construct}` — a nonzero equilibrium/center for this construct.",
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

    if (
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
