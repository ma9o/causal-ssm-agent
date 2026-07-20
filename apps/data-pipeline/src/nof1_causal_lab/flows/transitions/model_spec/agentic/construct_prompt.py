"""Per-construct prompt for the gradual model-spec flow.

Renders one construct's causal role, its indicators, the canonical parameter
names to author, and — on a re-attempt — the reachability feedback the reviser
must address. Deliberately thin: the reachability battery, not the prompt, is the
source of truth for admission.
"""

from __future__ import annotations

import math
from itertools import pairwise
from statistics import median
from typing import TYPE_CHECKING, Any

import polars as pl

from nof1_causal_lab.distributions import constraint_domain
from nof1_causal_lab.utils.causal_design import (
    choose_reference_indicator,
    get_constructs,
    get_effective_observation_window,
    get_estimation_edges,
    get_estimation_state_order,
    get_indicator_polarity,
    get_indicators,
    get_known_inputs,
)
from nof1_causal_lab.utils.observation_semantics import get_observation_semantics

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
    names = state.catalog.prior_names_for(
        construct,
        admitted_prior_names=state.admission.priors,
    )
    names |= deferred_closing_edge_params(
        state.causal_design, construct, set(state.admission.names)
    )
    return sorted(names)


def _parameter_activation_note(parameter: dict[str, Any]) -> str:
    """Describe the submitted-likelihood condition for a conditional prior surface."""
    if not parameter.get("conditional_prior_surface"):
        return ""
    role = str(parameter["role"])
    if role == "observation_intercept":
        return (
            " — conditional: include only when the chosen channel needs an observation "
            "intercept; omit for threshold/categorical and auto-standardized channels"
        )
    if role == "loading":
        return " — conditional: omit when this indicator uses `categorical`"
    if role == "initial_state_mean":
        return " — conditional: include only when this time-invariant construct is standardized"
    families = parameter.get("activation_distribution_families") or ()
    if families:
        rendered = ", ".join(f"`{family}`" for family in families)
        return f" — conditional: include only when a relevant channel uses {rendered}"
    return " — conditional on the locked model choices"


def _format_number(value: Any) -> str:
    """Compact, deterministic prompt formatting that preserves zero and false."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    return f"{number:.4g}" if math.isfinite(number) else "unavailable"


def _format_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    return f"{number:.1%}" if math.isfinite(number) else "unavailable"


def _validation_frame(validation_report: dict[str, Any]) -> list[str]:
    """Render only validation facts that apply to the whole authoring run."""
    is_valid = validation_report.get("is_valid")
    status = "VALID" if is_valid is True else "INVALID" if is_valid is False else "UNKNOWN"
    lines = [f"Validation report status: **{status}**"]
    dataset_issues = validation_report.get("dataset_issues") or []
    if dataset_issues:
        lines.append("Dataset-level validation issues:")
        for issue in dataset_issues:
            severity = str(issue.get("severity") or "unknown").upper()
            issue_type = str(issue.get("issue_type") or "unspecified")
            message = str(issue.get("message") or "")
            lines.append(f"- [{severity}] {issue_type}: {message}")
    return lines


def _active_construct_frame(causal_design: dict, construct: str) -> list[str]:
    construct_meta = next(
        (item for item in get_constructs(causal_design) if item.get("name") == construct),
        {},
    )
    model_clock = causal_design.get("measurement", {}).get("model_clock") or "unknown"
    theoretical_role = construct_meta.get("role") or "unknown"
    temporal_status = construct_meta.get("temporal_status") or "unknown"
    outcome = "yes" if construct_meta.get("is_outcome") else "no"
    description = str(construct_meta.get("description") or "").strip()
    lines = [
        "# Fixed model frame",
        "",
        f"- Model clock / authored default effect interval: `{model_clock}`",
        "- Raw empirical summaries below are descriptive input data; the compiler-owned "
        "model scale is stated separately for each observation channel.",
        "",
        f"# Active construct: `{construct}`",
        "",
        "- Estimation role: **retained latent state**",
        f"- Theoretical role: `{theoretical_role}`",
        f"- Temporal status: `{temporal_status}`",
        f"- Outcome construct: `{outcome}`",
    ]
    if description:
        lines.append(f"- Meaning: {description}")
    return lines


def _incoming_driver_context(
    state: ConstructBuildState,
    causal_design: dict,
    construct: str,
) -> list[str]:
    """Render executable incoming causes, separated by estimation role."""
    edges = [
        edge for edge in get_estimation_edges(causal_design) if edge.get("effect") == construct
    ]
    state_names = set(get_estimation_state_order(causal_design))
    known_input_by_name = {
        str(item.get("construct") or item.get("construct_name")): item
        for item in get_known_inputs(causal_design)
        if item.get("construct") or item.get("construct_name")
    }
    lines = ["## Incoming drivers"]
    if not edges:
        lines.append("- None — this is an executable root/source.")
        return lines

    for edge in edges:
        cause = str(edge.get("cause"))
        timing = "lagged" if edge.get("lagged", True) else "contemporaneous"
        description = str(edge.get("description") or "").strip()
        suffix = f"; {description}" if description else ""
        if cause in known_input_by_name:
            known_input = known_input_by_name[cause]
            lines.append(
                f"- `{cause}` — **known transition input**, {timing}; "
                f"source indicator=`{known_input.get('source_indicator')}`, "
                f"scale divisor={_format_number(known_input.get('scale', 1.0))}, "
                f"missing policy=`{known_input.get('missing_policy', 'zero')}`{suffix}"
            )
        elif cause in state_names:
            status = (
                "admitted and authorable now"
                if cause in state.admission.names
                else (
                    "deferred feedback parent; its incoming effect is not authorable on this turn"
                )
            )
            lines.append(f"- `{cause}` — retained latent state, {timing}; {status}{suffix}")
        else:
            lines.append(f"- `{cause}` — unclassified estimation cause, {timing}{suffix}")
    return lines


def _observed_values(data_for_model: pl.DataFrame, indicator: str) -> list[float]:
    if not {"indicator", "value"} <= set(data_for_model.columns):
        return []
    values = (
        data_for_model.filter(pl.col("indicator") == indicator)
        .select(pl.col("value").cast(pl.Float64, strict=False))
        .drop_nulls()
        .get_column("value")
        .to_list()
    )
    return [float(value) for value in values if math.isfinite(float(value))]


def _ordinal_occupancy(indicator: dict, values: list[float]) -> str | None:
    levels = indicator.get("ordinal_levels") or []
    if indicator.get("measurement_dtype") != "ordinal" or not levels:
        return None
    counts = [0] * len(levels)
    invalid = 0
    for value in values:
        code = round(value)
        if math.isclose(value, code, abs_tol=1e-8) and 0 <= code < len(levels):
            counts[code] += 1
        else:
            invalid += 1
    entries = [f"{index}={label} ({counts[index]})" for index, label in enumerate(levels)]
    if invalid:
        entries.append(f"invalid/out-of-range ({invalid})")
    return ", ".join(entries)


def _schedule_context(
    data_for_model: pl.DataFrame,
    indicator_names: list[str],
    *,
    temporal_status: str | None,
) -> list[str]:
    """Mirror C3's union-of-observed-anchors design before a proposal is compiled."""
    lines = ["## Dynamics design context"]
    if temporal_status == "time_invariant":
        lines.append(
            "- Time-invariant construct: settling-time and transmission checks do not apply."
        )
        return lines
    if not {"indicator", "value", "anchor_time"} <= set(data_for_model.columns):
        lines.append("- Observed anchor schedule: unavailable in the current panel.")
        return lines
    observed = data_for_model.filter(
        pl.col("indicator").is_in(indicator_names)
        & pl.col("value").is_not_null()
        & pl.col("anchor_time").is_not_null()
    )
    if observed.is_empty():
        lines.append("- Observed anchor schedule: no non-null active-indicator observations.")
        return lines
    anchors = sorted(set(observed.get_column("anchor_time").to_list()))
    if not anchors:
        lines.append("- Observed anchor schedule: unavailable in the current panel.")
        return lines
    span_days = (anchors[-1] - anchors[0]).total_seconds() / 86_400.0
    gaps = [(right - left).total_seconds() / 86_400.0 for left, right in pairwise(anchors)]
    if gaps:
        lines.append(
            f"- Actual observed-anchor schedule across this construct: {len(anchors)} distinct "
            f"times; span={_format_number(span_days)} days; median gap="
            f"{_format_number(median(gaps))} days; maximum gap={_format_number(max(gaps))} days."
        )
    else:
        lines.append(
            "- Actual observed-anchor schedule across this construct: 1 distinct time; "
            "span=0 days; gaps unavailable."
        )
    return lines


def _profile_lines(profile: dict[str, Any] | None) -> list[str]:
    if not profile:
        return ["  - Raw empirical profile: unavailable (no numeric observations)."]

    lines = [
        "  - Raw empirical profile: "
        + "; ".join(
            f"{label}={_format_number(profile.get(key))}"
            for key, label in (
                ("n_obs", "n"),
                ("mean", "mean"),
                ("std", "sd"),
                ("variance", "variance"),
            )
        )
    ]
    lines.append(
        "  - Raw five-number summary (min/q25/median/q75/max): "
        + "/".join(_format_number(profile.get(key)) for key in ("min", "q25", "q50", "q75", "max"))
    )
    shape_parts = [
        f"zero fraction={_format_percent(profile.get('zero_fraction'))}",
        f"variance/mean={_format_number(profile.get('variance_to_mean_ratio'))}",
        f"nonnegative={_format_number(profile.get('is_nonnegative'))}",
        f"unit interval={_format_number(profile.get('is_unit_interval'))}",
        f"integer-like={_format_number(profile.get('looks_integer_valued'))}",
    ]
    lines.append("  - Observed shape: " + "; ".join(shape_parts))

    temporal_parts: list[str] = []
    if profile.get("time_coverage_ratio") is not None:
        temporal_parts.append(
            "coverage/minimum-required-span=" + _format_percent(profile.get("time_coverage_ratio"))
        )
    if profile.get("max_gap_ratio") is not None:
        temporal_parts.append(
            "largest-gap/allowed-threshold=" + _format_number(profile.get("max_gap_ratio")) + "x"
        )
    if temporal_parts:
        lines.append("  - Validation timing ratios: " + "; ".join(temporal_parts))

    quality_parts: list[str] = []
    for key, label in (
        ("dtype_violations", "dtype violations"),
        ("duplicate_pct", "duplicate fraction"),
        ("n_unparseable_timestamps", "unparseable timestamps"),
        ("arithmetic_sequence_detected", "arithmetic sequence detected"),
    ):
        value = profile.get(key)
        if value is not None:
            rendered = _format_percent(value) if key == "duplicate_pct" else _format_number(value)
            quality_parts.append(f"{label}={rendered}")
    if quality_parts:
        lines.append("  - Data-quality metrics: " + "; ".join(quality_parts))
    return lines


def _indicator_card(
    *,
    indicator: dict,
    reference_var: str | None,
    audit: dict[str, Any],
    model_clock: str | None,
    data_for_model: pl.DataFrame,
) -> list[str]:
    variable = str(indicator["name"])
    polarity = get_indicator_polarity(indicator)
    if variable == reference_var:
        loading = "+1" if polarity == "positive" else "-1"
        role = f"reference indicator (compiler-fixed loading {loading})"
    else:
        role = f"free {polarity} loading"
    semantics = get_observation_semantics(indicator)
    effective_window = get_effective_observation_window(indicator, model_clock) or "unknown"
    dtype = indicator.get("measurement_dtype") or "unknown"
    aggregation = indicator.get("aggregation") or "unknown"
    lines = [
        f"### `{variable}` — {role}",
        f"- Declared observation semantics: dtype=`{dtype}`; aggregation=`{aggregation}`; "
        f"support=`{semantics.support_kind.value}`; summary=`{semantics.summary_operator.value}`; "
        f"anchor=`{semantics.anchor_policy.value}`; effective window=`{effective_window}`.",
    ]
    levels = indicator.get("ordinal_levels") or []
    if levels:
        lines.append(
            "- Declared ordinal codebook (structural support): "
            + ", ".join(f"{index}={level}" for index, level in enumerate(levels))
        )
    how = str(indicator.get("how_to_measure") or "").strip()
    if how:
        lines.append(f"- Measurement rule: {how}")
    lines.append(
        "- Compiler model scale: compatible discrete/bounded families retain their natural "
        "scale; Gaussian/Student-t identity is standardized only for additive-location "
        "mean/first/last channels."
    )

    profile = audit.get("profile") or {}
    lines.extend(_profile_lines(profile))
    values = _observed_values(data_for_model, variable)
    if occupancy := _ordinal_occupancy(indicator, values):
        lines.append("  - Observed ordinal occupancy: " + occupancy)

    issues = (audit.get("validation") or {}).get("issues") or []
    if issues:
        lines.append("  - Validation issues:")
        for issue in issues:
            severity = str(issue.get("severity") or "unknown").upper()
            issue_type = str(issue.get("issue_type") or "unspecified")
            message = str(issue.get("message") or "")
            lines.append(f"    - [{severity}] {issue_type}: {message}")
    else:
        lines.append("  - Validation issues: none.")

    sparse_declared_levels = (
        dtype == "ordinal"
        and len(levels) >= 2
        and profile.get("n_obs", 0) > 0
        and profile.get("min") == profile.get("max")
    )
    if sparse_declared_levels:
        lines.append(
            "  - SPARSE LEVEL COVERAGE: only one level is observed, but the declared "
            "ordinal levels define the likelihood support; keep the compatible discrete "
            "emission and treat limited learning as a data limitation."
        )
    return lines


def build_construct_messages(
    *,
    state: ConstructBuildState,
    construct: str,
    question: str,
    causal_design: dict,
    validation_report: dict[str, Any],
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
    audits = dict(validation_report.get("indicators") or {})
    param_names = _canonical_parameter_names(state, construct)
    construct_meta = next(
        (item for item in get_constructs(causal_design) if item.get("name") == construct),
        {},
    )
    model_clock = causal_design.get("measurement", {}).get("model_clock")
    validation_frame = _validation_frame(validation_report)

    lines: list[str] = [
        f"# Research question\n\n{question}",
        "",
        *validation_frame,
        *([""] if validation_frame else []),
        *_active_construct_frame(causal_design, construct),
        "",
        "Already admitted: " + (", ".join(state.admission.names) or "(none yet)"),
        "",
        *_incoming_driver_context(state, causal_design, construct),
        "",
        *_schedule_context(
            state.data_for_model,
            [str(indicator["name"]) for indicator in indicators],
            temporal_status=construct_meta.get("temporal_status"),
        ),
        "",
        "## Indicators of this construct",
    ]
    for ind in indicators:
        lines.extend(
            [
                "",
                *_indicator_card(
                    indicator=ind,
                    reference_var=reference_var,
                    audit=audits.get(str(ind["name"])) or {},
                    model_clock=model_clock,
                    data_for_model=state.data_for_model,
                ),
            ]
        )

    closing_betas = sorted(
        n
        for n in deferred_closing_edge_params(causal_design, construct, set(state.admission.names))
        if n.startswith("beta_")
    )
    catalog = state.catalog
    assert catalog is not None  # guaranteed by _canonical_parameter_names above
    saturating_parents = [
        parent
        for parent in construct_parents(causal_design, construct)
        if parent in state.admission.names
    ]
    lines += [
        "",
        "## Canonical parameters available for this construct",
        "",
        "Author a prior for each active parameter below, plus any optional structural "
        "declaration you choose to enable (listed next). Parameters marked conditional "
        "must be omitted when your submitted likelihood does not activate them; the tool "
        "checks this against the locked family and link. Do NOT author a prior for any name "
        "in neither list — it is not a free parameter of this construct and is rejected. "
        "Each prior's support must lie within the stated domain. "
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
            f"{_parameter_activation_note(dict(catalog.metadata_for(n)))}"
        )
    if "obs_cat_slopes" in param_names:
        lines.append(
            "- Note: if every indicator of this construct uses a `categorical` "
            "likelihood, the reference channel's first non-baseline slope is "
            "compiler-pinned to +1 (the construct's scale and sign anchor); the "
            "`obs_cat_slopes` prior applies to the remaining free slopes."
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
    structural_lines = [
        "Optional structural declarations (author the prior to enable):",
        f"- `self_limit_{construct}` — a self-limiting (quartic) well for bounded excursions.",
    ]
    if saturating_parents:
        structural_lines.append(
            "- Saturating effects are available only for these admitted latent parents: "
            + ", ".join(f"`{parent}`" for parent in saturating_parents)
            + ". Replace the corresponding linear `beta` with its `hill_emax`, "
            "`hill_ec50`, and `hill_n` priors. Known-input effects are linear-only."
        )
    else:
        structural_lines.append(
            "- No saturating parent effect is authorable on this turn. Known-input effects "
            "are linear-only."
        )
    lines += [
        *structural_lines,
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
