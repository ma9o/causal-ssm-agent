"""Stage 4 megaprompt: single-prompt mode for the full model + priors task.

When the Stage 4 state machine is disabled via ``Stage4Config.state_machine_enabled``
the orchestrator exposes every submit tool at once and lets the LLM structure
its own work. These builders render the cohesive system + user prompts for
that mode, reusing the per-scope card rendering already defined in
``model_proposal`` so the contextual material stays consistent with the
state-machine flow.
"""

from __future__ import annotations

from typing import Any

from causal_ssm_agent.distributions import (
    PRIOR_PARAMETER_GUIDANCE_ROWS,
    render_dynamic_prior_scale_guidance,
    render_lagged_beta_authored_interval_guidance,
    render_observation_distribution_guidance_bullets,
    render_observation_link_guidance_bullets,
    render_prior_distribution_guidance_bullets,
)

from .model_proposal import (
    PRIOR_SOURCE_GUIDANCE,
    _join_sections,
    format_construct_scale_cards,
    format_distribution_cards,
    format_loading_params,
    format_model_topology,
    format_prior_cards,
)


def _render_parameter_guidance_table() -> str:
    """Render the full parameter guidance table (all rows) for the megaprompt."""
    lines = [
        "| Type | Typical Distribution | Typical Range | Scale |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {row.parameter_type} | {row.typical_distribution} | {row.typical_range} | {row.scale} |"
        for row in PRIOR_PARAMETER_GUIDANCE_ROWS
    )
    return "\n".join(lines)


def build_stage4_megaprompt_system_prompt(
    *,
    enable_literature: bool,
    enable_paraphrasing: bool,
) -> str:
    """Build the Stage 4 megaprompt system prompt.

    Unlike the frontier-scoped system prompt, this one names every tool the
    model may call and documents the validation gates each submission has to
    clear. The action space is identical to the state-machine mode; the
    difference is only that the harness does not restrict which tool is
    available at any given instant.
    """
    tools = [
        "- `submit_model_configuration`: lock `initialization_policy`, "
        "`observation_intercept_policy`, and `equilibrium_forcing` for the whole "
        "model. Call this exactly once.",
        "- `submit_indicator_choice`: choose the likelihood `distribution` and "
        "`link` for one ambiguous indicator. Call this once per variable listed "
        "in the Distribution Decision Cards.",
        "- `submit_prior_block`: submit prior proposals keyed by parameter name "
        "for any subset of the parameter inventory. Call this as many times as "
        "you need — each call merges into the accepted authored-priors state.",
    ]
    if enable_literature:
        tools.append(
            "- `search_literature`: fetch empirical effect-size evidence for a "
            "single parameter before you set its prior. Pass the parameter's "
            "exact name. Prefer a single well-formed query per parameter."
        )
    if enable_paraphrasing:
        tools.append(
            "- `elicit_prior_gmm`: run robust paraphrased prior elicitation with "
            "GMM aggregation for one hard-to-calibrate parameter."
        )
    sections = [
        (
            "You are a Bayesian statistician completing Stage 4 of a causal-inference "
            "pipeline that fits a continuous-time latent state-space model (CT-SSM) to "
            "observational longitudinal data. The causal structure, indicator inventory, "
            "and parameter inventory are already fixed upstream. Stage 4 turns that "
            "structure into a fully specified generative model with valid priors."
        ),
        (
            "## Operating Mode\n\n"
            "- Do NOT run shell commands, read files, browse a codebase, or use any "
            "ambient agentic tools. The user message already contains every card, "
            "topology summary, parameter inventory, and guidance table you need.\n"
            "- The only tools that advance Stage 4 are the MCP submit tools listed "
            "under `Tool Contract` below. Act by calling them; do not explore.\n"
            "- Reason directly from the provided cards. If information seems missing, "
            "fall back to the weakly-informative defaults documented in the prior "
            "distribution guidance — do not try to fetch anything from disk or the web."
        ),
        (
            "## Your Job\n\n"
            "- Finalize the model specification: pick a likelihood for every ambiguous "
            "indicator, and lock the global initialization / observation-intercept / "
            "equilibrium-forcing configuration.\n"
            "- Elicit priors for every non-optional parameter listed in the Parameter "
            "Prior Cards. Priors for `t0_mean_*` and `t0_sd_*` parameters are optional; "
            "all other parameters require a prior before the stage can finish.\n"
            "- Do all of this with one shared tool surface — you decide the order. "
            "There is no state machine narrowing the scope for you."
        ),
        (
            "## What Is Already Fixed\n\n"
            "- deterministic likelihoods where indicator dtype leaves no ambiguity — "
            "those indicators do not appear in the Distribution Decision Cards.\n"
            "- the complete parameter inventory implied by the causal structure.\n"
            "- loading orientations (fixed from Stage 1b indicator polarity).\n"
            "- construct scale cards and empirical indicator profiles prepared by the "
            "pipeline."
        ),
        (
            "## Submission Rules\n\n"
            "Every submission is routed through the same validation pipeline used by "
            "the state-machine mode: Pydantic schema validation, model compilation, "
            "prior-predictive checks, and output-Jacobian sensitivity. Submissions are "
            "accepted incrementally:\n\n"
            "- `submit_model_configuration` and `submit_indicator_choice` feed a draft "
            "ModelSpec. Once the configuration plus every ambiguous indicator has been "
            "submitted, the draft is locked and the validator returns "
            "`MODEL SPEC LOCKED`.\n"
            "- `submit_prior_block` accepts any subset of the parameter inventory. Each "
            "call is schema-checked per parameter, then merged into the accepted "
            "authored-priors state. You may resubmit a parameter to overwrite its prior.\n"
            "- After every submit call you will receive compact validator feedback. If "
            "feedback starts with `VALIDATION ERRORS` or `COMPILE ERROR`, correct only "
            "the fields it names and resubmit — previously accepted state is preserved.\n"
            "- When the model spec is locked, every non-optional prior is authored, and "
            "the combined validation passes, the validator returns `VALID` and the stage "
            "is done. Stop immediately after you see `VALID`."
        ),
        "## Observation Distribution Guidance\n\n"
        + render_observation_distribution_guidance_bullets(),
        (
            "## Link Function Rules\n\n"
            "Most distributions have exactly one valid link (auto-determined). "
            "You only choose when multiple are valid:\n"
            + render_observation_link_guidance_bullets()
        ),
        "## Prior Distribution Types\n\n" + render_prior_distribution_guidance_bullets(),
        "## Parameter Guidance\n\n" + _render_parameter_guidance_table(),
        "## Continuous-Time Dynamics\n\n" + render_dynamic_prior_scale_guidance(),
        "## Lagged Effect Interval Guidance\n\n" + render_lagged_beta_authored_interval_guidance(),
        (
            "## Initial-State Scale Discipline\n\n"
            "- `t0_mean_*` and `t0_sd_*` live on the latent state scale.\n"
            "- Do not set `t0_mean_*` to the raw reference-indicator mean or "
            "`log(mean(indicator))` just because the indicator uses an identity or log link.\n"
            "- Default to weakly informative latent-scale priors such as `Normal(0, 1)` "
            "and `HalfNormal(1)` unless the construct is explicitly identified on an "
            "observed scale."
        ),
        (
            "## Dynamics / Effect Budget Discipline\n\n"
            "- AR coefficients and residual SDs determine how much damping is available "
            "for downstream incoming lagged effects. Avoid near-unit-root persistence "
            "and overly wide uncertainty unless you have strong evidence.\n"
            "- In dense SCC rows and feedback-coupled edges, start incoming effects from "
            "tightly zero-centered priors with modest uncertainty (often `Normal(0, "
            "0.1-0.2)`) unless longitudinal evidence clearly justifies more.\n"
            "- If the validator reports a partial-drift failure, tighten dynamics priors "
            "toward faster decay and/or shrink effect means and scales."
        ),
        "## Tool Contract\n\n"
        "The available tools are:\n"
        + "\n".join(tools)
        + (
            "\n\nYou may call any tool at any time. You may call the same tool "
            "multiple times. Every call returns validator feedback; read it and act "
            "on it. Do not paraphrase the same literature search for the same "
            "parameter twice. Once the validator returns `VALID`, stop."
        ),
    ]
    return _join_sections(sections)


def _render_ambiguous_indicator_decision_block(
    item: dict[str, Any],
    *,
    current_choice: dict[str, Any] | None,
) -> str:
    """Render a compact enumeration of allowed likelihoods for one ambiguous indicator."""
    variable = str(item["variable"])
    if "fixed_distribution" in item:
        fixed = str(item["fixed_distribution"])
        links = [str(link) for link in item.get("valid_links") or []]
        option_line = f"- distribution: `{fixed}` (fixed); allowed links: {', '.join(f'`{link}`' for link in links) or '(none)'}"
    else:
        option_line_parts = []
        link_options = item.get("link_options") or {}
        for distribution in item.get("valid_distributions") or []:
            links = [str(link) for link in link_options.get(distribution) or []]
            link_text = ", ".join(f"`{link}`" for link in links) if links else "(none)"
            option_line_parts.append(f"`{distribution}` → {link_text}")
        option_line = "- allowed options: " + "; ".join(option_line_parts)
    current_text = (
        f"`{current_choice['distribution']}` / `{current_choice['link']}`"
        if current_choice
        else "(unset)"
    )
    return f"- `{variable}`: current choice: {current_text}\n  {option_line}"


def _render_model_decision_status(
    *,
    ambiguous_indicators: list[dict[str, Any]],
    distribution_choices: dict[str, dict[str, Any]],
    initialization_policy: str | None,
    observation_intercept_policy: str | None,
    equilibrium_forcing: bool | None,
    centerable_construct_names: tuple[str, ...],
    baseline_factor_names: tuple[str, ...],
) -> str:
    """Render the model-decision status table for the megaprompt user message."""
    init_text = f"`{initialization_policy}`" if initialization_policy else "(unset)"
    obs_text = f"`{observation_intercept_policy}`" if observation_intercept_policy else "(unset)"
    if equilibrium_forcing is None:
        forcing_text = "(unset)"
    else:
        forcing_text = f"`{str(bool(equilibrium_forcing)).lower()}`"

    indicator_lines = [
        _render_ambiguous_indicator_decision_block(
            item,
            current_choice=distribution_choices.get(str(item["variable"])),
        )
        for item in ambiguous_indicators
    ] or ["(no ambiguous indicators — skip `submit_indicator_choice`)"]

    centerable = ", ".join(f"`{name}`" for name in centerable_construct_names) or "(none)"
    baseline = ", ".join(f"`{name}`" for name in baseline_factor_names) or "(none)"

    return "\n".join(
        [
            "### Global Configuration (`submit_model_configuration`)",
            "",
            f"- `initialization_policy` (`stationary` / `free`): {init_text}",
            f"- `observation_intercept_policy` (`fixed` / `free`): {obs_text}",
            f"- `equilibrium_forcing` (`true` / `false`): {forcing_text}",
            f"- centered-indicator constructs that could identify a latent baseline: {centerable}",
            f"- compiled baseline-factor scales from marginalized confounders: {baseline}",
            "",
            "### Ambiguous Indicators (`submit_indicator_choice`)",
            "",
            *indicator_lines,
        ]
    )


def _render_prior_status(
    *,
    required_prior_names: tuple[str, ...],
    optional_prior_names: tuple[str, ...],
    authored_priors: dict[str, dict[str, Any]],
) -> str:
    """Render the current authored-prior coverage summary."""
    missing_required = [name for name in required_prior_names if name not in authored_priors]
    authored_required = [name for name in required_prior_names if name in authored_priors]
    authored_optional = [name for name in optional_prior_names if name in authored_priors]

    lines = ["### Prior Coverage", ""]
    lines.append(
        f"- required priors authored: `{len(authored_required)}/{len(required_prior_names)}`"
    )
    if missing_required:
        preview = ", ".join(f"`{name}`" for name in missing_required[:20])
        if len(missing_required) > 20:
            preview += f", … ({len(missing_required) - 20} more)"
        lines.append(f"- still missing: {preview}")
    else:
        lines.append("- still missing: (none)")
    if optional_prior_names:
        lines.append(
            f"- optional priors authored: `{len(authored_optional)}/{len(optional_prior_names)}`"
            + (
                " (optional priors may be omitted — defaults will be used)"
                if len(authored_optional) < len(optional_prior_names)
                else ""
            )
        )
    return "\n".join(lines)


def build_stage4_megaprompt_user_prompt(
    *,
    question: str,
    model_topology: dict[str, Any],
    distribution_cards: list[dict[str, Any]],
    loading_params: list[dict[str, Any]],
    construct_scale_cards: list[dict[str, Any]],
    prior_cards: list[dict[str, Any]],
    ambiguous_indicators: list[dict[str, Any]],
    distribution_choices: dict[str, dict[str, Any]],
    initialization_policy: str | None,
    observation_intercept_policy: str | None,
    equilibrium_forcing: bool | None,
    centerable_construct_names: tuple[str, ...],
    baseline_factor_names: tuple[str, ...],
    required_prior_names: tuple[str, ...],
    optional_prior_names: tuple[str, ...],
    authored_priors: dict[str, dict[str, Any]],
    model_spec_locked: bool,
    latest_feedback: str,
    include_prior_source_guidance: bool,
) -> str:
    """Build the Stage 4 megaprompt user message for one outer agent turn."""
    decision_status = _render_model_decision_status(
        ambiguous_indicators=ambiguous_indicators,
        distribution_choices=distribution_choices,
        initialization_policy=initialization_policy,
        observation_intercept_policy=observation_intercept_policy,
        equilibrium_forcing=equilibrium_forcing,
        centerable_construct_names=centerable_construct_names,
        baseline_factor_names=baseline_factor_names,
    )
    prior_status = _render_prior_status(
        required_prior_names=required_prior_names,
        optional_prior_names=optional_prior_names,
        authored_priors=authored_priors,
    )
    model_spec_text = (
        "`locked` — ready for prior elicitation"
        if model_spec_locked
        else "`not yet locked` — finish the model decisions above first"
    )

    sections: list[str] = [
        "## Research Question\n\n" + question,
        "## Model Topology\n\n" + format_model_topology(model_topology),
        (f"## Overall Status\n\n- model spec: {model_spec_text}\n{prior_status}"),
        "## Open Model Decisions\n\n" + decision_status,
    ]

    if distribution_cards:
        sections.append(
            "## Distribution Decision Cards\n\n" + format_distribution_cards(distribution_cards)
        )
    if loading_params:
        sections.append("## Loading Orientation\n\n" + format_loading_params(loading_params))
    if construct_scale_cards:
        sections.append(
            "## Construct Scale Cards\n\n" + format_construct_scale_cards(construct_scale_cards)
        )
    if prior_cards:
        sections.append("## Parameter Prior Cards\n\n" + format_prior_cards(prior_cards))

    submission_example = {
        "priors": {
            "<parameter_name>": {
                "parameter": "<parameter_name>",
                "distribution": "<Distribution>",
                "params": {"mu": 0.0, "sigma": 1.0},
                "sources": [],
                "reasoning": "Why this prior is plausible for this parameter.",
            }
        }
    }
    import json as _json

    submission_lines = [
        "## Submission Contract",
        "",
        "`submit_model_configuration` argument object:",
        "",
        "```json",
        _json.dumps(
            {
                "initialization_policy": "stationary",
                "observation_intercept_policy": "free",
                "equilibrium_forcing": False,
                "reasoning": "Rationale for the global configuration.",
            },
            indent=2,
        ),
        "```",
        "",
        "`submit_indicator_choice` argument object (one call per ambiguous indicator):",
        "",
        "```json",
        _json.dumps(
            {
                "variable": "<indicator_name>",
                "distribution": "<Distribution>",
                "link": "<link>",
                "reasoning": "Rationale for the likelihood choice.",
            },
            indent=2,
        ),
        "```",
        "",
        "`submit_prior_block` argument object (any subset of parameters; repeat as needed):",
        "",
        "```json",
        _json.dumps(submission_example, indent=2),
        "```",
    ]
    if include_prior_source_guidance:
        submission_lines.append("")
        submission_lines.append(PRIOR_SOURCE_GUIDANCE.replace("{{", "{").replace("}}", "}"))
    sections.append("\n".join(submission_lines))

    sections.append(
        "## Latest Validator Feedback\n\n" + (latest_feedback or "(no submissions yet)")
    )

    if model_spec_locked and not {
        name for name in required_prior_names if name not in authored_priors
    }:
        sections.append(
            "## Next Step\n\n"
            "All required priors are authored and the model spec is locked. The "
            "validator should report `VALID` after the latest submission. If it does, "
            "stop. Otherwise, act on the validator feedback above."
        )
    elif not model_spec_locked:
        sections.append(
            "## Next Step\n\n"
            "Finish the open model decisions first — the model spec cannot lock until "
            "every ambiguous indicator has a likelihood and the global configuration "
            "has been submitted."
        )
    else:
        sections.append(
            "## Next Step\n\n"
            "The model spec is locked. Elicit priors for the remaining required "
            "parameters via `submit_prior_block`. You may submit any subset per call "
            "and you may resubmit parameters to correct them."
        )
    return _join_sections(sections)
