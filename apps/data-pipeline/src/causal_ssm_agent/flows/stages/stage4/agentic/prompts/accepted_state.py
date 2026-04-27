"""Shared renderers for accepted Stage 4 state shown at prompt seed time."""

from __future__ import annotations

from typing import Any


def render_locked_model_spec(
    model_spec: dict[str, Any],
    *,
    centerable_construct_names: tuple[str, ...],
    baseline_factor_names: tuple[str, ...],
) -> str:
    """Render the locked ``model_spec`` artifact as a prompt section.

    Shown whenever the accepted locked model spec already exists so the agent keeps
    the authoritative view of indicator likelihoods, the parameter inventory, and
    the global configuration in context.
    """
    init_text = f"`{model_spec.get('initialization_policy') or 'unset'}`"
    obs_text = f"`{model_spec.get('observation_intercept_policy') or 'unset'}`"
    equilibrium = model_spec.get("equilibrium_forcing")
    forcing_text = "(unset)" if equilibrium is None else f"`{str(bool(equilibrium)).lower()}`"
    centerable = ", ".join(f"`{name}`" for name in centerable_construct_names) or "(none)"
    baseline = ", ".join(f"`{name}`" for name in baseline_factor_names) or "(none)"

    lines: list[str] = [
        "### Global Configuration",
        "",
        f"- `initialization_policy`: {init_text}",
        f"- `observation_intercept_policy`: {obs_text}",
        f"- `equilibrium_forcing`: {forcing_text}",
        f"- centered-indicator constructs identifying latent baselines: {centerable}",
        f"- compiled baseline-factor scales from marginalized confounders: {baseline}",
    ]

    likelihoods = [
        item for item in (model_spec.get("likelihoods") or []) if isinstance(item, dict)
    ]
    if likelihoods:
        lines.extend(["", "### Indicator Likelihoods", ""])
        for item in likelihoods:
            variable = item.get("indicator") or item.get("variable") or "?"
            distribution = item.get("distribution") or "?"
            link = item.get("link") or "?"
            lines.append(f"- `{variable}`: `{distribution}` / `{link}`")

    parameters = [
        param
        for param in (model_spec.get("parameters") or [])
        if isinstance(param, dict) and isinstance(param.get("name"), str)
    ]
    if parameters:
        lines.extend(["", "### Parameters (by role)", ""])
        grouped: dict[str, list[dict[str, Any]]] = {}
        for param in parameters:
            role = str(param.get("role") or "other")
            grouped.setdefault(role, []).append(param)
        for role in sorted(grouped):
            lines.append(f"**{role}**")
            for param in grouped[role]:
                name = param["name"]
                constraint = param.get("constraint")
                description = param.get("description") or ""
                constraint_text = f" [constraint=`{constraint}`]" if constraint else ""
                description_text = f" — {description}" if description else ""
                lines.append(f"- `{name}`{constraint_text}{description_text}")
            lines.append("")
    return "\n".join(lines).rstrip()


def render_authored_priors(authored_priors: dict[str, dict[str, Any]]) -> str:
    """Render the full values of every authored prior."""
    if not authored_priors:
        return ""
    import json as _json

    lines: list[str] = []
    for name in sorted(authored_priors):
        prior = authored_priors[name]
        if not isinstance(prior, dict):
            continue
        distribution = prior.get("distribution") or "?"
        params = prior.get("params") or {}
        params_text = _json.dumps(params, default=str) if isinstance(params, dict) else str(params)
        reasoning = (prior.get("reasoning") or "").strip()
        sources = prior.get("sources") or []
        header = f"- `{name}`: `{distribution}`({params_text})"
        lines.append(header)
        if reasoning:
            lines.append(f"  - reasoning: {reasoning}")
        if isinstance(sources, list) and sources:
            src_text = "; ".join(
                (s.get("citation") or s.get("url") or str(s)) if isinstance(s, dict) else str(s)
                for s in sources[:3]
            )
            more = f" (+{len(sources) - 3} more)" if len(sources) > 3 else ""
            lines.append(f"  - sources: {src_text}{more}")
    return "\n".join(lines)


def build_accepted_state_sections(
    *,
    accepted_model_spec: dict[str, Any] | None,
    authored_priors: dict[str, dict[str, Any]],
    centerable_construct_names: tuple[str, ...],
    baseline_factor_names: tuple[str, ...],
) -> list[str]:
    """Build the shared accepted-state prompt sections for Stage 4 seed context."""
    sections: list[str] = []
    if accepted_model_spec is not None:
        sections.append(
            "## Accepted Locked Model Spec\n\n"
            + render_locked_model_spec(
                accepted_model_spec,
                centerable_construct_names=centerable_construct_names,
                baseline_factor_names=baseline_factor_names,
            )
        )
    if authored_priors:
        sections.append("## Accepted Authored Priors\n\n" + render_authored_priors(authored_priors))
    return sections
