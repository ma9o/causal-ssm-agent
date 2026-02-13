"""Render a ModelSpec + CausalSpec as LaTeX equations.

Pure utility — no UI dependencies. Returns dicts of LaTeX strings
organised by model section (measurement, structural, priors, random effects).
"""

from __future__ import annotations

# ── Distribution → LaTeX templates ──────────────────────────────────────────
# Each template uses {var}, {construct}, {loading} placeholders.

_MEASUREMENT_TEMPLATES: dict[str, str] = {
    "Normal/identity": (
        r"{var}_t \sim \mathcal{{N}}"
        r"\!\left({loading}\,\eta_{{\text{{{construct}}},t}},\;"
        r"\sigma^2_{{\text{{{var}}}}}\right)"
    ),
    "Bernoulli/logit": (
        r"{var}_t \sim \text{{Bern}}\!\left("
        r"\text{{logit}}^{{-1}}\!\left({loading}\,\eta_{{\text{{{construct}}},t}}\right)\right)"
    ),
    "Bernoulli/probit": (
        r"{var}_t \sim \text{{Bern}}\!\left("
        r"\Phi\!\left({loading}\,\eta_{{\text{{{construct}}},t}}\right)\right)"
    ),
    "Poisson/log": (
        r"{var}_t \sim \text{{Pois}}\!\left("
        r"\exp\!\left({loading}\,\eta_{{\text{{{construct}}},t}}\right)\right)"
    ),
    "NegativeBinomial/log": (
        r"{var}_t \sim \text{{NegBin}}\!\left("
        r"\exp\!\left({loading}\,\eta_{{\text{{{construct}}},t}}\right),\;r_{{\text{{{var}}}}}\right)"
    ),
    "Gamma/log": (
        r"{var}_t \sim \text{{Gamma}}\!\left(\alpha_{{\text{{{var}}}}},\;"
        r"\exp\!\left(-{loading}\,\eta_{{\text{{{construct}}},t}}\right)\right)"
    ),
    "OrderedLogistic/cumulative_logit": (
        r"{var}_t \sim \text{{OrdLogistic}}\!\left("
        r"{loading}\,\eta_{{\text{{{construct}}},t}},\;\mathbf{{c}}_{{\text{{{var}}}}}\right)"
    ),
    "Beta/logit": (
        r"{var}_t \sim \text{{Beta}}\!\left("
        r"\text{{logit}}^{{-1}}\!\left({loading}\,\eta_{{\text{{{construct}}},t}}\right)"
        r"\cdot\phi,\;"
        r"\left(1-\text{{logit}}^{{-1}}\!\left({loading}\,\eta_{{\text{{{construct}}},t}}"
        r"\right)\right)\cdot\phi\right)"
    ),
    "Categorical/softmax": (
        r"{var}_t \sim \text{{Cat}}\!\left("
        r"\text{{softmax}}\!\left({loading}\,\eta_{{\text{{{construct}}},t}}\right)\right)"
    ),
}

# ── Prior templates by role ─────────────────────────────────────────────────

_PRIOR_TEMPLATES: dict[str, str] = {
    "fixed_effect": r"{symbol} \sim \mathcal{{N}}(0,\,1)",
    "ar_coefficient": r"{symbol} \sim \text{{Beta}}(2,\,2)",
    "residual_sd": r"{symbol} \sim \text{{HalfNormal}}(1)",
    "loading": r"{symbol} \sim \text{{HalfNormal}}(1)",
    "random_intercept_sd": r"{symbol} \sim \text{{HalfNormal}}(0.5)",
    "random_slope_sd": r"{symbol} \sim \text{{HalfNormal}}(0.25)",
    "correlation": r"{symbol} \sim \text{{LKJ}}(2)",
}

_ROLE_GREEK: dict[str, str] = {
    "fixed_effect": r"\beta",
    "ar_coefficient": r"\rho",
    "residual_sd": r"\sigma",
    "loading": r"\lambda",
    "random_intercept_sd": r"\tau",
    "random_slope_sd": r"\tau^s",
    "correlation": r"r",
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _tex(name: str) -> str:
    """Sanitise a human name for LaTeX text subscripts."""
    return name.lower().replace("_", r"\_")


def _build_indicator_to_construct(causal_spec: dict) -> dict[str, str]:
    """Map indicator name → construct name from CausalSpec."""
    mapping: dict[str, str] = {}
    for ind in causal_spec.get("measurement", {}).get("indicators", []):
        mapping[ind["name"]] = ind.get("construct_name", "?")
    return mapping


def _build_loading_set(model_spec: dict) -> set[str]:
    """Return set of indicator names that have a free loading parameter."""
    names: set[str] = set()
    for p in model_spec.get("parameters", []):
        if p.get("role") == "loading":
            # loading_<indicator_name>
            raw = p["name"]
            if raw.startswith("loading_"):
                names.add(raw[len("loading_"):])
    return names


def _build_edge_lookup(causal_spec: dict) -> dict[str, list[tuple[str, bool]]]:
    """Map effect construct → list of (cause construct, is_lagged)."""
    edges: dict[str, list[tuple[str, bool]]] = {}
    for e in causal_spec.get("latent", {}).get("edges", []):
        effect = e["effect"]
        edges.setdefault(effect, []).append((e["cause"], e.get("lagged", True)))
    return edges


def _build_param_lookup(model_spec: dict) -> dict[str, list[dict]]:
    """Map role → list of parameter dicts."""
    by_role: dict[str, list[dict]] = {}
    for p in model_spec.get("parameters", []):
        by_role.setdefault(p["role"], []).append(p)
    return by_role


def _snake(name: str) -> str:
    """'Sleep Quality' → 'sleep_quality'."""
    return name.lower().replace(" ", "_")


def _find_beta_param(cause: str, effect: str, params: list[dict]) -> str | None:
    """Find the beta parameter name for a cause→effect edge."""
    cause_s = _snake(cause)
    effect_s = _snake(effect)
    target = f"beta_{cause_s}_{effect_s}"
    for p in params:
        if p["name"] == target:
            return p["name"]
    return None


def _param_symbol(name: str, role: str) -> str:
    """Convert a parameter name to a LaTeX symbol with subscript."""
    greek = _ROLE_GREEK.get(role, name)
    # Strip the role prefix to get the subscript
    prefix_map = {
        "fixed_effect": "beta_",
        "ar_coefficient": "ar_",
        "residual_sd": "residual_sd_",
        "loading": "loading_",
        "random_intercept_sd": "random_intercept_sd_",
        "random_slope_sd": "random_slope_sd_",
        "correlation": "correlation_",
    }
    prefix = prefix_map.get(role, "")
    subscript = name[len(prefix):] if name.startswith(prefix) else name

    if role == "fixed_effect" and "_" in subscript:
        # Try to split cause_effect — find the split point by trying all positions
        # We don't have construct names here, so just use the raw subscript
        subscript_tex = _tex(subscript).replace(r"\_", r" \to ", 1)
        # Find the best split: replace the LAST occurrence that makes sense
        # Heuristic: just render as-is with arrow replacing first underscore group
        return rf"{greek}_{{\text{{{subscript_tex}}}}}"

    return rf"{greek}_{{\text{{{_tex(subscript)}}}}}"


# ── Public API ──────────────────────────────────────────────────────────────


def render_measurement(model_spec: dict, causal_spec: dict | None = None) -> list[str]:
    """Render one LaTeX equation per indicator likelihood."""
    ind_to_construct = _build_indicator_to_construct(causal_spec) if causal_spec else {}
    free_loadings = _build_loading_set(model_spec)

    equations: list[str] = []
    for lik in model_spec.get("likelihoods", []):
        var = lik["variable"]
        dist = lik["distribution"]
        link = lik["link"]
        construct = ind_to_construct.get(var, "?")

        key = f"{dist}/{link}"
        template = _MEASUREMENT_TEMPLATES.get(key)
        if template is None:
            equations.append(rf"\text{{{var}}}: \text{{{dist}}} / \text{{{link}}} (unsupported)")
            continue

        # Reference indicator (loading=1) vs free loading
        if var in free_loadings:
            loading = rf"\lambda_{{\text{{{_tex(var)}}}}}"
        else:
            loading = "1"

        equations.append(template.format(var=_tex(var), construct=_tex(construct), loading=loading))

    return equations


def render_structural(model_spec: dict, causal_spec: dict) -> list[str]:
    """Render one LaTeX equation per endogenous construct."""
    constructs = causal_spec.get("latent", {}).get("constructs", [])
    edge_lookup = _build_edge_lookup(causal_spec)
    params_by_role = _build_param_lookup(model_spec)
    beta_params = params_by_role.get("fixed_effect", [])
    ar_params = {p["name"]: p for p in params_by_role.get("ar_coefficient", [])}

    endogenous_tv = [
        c for c in constructs
        if c.get("role") == "endogenous" and c.get("temporal_status") == "time_varying"
    ]

    equations: list[str] = []
    for c in endogenous_tv:
        name = c["name"]
        name_s = _snake(name)
        lhs = rf"\eta_{{\text{{{_tex(name)}}},t}}"

        terms: list[str] = []

        # AR(1) term
        ar_key = f"ar_{name_s}"
        if ar_key in ar_params:
            terms.append(rf"\rho_{{\text{{{_tex(name)}}}}}\,\eta_{{\text{{{_tex(name)}}},t-1}}")

        # Parent effects
        parents = edge_lookup.get(name, [])
        for cause, is_lagged in parents:
            beta_name = _find_beta_param(cause, name, beta_params)
            if beta_name:
                cause_s = _tex(cause)
                time = "t-1" if is_lagged else "t"
                terms.append(
                    rf"\beta_{{\text{{{_tex(_snake(cause))} \to {_tex(name_s)}}}}}"
                    rf"\,\eta_{{\text{{{cause_s}}},{time}}}"
                )

        # Innovation
        terms.append(rf"\varepsilon_{{\text{{{_tex(name)}}},t}}")

        equations.append(lhs + " = " + " + ".join(terms))

    return equations


def render_priors(model_spec: dict) -> dict[str, list[str]]:
    """Render prior equations grouped by role.

    Returns dict mapping role name → list of LaTeX strings.
    """
    params_by_role = _build_param_lookup(model_spec)
    grouped: dict[str, list[str]] = {}

    for role, params in params_by_role.items():
        template = _PRIOR_TEMPLATES.get(role)
        if template is None:
            continue
        eqs: list[str] = []
        for p in params:
            symbol = _param_symbol(p["name"], role)
            eqs.append(template.format(symbol=symbol))
        if eqs:
            grouped[role] = eqs

    return grouped


def model_spec_to_latex(
    model_spec: dict,
    causal_spec: dict | None = None,
) -> dict[str, list[str] | dict[str, list[str]]]:
    """Convert a ModelSpec (+ optional CausalSpec) to LaTeX.

    Returns:
        {
            "measurement": list[str],      # one equation per indicator
            "structural": list[str],       # one equation per endogenous construct (requires causal_spec)
            "priors": dict[str, list[str]], # role → list of prior equations
        }
    """
    result: dict[str, list[str] | dict[str, list[str]]] = {
        "measurement": render_measurement(model_spec, causal_spec),
        "structural": render_structural(model_spec, causal_spec) if causal_spec else [],
        "priors": render_priors(model_spec),
    }
    return result
