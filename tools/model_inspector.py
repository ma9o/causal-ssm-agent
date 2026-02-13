import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Model Inspector

        Select an evaluation model by index to inspect its full pipeline:
        **Causal DAG** → **Identifiability** → **Functional Specification (LaTeX)**
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import json
    import sys
    from pathlib import Path

    import yaml

    # Add project paths
    _project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(_project_root / "src"))
    sys.path.insert(0, str(Path(__file__).parent))

    # Load eval config
    _config_path = _project_root / "evals" / "config.yaml"
    with _config_path.open() as _f:
        _config = yaml.safe_load(_f)

    _questions = _config["questions"]
    _options = {str(q["id"]): f"Q{q['id']}: {q['question']}" for q in _questions}

    question_selector = mo.ui.dropdown(
        options=_options,
        value=str(_questions[0]["id"]),
        label="Evaluation model",
    )
    return Path, json, question_selector, sys, yaml


@app.cell
def _(Path, json, question_selector, yaml):
    # Load data files for selected question
    _project_root = Path(__file__).parent.parent
    _data_dir = _project_root / "data"

    _config_path = _project_root / "evals" / "config.yaml"
    with _config_path.open() as _f:
        _config = yaml.safe_load(_f)

    _q = next(q for q in _config["questions"] if str(q["id"]) == question_selector.value)

    # Load causal spec
    _cs_path = _data_dir / _q["dsem"]
    with _cs_path.open() as _f:
        causal_spec = json.load(_f)

    # Load model spec (may not exist yet)
    _ms_path = _data_dir / _q.get("model_spec", f"eval/model_spec{_q['id']}.json")
    try:
        with _ms_path.open() as _f:
            model_spec = json.load(_f)
    except FileNotFoundError:
        model_spec = None

    question_text = _q["question"]
    question_id = _q["id"]
    return causal_spec, model_spec, question_id, question_text


@app.cell
def _(mo, question_id, question_text):
    mo.md(
        f"""
        ---
        ## Q{question_id}: *"{question_text}"*
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 1. Causal DAG")
    return


@app.cell
def _(causal_spec, mo):
    from pyvis.network import Network

    def _build_dag_html(spec: dict) -> str:
        constructs = spec.get("latent", {}).get("constructs", [])
        edges = spec.get("latent", {}).get("edges", [])
        indicators = spec.get("measurement", {}).get("indicators", [])

        measured = {
            ind.get("construct") or ind.get("construct_name") for ind in indicators
        }

        # Detect feedback pairs for curving
        edge_set = {(e["cause"], e["effect"]) for e in edges}
        feedback = {p for p in edge_set if (p[1], p[0]) in edge_set}

        net = Network(
            directed=True,
            cdn_resources="remote",
            height="550px",
            width="100%",
            bgcolor="#0d1117",
            font_color="#c9d1d9",
        )
        net.set_options("""
        {
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 200,
                    "springConstant": 0.01,
                    "nodeDistance": 180
                },
                "solver": "hierarchicalRepulsion"
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": 200,
                    "nodeSpacing": 120
                }
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
                "color": {"inherit": false}
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100
            }
        }
        """)

        COLORS = {
            "endogenous": "#58a6ff",
            "exogenous": "#f78166",
            "outcome": "#a371f7",
            "edge": "#8b949e",
            "feedback": "#f0883e",
        }

        for c in constructs:
            name = c["name"]
            if c.get("is_outcome"):
                color = COLORS["outcome"]
            elif c.get("role") == "exogenous":
                color = COLORS["exogenous"]
            else:
                color = COLORS["endogenous"]

            is_unmeasured = name not in measured
            label = name
            if c.get("causal_granularity"):
                label += f"\n({c['causal_granularity']})"

            shape = "ellipse" if is_unmeasured else "box"

            net.add_node(
                name,
                label=label,
                color={"background": color if not is_unmeasured else color + "33", "border": color},
                shape=shape,
                borderWidth=2 if is_unmeasured else 1,
                title=c.get("description", ""),
                font={"color": "#ffffff", "size": 12},
            )

        for e in edges:
            pair = (e["cause"], e["effect"])
            is_feedback = pair in feedback
            net.add_edge(
                e["cause"],
                e["effect"],
                color=COLORS["feedback"] if is_feedback else COLORS["edge"],
                dashes=e.get("lagged", False),
                width=1.5,
                smooth={"type": "curvedCW", "roundness": 0.2} if is_feedback else False,
                title=e.get("description", ""),
            )

        return net.generate_html()

    dag_html = _build_dag_html(causal_spec)
    mo.Html(f'<iframe srcdoc="{dag_html.replace(chr(34), "&quot;")}" '
            f'width="100%" height="580" frameborder="0"></iframe>')
    return


@app.cell
def _(causal_spec, mo):
    # Summary table of constructs
    _constructs = causal_spec.get("latent", {}).get("constructs", [])
    _indicators = causal_spec.get("measurement", {}).get("indicators", [])
    _measured = {ind.get("construct") or ind.get("construct_name") for ind in _indicators}

    _rows = []
    for _c in _constructs:
        _rows.append({
            "Name": _c["name"],
            "Role": _c.get("role", "?"),
            "Temporal": _c.get("temporal_status", "?"),
            "Granularity": _c.get("causal_granularity") or "—",
            "Measured": "yes" if _c["name"] in _measured else "no",
            "Outcome": "yes" if _c.get("is_outcome") else "",
        })

    mo.md(f"""
    **{len(_constructs)} constructs**, **{len(causal_spec.get('latent', {}).get('edges', []))} edges**, **{len(_indicators)} indicators**
    """)
    mo.ui.table(_rows, label="Constructs")
    return


@app.cell
def _(mo):
    mo.md("## 2. Identifiability Analysis")
    return


@app.cell
def _(causal_spec, mo):
    from dsem_agent.utils.effects import get_outcome_from_latent_model
    from dsem_agent.utils.identifiability import (
        analyze_unobserved_constructs,
        check_identifiability,
        format_identifiability_report,
        format_marginalization_report,
    )

    _latent = {"constructs": causal_spec["latent"]["constructs"], "edges": causal_spec["latent"]["edges"]}
    _measurement = {"indicators": causal_spec.get("measurement", {}).get("indicators", [])}

    id_result = check_identifiability(_latent, _measurement)
    marg_result = analyze_unobserved_constructs(_latent, _measurement, id_result)
    _outcome = get_outcome_from_latent_model(_latent) or "unknown"

    id_report = format_identifiability_report(id_result, _outcome)
    marg_report = format_marginalization_report(marg_result)

    mo.md(f"""
    ```
    {id_report}
    ```

    ```
    {marg_report}
    ```
    """)
    return id_result, id_report, marg_report, marg_result


@app.cell
def _(mo, model_spec):
    mo.md("## 3. Functional Specification" if model_spec else "## 3. Functional Specification\n\n*No model spec found for this question. Run eval5 to generate one.*")
    return


@app.cell
def _(causal_spec, mo, model_spec):
    from utils.latex_renderer import model_spec_to_latex

    if model_spec is None:
        mo.md("*No model spec available.*")
        latex_sections = None
    else:
        latex_sections = model_spec_to_latex(model_spec, causal_spec)

        # Measurement model
        _meas_lines = []
        for _eq in latex_sections["measurement"]:
            _meas_lines.append(f"$${_eq}$$\n")
        mo.md("### Measurement Model\n\n" + "\n".join(_meas_lines))
    return (latex_sections,)


@app.cell
def _(latex_sections, mo):
    if latex_sections and latex_sections.get("structural"):
        _struct_lines = []
        for _eq in latex_sections["structural"]:
            _struct_lines.append(f"$${_eq}$$\n")
        mo.md("### Structural Model (Latent Dynamics)\n\n" + "\n".join(_struct_lines))
    return


@app.cell
def _(latex_sections, mo):
    if latex_sections and latex_sections.get("random_effects"):
        _re_lines = []
        for _eq in latex_sections["random_effects"]:
            _re_lines.append(f"$${_eq}$$\n")
        mo.md("### Random Effects\n\n" + "\n".join(_re_lines))
    return


@app.cell
def _(latex_sections, mo):
    if latex_sections and latex_sections.get("priors"):
        _sections = []
        _role_labels = {
            "fixed_effect": "Fixed Effects",
            "ar_coefficient": "AR Coefficients",
            "residual_sd": "Residual SDs",
            "loading": "Factor Loadings",
            "random_intercept_sd": "Random Intercept SDs",
            "random_slope_sd": "Random Slope SDs",
            "correlation": "Correlations",
        }
        for role, eqs in latex_sections["priors"].items():
            label = _role_labels.get(role, role)
            _lines = [f"#### {label} ({len(eqs)})\n"]
            for _eq in eqs:
                _lines.append(f"$${_eq}$$\n")
            _sections.append("\n".join(_lines))

        mo.md("### Priors\n\n" + "\n\n".join(_sections))
    return


@app.cell
def _(mo, model_spec):
    if model_spec:
        from collections import Counter

        _params = model_spec.get("parameters", [])
        _liks = model_spec.get("likelihoods", [])
        _role_counts = Counter(p["role"] for p in _params)
        _dist_counts = Counter(lik["distribution"] for lik in _liks)

        _summary = f"""
        ### Model Summary

        | Property | Value |
        |----------|-------|
        | Model clock | `{model_spec.get('model_clock', '?')}` |
        | Likelihoods | {len(_liks)} |
        | Parameters | {len(_params)} |
        | Random effects | {len(model_spec.get('random_effects', []))} |

        **Distributions**: {', '.join(f'{d} ({n})' for d, n in _dist_counts.most_common())}

        **Parameters by role**: {', '.join(f'{r} ({n})' for r, n in _role_counts.most_common())}

        **Reasoning**: {model_spec.get('reasoning', '—')}
        """
        mo.md(_summary)
    return


if __name__ == "__main__":
    app.run()
