import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Model Inspector

    Select an evaluation model by index to inspect its full pipeline:
    **Causal DAG** → **Identifiability** → **Functional Specification (LaTeX)**
    """)
    return


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import yaml

    # Add project paths — use absolute paths to avoid __file__ issues
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root / "src") not in sys.path:
        sys.path.insert(0, str(_project_root / "src"))
    if str(_project_root / "tools") not in sys.path:
        sys.path.insert(0, str(_project_root / "tools"))
    return Path, json, yaml


@app.cell
def _(Path, mo, yaml):
    _questions_dir = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions"
    _questions = []
    for _qfile in sorted(_questions_dir.glob("*/question.yaml")):
        with _qfile.open() as _f:
            _data = yaml.safe_load(_f)
        _questions.append({"slug": _qfile.parent.name, "question": _data["question"], "dir": _qfile.parent})

    # marimo dropdown: {display_label: returned_value}
    _options = {f"Q{q['slug'].split('_', 1)[0]}: {q['question']}": q["slug"] for q in _questions}
    _first_label = f"Q{_questions[0]['slug'].split('_', 1)[0]}: {_questions[0]['question']}"

    question_selector = mo.ui.dropdown(
        options=_options,
        value=_first_label,
        label="Evaluation model",
    )
    question_selector
    return (question_selector,)


@app.cell
def _(Path, json, question_selector, yaml):
    _questions_dir = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions"
    _questions = []
    for _qfile in sorted(_questions_dir.glob("*/question.yaml")):
        with _qfile.open() as _f:
            _data = yaml.safe_load(_f)
        _questions.append({"slug": _qfile.parent.name, "question": _data["question"], "dir": _qfile.parent})

    _selected = question_selector.value or _questions[0]["slug"]
    _q = next(q for q in _questions if q["slug"] == _selected)
    _qdir = _q["dir"]

    _cs_path = _qdir / "causal_spec.json"
    try:
        with _cs_path.open() as _f:
            causal_spec = json.load(_f)
    except FileNotFoundError:
        causal_spec = {"latent": {"constructs": [], "edges": []}, "measurement": {"indicators": []}}

    _ms_path = _qdir / "model_spec.json"
    try:
        with _ms_path.open() as _f:
            model_spec = json.load(_f)
    except FileNotFoundError:
        model_spec = None

    question_text = _q["question"]
    question_id = _q["slug"].split("_", 1)[0]
    return causal_spec, model_spec, question_id, question_text


@app.cell
def _(mo, question_id, question_text):
    mo.md(f"""
    ---
    ## Q{question_id}: *"{question_text}"*
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Causal DAG
    """)
    return


@app.cell
def _(causal_spec, mo):
    from pyvis.network import Network as _Network

    def _build_dag_html(_spec: dict) -> str:
        _constructs = _spec.get("latent", {}).get("constructs", [])
        _edges = _spec.get("latent", {}).get("edges", [])
        _indicators = _spec.get("measurement", {}).get("indicators", [])

        _measured = {
            _ind.get("construct_name") for _ind in _indicators
        }

        _edge_set = {(_e["cause"], _e["effect"]) for _e in _edges}
        _feedback = {_p for _p in _edge_set if (_p[1], _p[0]) in _edge_set}

        _net = _Network(
            directed=True,
            cdn_resources="remote",
            height="550px",
            width="100%",
            bgcolor="#0d1117",
            font_color="#c9d1d9",
        )
        _net.set_options("""
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

        _COLORS = {
            "endogenous": "#58a6ff",
            "exogenous": "#f78166",
            "outcome": "#a371f7",
            "edge": "#8b949e",
            "feedback": "#f0883e",
        }

        for _c in _constructs:
            _name = _c["name"]
            if _c.get("is_outcome"):
                _color = _COLORS["outcome"]
            elif _c.get("role") == "exogenous":
                _color = _COLORS["exogenous"]
            else:
                _color = _COLORS["endogenous"]

            _is_unmeasured = _name not in _measured
            _label = _name
            if _c.get("causal_granularity"):
                _label += f"\n({_c['causal_granularity']})"

            _shape = "ellipse" if _is_unmeasured else "box"

            _net.add_node(
                _name,
                label=_label,
                color={"background": _color if not _is_unmeasured else _color + "33", "border": _color},
                shape=_shape,
                borderWidth=2 if _is_unmeasured else 1,
                title=_c.get("description", ""),
                font={"color": "#ffffff", "size": 12},
            )

        for _e in _edges:
            _pair = (_e["cause"], _e["effect"])
            _is_fb = _pair in _feedback
            _net.add_edge(
                _e["cause"],
                _e["effect"],
                color=_COLORS["feedback"] if _is_fb else _COLORS["edge"],
                dashes=_e.get("lagged", False),
                width=1.5,
                smooth={"type": "curvedCW", "roundness": 0.2} if _is_fb else False,
                title=_e.get("description", ""),
            )

        return _net.generate_html()

    _dag_html = _build_dag_html(causal_spec)
    # Escape for srcdoc embedding
    _escaped = _dag_html.replace("&", "&amp;").replace('"', "&quot;")
    mo.Html(
        f'<iframe srcdoc="{_escaped}" '
        f'width="100%" height="580" style="border:none;"></iframe>'
    )
    return


@app.cell
def _(causal_spec, mo):
    _constructs = causal_spec.get("latent", {}).get("constructs", [])
    _indicators = causal_spec.get("measurement", {}).get("indicators", [])
    _measured = {_ind.get("construct_name") for _ind in _indicators}
    _edges = causal_spec.get("latent", {}).get("edges", [])

    _rows = []
    for _c in _constructs:
        _rows.append({
            "Name": _c["name"],
            "Role": _c.get("role", "?"),
            "Temporal": _c.get("temporal_status", "?"),
            "Granularity": _c.get("causal_granularity") or "\u2014",
            "Measured": "yes" if _c["name"] in _measured else "no",
            "Outcome": "yes" if _c.get("is_outcome") else "",
        })

    mo.vstack([
        mo.md(f"**{len(_constructs)} constructs**, **{len(_edges)} edges**, **{len(_indicators)} indicators**"),
        mo.ui.table(_rows, label="Constructs"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Identifiability Analysis
    """)
    return


@app.cell
def _(causal_spec, mo):
    from causal_ssm_agent.utils.effects import get_outcome_from_latent_model as _get_outcome
    from causal_ssm_agent.utils.identifiability import (
        analyze_unobserved_constructs as _analyze,
    )
    from causal_ssm_agent.utils.identifiability import (
        check_identifiability as _check_id,
    )
    from causal_ssm_agent.utils.identifiability import (
        format_identifiability_report as _fmt_id,
    )
    from causal_ssm_agent.utils.identifiability import (
        format_marginalization_report as _fmt_marg,
    )

    _latent = {"constructs": causal_spec["latent"]["constructs"], "edges": causal_spec["latent"]["edges"]}
    _measurement = {"indicators": causal_spec.get("measurement", {}).get("indicators", [])}

    _id_result = _check_id(_latent, _measurement)
    _marg_result = _analyze(_latent, _measurement, _id_result)
    _outcome = _get_outcome(_latent) or "unknown"

    _id_report = _fmt_id(_id_result, _outcome)
    _marg_report = _fmt_marg(_marg_result)

    mo.vstack([
        mo.md(f"```\n{_id_report}\n```"),
        mo.md(f"```\n{_marg_report}\n```"),
    ])
    return


@app.cell
def _(mo, model_spec):
    if model_spec:
        mo.md("## 3. Functional Specification")
    else:
        mo.md("## 3. Functional Specification\n\n*No model spec found for this question. Run eval5 to generate one.*")
    return


@app.cell
def _(causal_spec, mo, model_spec):
    from utils.latex_renderer import model_spec_to_latex as _to_latex

    if model_spec is None:
        latex_sections = None
        mo.md("*No model spec available.*")
    else:
        latex_sections = _to_latex(model_spec, causal_spec)
        _meas_lines = [f"$${_eq}$$\n" for _eq in latex_sections["measurement"]]
        mo.md("### Measurement Model\n\n" + "\n".join(_meas_lines))
    return (latex_sections,)


@app.cell
def _(latex_sections, mo):
    if latex_sections and latex_sections.get("structural"):
        _lines = [f"$${_eq}$$\n" for _eq in latex_sections["structural"]]
        mo.md("### Structural Model (Latent Dynamics)\n\n" + "\n".join(_lines))
    return


@app.cell
def _(latex_sections, mo):
    if latex_sections and latex_sections.get("priors"):
        _role_labels = {
            "fixed_effect": "Fixed Effects",
            "ar_coefficient": "AR Coefficients",
            "residual_sd": "Residual SDs",
            "loading": "Factor Loadings",
            "random_intercept_sd": "Random Intercept SDs",
            "random_slope_sd": "Random Slope SDs",
            "correlation": "Correlations",
        }
        _sections = []
        for _role, _eqs in latex_sections["priors"].items():
            _label = _role_labels.get(_role, _role)
            _lines = [f"#### {_label} ({len(_eqs)})\n"]
            _lines.extend(f"$${_eq}$$\n" for _eq in _eqs)
            _sections.append("\n".join(_lines))

        mo.md("### Priors\n\n" + "\n\n".join(_sections))
    return


@app.cell
def _(mo, model_spec):
    if model_spec:
        from collections import Counter as _Counter

        _params = model_spec.get("parameters", [])
        _liks = model_spec.get("likelihoods", [])
        _role_counts = _Counter(_p["role"] for _p in _params)
        _dist_counts = _Counter(_lik["distribution"] for _lik in _liks)

        mo.md(f"""
    ### Model Summary

    | Property | Value |
    |----------|-------|
    | Model clock | `{model_spec.get('model_clock', '?')}` |
    | Likelihoods | {len(_liks)} |
    | Parameters | {len(_params)} |

    **Distributions**: {', '.join(f'{_d} ({_n})' for _d, _n in _dist_counts.most_common())}

    **Parameters by role**: {', '.join(f'{_r} ({_n})' for _r, _n in _role_counts.most_common())}

    **Reasoning**: {model_spec.get('reasoning', '\u2014')}
        """)
    return


if __name__ == "__main__":
    app.run()
