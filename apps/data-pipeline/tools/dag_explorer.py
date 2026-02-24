"""
DAG Explorer - Streamlit UI for DAG visualization.

Run with: uv run streamlit run tools/dag_explorer.py
"""

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import Config, Edge, Node, agraph

# Add src to path for causal_ssm_agent imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from commons import COPY_GRAPH_HTML, parse_dag_json
from dag_diagnostics import DagDiagnostics, run_diagnostics

st.set_page_config(page_title="DAG Explorer", layout="wide")

# Dark theme CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; }
    .stTextArea textarea {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 12px;
        background-color: #161b22;
        color: #c9d1d9;
    }
    h1, h2, h3 { color: #f0f6fc; }
    .info-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .info-title {
        color: #f0f6fc;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .info-row {
        display: flex;
        margin-bottom: 6px;
    }
    .info-label {
        color: #8b949e;
        font-size: 11px;
        text-transform: uppercase;
        min-width: 100px;
    }
    .info-value {
        color: #c9d1d9;
        font-size: 13px;
    }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
        margin-right: 4px;
    }
    .tag-outcome { background: #8957e5aa; color: #d2a8ff; }
    .tag-endogenous { background: #1f6feb33; color: #58a6ff; }
    .tag-exogenous { background: #da3633aa; color: #ff7b72; }
    .tag-measured { background: #238636aa; color: #3fb950; }
    .tag-unmeasured { background: #6e768133; color: #8b949e; }
    .tag-marginalize { background: #238636aa; color: #3fb950; }
    .tag-needsmodel { background: #da3633aa; color: #ff7b72; }
    .indicator-box {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .indicator-name {
        color: #7ee787;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .indicator-detail {
        color: #8b949e;
        font-size: 11px;
        margin-bottom: 4px;
    }
    .indicator-detail strong {
        color: #c9d1d9;
    }
    .empty-state {
        color: #8b949e;
        font-style: italic;
        font-size: 12px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = {
    "endogenous": "#58a6ff",
    "exogenous": "#f78166",
    "outcome": "#a371f7",
    "edge": "#8b949e",
    "feedback": "#f0883e",  # Orange - feedback loop edges
    "background": "#0d1117",
    "can_marginalize": "#3fb950",  # Green - safe to ignore
    "needs_modeling": "#f85149",  # Red - blocks identification
}


def create_agraph_elements(
    data: dict,
    marg_analysis: dict | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Create agraph nodes and edges from DAG data.

    Expects normalized data with 'constructs', 'edges', and 'indicators' keys.
    Constructs without indicators are shown as "unmeasured" (dashed ellipse).
    If marg_analysis is provided, unmeasured nodes are colored based on whether
    they can be marginalized (green) or need explicit modeling (red).
    """
    nodes = []
    edges = []

    # Detect feedback loops (edges in both directions between same node pair)
    # so we can curve them to avoid visual overlap
    edge_pairs = set()
    reciprocal_pairs = set()
    for edge in data.get("edges", []):
        pair = (edge["cause"], edge["effect"])
        reverse = (edge["effect"], edge["cause"])
        if reverse in edge_pairs:
            reciprocal_pairs.add(pair)
            reciprocal_pairs.add(reverse)
        edge_pairs.add(pair)

    # Compute which constructs have indicators (derived observability)
    measured_constructs = {ind.get("construct_name") for ind in data.get("indicators", [])}

    # Extract marginalization sets
    can_marginalize = marg_analysis.get("can_marginalize", set()) if marg_analysis else set()
    blocking_details = marg_analysis.get("blocking_details", {}) if marg_analysis else {}
    needs_modeling = set(blocking_details.keys())

    for construct in data["constructs"]:
        name = construct["name"]

        # Determine base color by role
        if construct.get("is_outcome"):
            color = COLORS["outcome"]
        elif construct.get("role") == "exogenous":
            color = COLORS["exogenous"]
        else:
            color = COLORS["endogenous"]

        label = name
        if construct.get("temporal_scale"):
            label += f"\n({construct['temporal_scale']})"

        # A construct is "unmeasured" if it has no indicators
        is_unmeasured = name not in measured_constructs

        if is_unmeasured:
            # Override color based on marginalization status
            if name in needs_modeling:
                border_color = COLORS["needs_modeling"]
                label += "\n⚠ needs proxy"
            elif name in can_marginalize:
                border_color = COLORS["can_marginalize"]
                label += "\n✓ can marginalize"
            else:
                border_color = color

            nodes.append(
                Node(
                    id=name,
                    label=label,
                    color={
                        "background": border_color + "33",
                        "border": border_color,
                        "highlight": {"background": border_color, "border": "#f0f6fc"},
                    },
                    borderWidth=3 if name in needs_modeling else 2,
                    borderWidthSelected=4,
                    shapeProperties={"borderDashes": [5, 5]},
                    font={"color": "#ffffff"},
                    shape="ellipse",
                )
            )
        else:
            nodes.append(
                Node(
                    id=name,
                    label=label,
                    color={
                        "background": color,
                        "border": "#30363d",
                        "highlight": {"background": color, "border": "#f0f6fc"},
                    },
                    borderWidth=1,
                    font={"color": "#ffffff"},
                    shape="box",
                )
            )

    for edge in data["edges"]:
        pair = (edge["cause"], edge["effect"])
        is_feedback = pair in reciprocal_pairs

        # Use curved edges for feedback loops so they don't overlap
        # curvedCW for one direction, curvedCCW for the other
        if is_feedback:
            # Use clockwise curve; the reverse edge will also use CW
            # but vis.js will curve them in opposite directions automatically
            smooth = {"enabled": True, "type": "curvedCW", "roundness": 0.2}
            edge_color = COLORS["feedback"]
        else:
            smooth = {"enabled": False}
            edge_color = COLORS["edge"]

        edges.append(
            Edge(
                source=edge["cause"],
                target=edge["effect"],
                color=edge_color,
                dashes=edge.get("lagged", False),
                width=1.5,
                smooth=smooth,
            )
        )

    return nodes, edges


def render_construct_info(
    construct: dict,
    is_measured: bool,
    marg_status: str | None = None,
):
    """Render construct info as formatted HTML.

    Args:
        construct: Construct dictionary
        is_measured: Whether this construct has at least one indicator
        marg_status: "can_marginalize", "needs_modeling", or None
    """
    role = construct.get("role", "endogenous")
    is_outcome = construct.get("is_outcome", False)

    tags = f'<span class="tag tag-{role}">{role}</span>'
    measurement_status = "measured" if is_measured else "unmeasured"
    tags += f'<span class="tag tag-{measurement_status}">{measurement_status}</span>'
    if is_outcome:
        tags += '<span class="tag tag-outcome">OUTCOME</span>'
    if marg_status == "can_marginalize":
        tags += '<span class="tag tag-marginalize">can marginalize</span>'
    elif marg_status == "needs_modeling":
        tags += '<span class="tag tag-needsmodel">needs proxy</span>'

    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">{construct["name"]}</div>
            <div class="info-row">
                <span class="info-label">Description</span>
                <span class="info-value">{construct.get("description", "—")}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Tags</span>
                <span class="info-value">{tags}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Temporal</span>
                <span class="info-value">{construct.get("temporal_status", "—")}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Granularity</span>
                <span class="info-value">{construct.get("temporal_scale", "—")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_edge_info(edge: dict):
    """Render edge info as formatted HTML."""
    timing = "Lagged (t-1 → t)" if edge.get("lagged") else "Contemporaneous (t → t)"

    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">{edge["cause"]} → {edge["effect"]}</div>
            <div class="info-row">
                <span class="info-label">Description</span>
                <span class="info-value">{edge.get("description", "—")}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Timing</span>
                <span class="info-value">{timing}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_indicator_info(indicator: dict):
    """Render indicator info as formatted HTML."""
    construct = indicator.get("construct_name", "—")
    dtype = indicator.get("measurement_dtype", "—")
    aggregation = indicator.get("aggregation", "—")
    how_to_measure = indicator.get("how_to_measure", "—")

    st.markdown(
        f"""
        <div class="indicator-box">
            <div class="indicator-name">{indicator.get("name", "unnamed")}</div>
            <div class="indicator-detail"><strong>Construct:</strong> {construct}</div>
            <div class="indicator-detail"><strong>Type:</strong> {dtype}</div>
            <div class="indicator-detail"><strong>Aggregation:</strong> {aggregation}</div>
            <div class="indicator-detail"><strong>How to measure:</strong> {how_to_measure}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_indicators_for_construct(indicators: list[dict], construct_name: str) -> list[dict]:
    """Get all indicators that measure a given construct."""
    return [ind for ind in indicators if (ind.get("construct_name")) == construct_name]


# =============================================================================
# Main UI
# =============================================================================

st.title("DAG Explorer")

col_input, col_graph, col_info = st.columns([1, 2, 1])

with col_input:
    st.subheader("Input")
    json_input = st.text_area(
        "Paste DAG JSON",
        height=350,
        placeholder='{\n  "latent": {\n    "constructs": [...],\n    "edges": [...]\n  },\n  "measurement": {\n    "indicators": [...]\n  }\n}',
    )

    st.markdown("---")
    st.markdown("**Legend**")
    st.markdown(
        """
        <div style="font-size: 12px; color: #8b949e;">
            <div style="font-weight: 600; margin-bottom: 4px;">Role</div>
            <div><span style="color: #a371f7;">■</span> Outcome</div>
            <div><span style="color: #58a6ff;">■</span> Endogenous</div>
            <div><span style="color: #f78166;">■</span> Exogenous</div>
            <div style="font-weight: 600; margin-top: 10px; margin-bottom: 4px;">Measurement</div>
            <div>▢ Has indicators (solid box)</div>
            <div>◯ No indicators (dashed ellipse)</div>
            <div style="font-weight: 600; margin-top: 10px; margin-bottom: 4px;">Identifiability Status</div>
            <div><span style="color: #f85149;">◯</span> Needs proxy (blocks ID)</div>
            <div><span style="color: #3fb950;">◯</span> Can marginalize</div>
            <div style="font-weight: 600; margin-top: 10px; margin-bottom: 4px;">Edges</div>
            <div><span style="color: #8b949e;">—</span> Contemporaneous (t→t)</div>
            <div><span style="color: #8b949e;">┅</span> Lagged (t-1→t)</div>
            <div><span style="color: #f0883e;">⟳</span> Feedback loop (curved, orange)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

data, error = None, None
if json_input.strip():
    data, error = parse_dag_json(json_input)

with col_graph:
    st.subheader("Graph")
    if error:
        st.error(error)
    elif data:
        # Pass marginalization analysis if available for color coding
        diagnostics: DagDiagnostics | None = st.session_state.get("dag_diagnostics")
        marg_analysis = diagnostics.marginalization if diagnostics else None
        nodes, edges = create_agraph_elements(data, marg_analysis)

        config = Config(
            width="100%",
            height=600,
            directed=True,
            hierarchical=True,
            levelSeparation=120,
            nodeSpacing=150,
            treeSpacing=200,
            physics=False,
            nodeHighlightBehavior=True,
            highlightColor="#f0f6fc",
            collapsible=False,
        )

        selected = agraph(nodes=nodes, edges=edges, config=config)

        if selected:
            st.session_state["selected_node"] = selected

        st.caption(
            f"**Constructs:** {len(data['constructs'])} | **Edges:** {len(data['edges'])} | **Indicators:** {len(data['indicators'])} — Click a node to inspect"
        )

        # Copy button
        col_stats, col_copy = st.columns([4, 1])
        with col_copy:
            components.html(COPY_GRAPH_HTML, height=35)

        # y0-based Identifiability Analysis Section
        st.markdown("---")
        st.subheader("Causal Identifiability (y0)")

        if st.button("Run Identifiability Analysis", type="primary"):
            with st.spinner("Analyzing with y0 ID algorithm..."):
                try:
                    diagnostics = run_diagnostics(data)
                    st.session_state["dag_diagnostics"] = diagnostics
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.session_state.pop("dag_diagnostics", None)

        # Display results if available
        diagnostics = st.session_state.get("dag_diagnostics")
        if diagnostics:
            id_result = diagnostics.identifiability
            marg_analysis = diagnostics.marginalization

            outcome = diagnostics.graph_summary.get("outcome", "outcome")

            # Summary status
            non_identifiable = id_result.get("non_identifiable_treatments", {})
            identifiable = id_result.get("identifiable_treatments", {})
            all_identifiable = len(non_identifiable) == 0
            if all_identifiable:
                st.success(f"All treatment effects on {outcome} are identifiable!")
            else:
                n_non_id = len(non_identifiable)
                n_total = n_non_id + len(identifiable)
                st.warning(f"{n_non_id}/{n_total} treatments have non-identifiable effects")

            # Identifiable treatments
            if identifiable:
                with st.expander(f"✓ Identifiable Treatments ({len(identifiable)})"):
                    for treatment in sorted(identifiable.keys()):
                        details = identifiable[treatment]
                        method = details.get("method", "unknown")
                        estimand = details.get("estimand", "")
                        st.markdown(f"**{treatment}** — via {method}")
                        if estimand:
                            st.code(estimand, language="text")

            # Non-identifiable treatments
            if non_identifiable:
                with st.expander(f"✗ Non-identifiable Treatments ({len(non_identifiable)})"):
                    for treatment in sorted(non_identifiable.keys()):
                        details = non_identifiable[treatment]
                        blockers = (
                            details.get("confounders", []) if isinstance(details, dict) else []
                        )
                        notes = details.get("notes") if isinstance(details, dict) else None
                        if blockers:
                            st.markdown(f"**{treatment}** — blocked by: {', '.join(blockers)}")
                        elif notes:
                            st.markdown(f"**{treatment}** — {notes}")
                        else:
                            st.markdown(f"**{treatment}** — structural non-identifiability")

            # Marginalization Analysis
            st.markdown("---")
            st.markdown("**Unobserved Construct Analysis**")

            can_marg = marg_analysis.get("can_marginalize", set())
            blocking_details = marg_analysis.get("blocking_details", {})
            needs_model = set(blocking_details.keys())

            if can_marg:
                st.markdown(
                    f'<div class="info-box">'
                    f'<div class="info-title" style="color: #3fb950;">✓ Can Marginalize ({len(can_marg)})</div>'
                    f'<div style="color: #8b949e; font-size: 11px; margin-bottom: 8px;">'
                    f"These can be omitted from model spec - effects absorbed into error terms</div>",
                    unsafe_allow_html=True,
                )
                for u in sorted(can_marg):
                    reason = marg_analysis.get("marginalize_reason", {}).get(u, "")
                    st.markdown(f"- **{u}**: {reason}")
                st.markdown("</div>", unsafe_allow_html=True)

            if needs_model:
                st.markdown(
                    f'<div class="info-box">'
                    f'<div class="info-title" style="color: #f85149;">✗ Needs Modeling ({len(needs_model)})</div>'
                    f'<div style="color: #8b949e; font-size: 11px; margin-bottom: 8px;">'
                    f"These block identification - need proxies or explicit latent variables</div>",
                    unsafe_allow_html=True,
                )
                for u in sorted(needs_model):
                    reason = ""
                    treatments = blocking_details.get(u, [])
                    if treatments:
                        reason = f"blocks identification of: {', '.join(treatments)}"
                    st.markdown(f"- **{u}**: {reason}")
                st.markdown("</div>", unsafe_allow_html=True)

            if not can_marg and not needs_model:
                st.info("All constructs are observed - no marginalization analysis needed")

            # Graph info
            with st.expander("Graph Info"):
                info = id_result["graph_info"]
                st.markdown(
                    f"- **Observed:** {len(info['observed_constructs'])}/{info['total_constructs']} constructs"
                )
                st.markdown(f"- **Directed edges:** {info['n_directed_edges']}")
                if info["unobserved_confounders"]:
                    st.markdown(
                        f"- **Unobserved confounders:** {', '.join(info['unobserved_confounders'])}"
                    )
    else:
        st.info("Paste DAG JSON to visualize")

with col_info:
    st.subheader("Inspector")
    if data:
        selected_node = st.session_state.get("selected_node")
        # Compute measured constructs from indicators
        measured_constructs = {ind.get("construct_name") for ind in data.get("indicators", [])}

        if selected_node:
            construct = next((c for c in data["constructs"] if c["name"] == selected_node), None)
            if construct:
                # Determine marginalization status
                diagnostics = st.session_state.get("dag_diagnostics")
                marg_analysis = diagnostics.marginalization if diagnostics else {}
                can_marg = marg_analysis.get("can_marginalize", set())
                needs_model = set(marg_analysis.get("blocking_details", {}).keys())
                if selected_node in can_marg:
                    marg_status = "can_marginalize"
                elif selected_node in needs_model:
                    marg_status = "needs_modeling"
                else:
                    marg_status = None

                render_construct_info(construct, selected_node in measured_constructs, marg_status)

                # Show indicators for this construct
                st.markdown("**Indicators**")
                construct_indicators = get_indicators_for_construct(
                    data["indicators"], selected_node
                )
                if construct_indicators:
                    for indicator in construct_indicators:
                        render_indicator_info(indicator)
                else:
                    st.markdown(
                        '<div class="empty-state">No indicators defined</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("**Connected Edges**")
                connected_edges = [
                    edge
                    for edge in data["edges"]
                    if edge["cause"] == selected_node or edge["effect"] == selected_node
                ]
                if connected_edges:
                    for edge in connected_edges:
                        render_edge_info(edge)
                else:
                    st.markdown(
                        '<div class="empty-state">No connected edges</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Click a node to inspect")

        # Show measurement model summary
        st.markdown("---")
        st.markdown("**Measurement Model**")
        if data["indicators"]:
            # Group by construct
            by_construct: dict[str, list[dict]] = {}
            for ind in data["indicators"]:
                c_name = ind.get("construct_name", "unknown")
                by_construct.setdefault(c_name, []).append(ind)

            for c_name, inds in sorted(by_construct.items()):
                st.markdown(f"**{c_name}** ({len(inds)} indicator{'s' if len(inds) > 1 else ''})")
                for ind in inds:
                    dtype = ind.get("measurement_dtype", "?")
                    st.markdown(f"  - `{ind.get('name')}` ({dtype})")
        else:
            st.markdown(
                '<div class="empty-state">No measurement model defined</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Load a DAG to explore")
