"""
DSEM DAG Explorer - Streamlit UI for DAG visualization.

Run with: uv run streamlit run tools/dag_explorer.py
"""

import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import Config, Edge, Node, agraph

from commons import (
    COPY_GRAPH_HTML,
    dag_to_networkx,
    parse_dag_json,
    run_identify_effect,
)

st.set_page_config(page_title="DSEM DAG Explorer", layout="wide")

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
    "background": "#0d1117",
}


def create_agraph_elements(data: dict) -> tuple[list[Node], list[Edge]]:
    """Create agraph nodes and edges from DAG data.

    Expects normalized data with 'constructs', 'edges', and 'indicators' keys.
    Constructs without indicators are shown as "unmeasured" (dashed ellipse).
    """
    nodes = []
    edges = []

    # Compute which constructs have indicators (derived observability)
    measured_constructs = {
        ind.get("construct") or ind.get("construct_name")
        for ind in data.get("indicators", [])
    }

    for construct in data["constructs"]:
        if construct.get("is_outcome"):
            color = COLORS["outcome"]
        elif construct.get("role") == "exogenous":
            color = COLORS["exogenous"]
        else:
            color = COLORS["endogenous"]

        label = construct["name"]
        if construct.get("causal_granularity"):
            label += f"\n({construct['causal_granularity']})"

        # A construct is "unmeasured" if it has no indicators
        is_unmeasured = construct["name"] not in measured_constructs
        if is_unmeasured:
            nodes.append(
                Node(
                    id=construct["name"],
                    label=label,
                    color={
                        "background": color + "66",
                        "border": color,
                        "highlight": {"background": color, "border": "#f0f6fc"},
                    },
                    borderWidth=2,
                    borderWidthSelected=3,
                    shapeProperties={"borderDashes": [5, 5]},
                    font={"color": "#ffffff"},
                    shape="ellipse",
                )
            )
        else:
            nodes.append(
                Node(
                    id=construct["name"],
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
        edges.append(
            Edge(
                source=edge["cause"],
                target=edge["effect"],
                color=COLORS["edge"],
                dashes=edge.get("lagged", False),
                width=1.5,
            )
        )

    return nodes, edges


def render_construct_info(construct: dict, is_measured: bool):
    """Render construct info as formatted HTML.

    Args:
        construct: Construct dictionary
        is_measured: Whether this construct has at least one indicator
    """
    role = construct.get("role", "endogenous")
    is_outcome = construct.get("is_outcome", False)

    tags = f'<span class="tag tag-{role}">{role}</span>'
    measurement_status = "measured" if is_measured else "unmeasured"
    tags += f'<span class="tag tag-{measurement_status}">{measurement_status}</span>'
    if is_outcome:
        tags += '<span class="tag tag-outcome">OUTCOME</span>'

    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">{construct['name']}</div>
            <div class="info-row">
                <span class="info-label">Description</span>
                <span class="info-value">{construct.get('description', '—')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Tags</span>
                <span class="info-value">{tags}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Temporal</span>
                <span class="info-value">{construct.get('temporal_status', '—')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Granularity</span>
                <span class="info-value">{construct.get('causal_granularity', '—')}</span>
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
            <div class="info-title">{edge['cause']} → {edge['effect']}</div>
            <div class="info-row">
                <span class="info-label">Description</span>
                <span class="info-value">{edge.get('description', '—')}</span>
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
    construct = indicator.get("construct") or indicator.get("construct_name", "—")
    dtype = indicator.get("measurement_dtype", "—")
    granularity = indicator.get("measurement_granularity", "—")
    aggregation = indicator.get("aggregation", "—")
    how_to_measure = indicator.get("how_to_measure", "—")

    st.markdown(
        f"""
        <div class="indicator-box">
            <div class="indicator-name">{indicator.get('name', 'unnamed')}</div>
            <div class="indicator-detail"><strong>Construct:</strong> {construct}</div>
            <div class="indicator-detail"><strong>Type:</strong> {dtype} @ {granularity}</div>
            <div class="indicator-detail"><strong>Aggregation:</strong> {aggregation}</div>
            <div class="indicator-detail"><strong>How to measure:</strong> {how_to_measure}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_indicators_for_construct(indicators: list[dict], construct_name: str) -> list[dict]:
    """Get all indicators that measure a given construct."""
    return [
        ind for ind in indicators
        if (ind.get("construct") or ind.get("construct_name")) == construct_name
    ]


# =============================================================================
# Main UI
# =============================================================================

st.title("DSEM DAG Explorer")

col_input, col_graph, col_info = st.columns([1, 2, 1])

with col_input:
    st.subheader("Input")
    json_input = st.text_area(
        "Paste DAG JSON",
        height=350,
        placeholder='{\n  "structural": {\n    "constructs": [...],\n    "edges": [...]\n  },\n  "measurement": {\n    "indicators": [...]\n  }\n}',
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
            <div style="font-weight: 600; margin-top: 10px; margin-bottom: 4px;">Edges</div>
            <div>— Contemporaneous</div>
            <div>┅ Lagged</div>
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
        nodes, edges = create_agraph_elements(data)

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

        # DoWhy Causal Analysis Section
        st.markdown("---")
        st.subheader("Causal Identifiability (DoWhy)")

        node_names = [c["name"] for c in data["constructs"]]
        # Find the marked outcome node (if any)
        default_outcome = next(
            (c["name"] for c in data["constructs"] if c.get("is_outcome")), None
        )

        col_treat, col_out = st.columns(2)

        with col_treat:
            treatment = st.selectbox("Treatment", options=node_names, index=0)
        with col_out:
            outcome_options = [n for n in node_names if n != treatment]
            outcome_idx = (
                outcome_options.index(default_outcome)
                if default_outcome and default_outcome in outcome_options
                else 0
            )
            outcome = st.selectbox("Outcome", options=outcome_options, index=outcome_idx)

        if st.button("Run Identifiability Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                graph = dag_to_networkx(data)
                # Constructs are "observed" if they have at least one indicator
                measured_construct_names = {
                    ind.get("construct") or ind.get("construct_name")
                    for ind in data.get("indicators", [])
                }
                observed_nodes = [
                    c["name"]
                    for c in data["constructs"]
                    if c["name"] in measured_construct_names
                ]
                result = run_identify_effect(graph, treatment, outcome, observed_nodes)

                st.markdown("**Identification Result**")
                if result.error:
                    st.error(f"Analysis failed: {result.error}")
                elif result.identifiable:
                    st.success("Effect is identifiable!")
                    st.markdown(
                        f"""
                        <div class="info-box">
                            <div class="info-row">
                                <span class="info-label">Method</span>
                                <span class="info-value">{result.method.title() if result.method else 'Unknown'}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if result.backdoor_vars:
                        st.markdown(f"**Backdoor variables:** {', '.join(result.backdoor_vars)}")
                    if result.frontdoor_vars:
                        st.markdown(f"**Frontdoor variables:** {', '.join(result.frontdoor_vars)}")
                    if result.iv_vars:
                        st.markdown(f"**Instrumental variables:** {', '.join(result.iv_vars)}")

                    with st.expander("Full Estimand Details"):
                        st.text(result.estimand_str)
                else:
                    st.warning("Effect may not be identifiable with standard methods.")
                    with st.expander("Details"):
                        st.text(result.estimand_str)
    else:
        st.info("Paste DAG JSON to visualize")

with col_info:
    st.subheader("Inspector")
    if data:
        selected_node = st.session_state.get("selected_node")
        # Compute measured constructs from indicators
        measured_constructs = {
            ind.get("construct") or ind.get("construct_name")
            for ind in data.get("indicators", [])
        }

        if selected_node:
            construct = next(
                (c for c in data["constructs"] if c["name"] == selected_node), None
            )
            if construct:
                render_construct_info(construct, selected_node in measured_constructs)

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
                    edge for edge in data["edges"]
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
                c_name = ind.get("construct") or ind.get("construct_name", "unknown")
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
