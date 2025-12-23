"""
DSEM DAG Explorer - Unified Streamlit UI for DAG visualization and structural analysis.

Run with: uv run streamlit run tools/dag_explorer.py
"""

import json

import networkx as nx
import pandas as pd
import streamlit as st
from dowhy import CausalModel
from streamlit_agraph import Config, Edge, Node, agraph

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
    .tag-observed { background: #238636aa; color: #3fb950; }
    .tag-latent { background: #6e768133; color: #8b949e; }
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


# =============================================================================
# Parsing
# =============================================================================


def parse_dag_json(json_str: str) -> tuple[dict | None, str | None]:
    """Parse the DAG JSON format."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    if "dimensions" not in data or "edges" not in data:
        return None, "JSON must have 'dimensions' and 'edges' arrays"

    node_names = {d["name"] for d in data["dimensions"]}
    for edge in data["edges"]:
        if edge["cause"] not in node_names:
            return None, f"Edge references unknown cause: '{edge['cause']}'"
        if edge["effect"] not in node_names:
            return None, f"Edge references unknown effect: '{edge['effect']}'"

    return data, None


def data_to_networkx(data: dict) -> nx.DiGraph:
    """Convert parsed DAG data to NetworkX DiGraph."""
    edge_list = [(e["cause"], e["effect"]) for e in data["edges"]]
    graph = nx.DiGraph(edge_list)
    for dim in data["dimensions"]:
        if dim["name"] not in graph:
            graph.add_node(dim["name"])
    return graph


# =============================================================================
# Visualization
# =============================================================================


def create_agraph_elements(data: dict) -> tuple[list[Node], list[Edge]]:
    """Create agraph nodes and edges from DAG data."""
    nodes = []
    edges = []

    for dim in data["dimensions"]:
        if dim.get("is_outcome"):
            color = COLORS["outcome"]
        elif dim.get("role") == "exogenous":
            color = COLORS["exogenous"]
        else:
            color = COLORS["endogenous"]

        label = dim["name"]
        if dim.get("causal_granularity"):
            label += f"\n({dim['causal_granularity']})"

        is_latent = dim.get("observability") == "latent"
        if is_latent:
            nodes.append(
                Node(
                    id=dim["name"],
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
                    id=dim["name"],
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


def render_dimension_info(dim: dict):
    """Render dimension info as formatted HTML."""
    role = dim.get("role", "endogenous")
    obs = dim.get("observability", "observed")
    is_outcome = dim.get("is_outcome", False)

    tags = f'<span class="tag tag-{role}">{role}</span>'
    tags += f'<span class="tag tag-{obs}">{obs}</span>'
    if is_outcome:
        tags += '<span class="tag tag-outcome">OUTCOME</span>'

    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">{dim['name']}</div>
            <div class="info-row">
                <span class="info-label">Description</span>
                <span class="info-value">{dim.get('description', '—')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Tags</span>
                <span class="info-value">{tags}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Temporal</span>
                <span class="info-value">{dim.get('temporal_status', '—')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Granularity</span>
                <span class="info-value">{dim.get('causal_granularity', '—')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Data Type</span>
                <span class="info-value">{dim.get('measurement_dtype', '—')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Aggregation</span>
                <span class="info-value">{dim.get('aggregation', '—')}</span>
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


# =============================================================================
# Structural Analysis (DoWhy)
# =============================================================================


def check_dag_validity(graph: nx.DiGraph) -> tuple[bool, str]:
    """Check if the graph is a valid DAG (no cycles)."""
    if nx.is_directed_acyclic_graph(graph):
        return True, "Valid DAG (no cycles)"
    else:
        cycles = list(nx.simple_cycles(graph))
        return False, f"Graph has cycles: {cycles[:3]}..."


def find_backdoor_paths(
    graph: nx.DiGraph, treatment: str, outcome: str
) -> list[list[str]]:
    """Find all backdoor paths from treatment to outcome."""
    backdoor_paths = []
    undirected = graph.to_undirected()

    for path in nx.all_simple_paths(undirected, treatment, outcome):
        if len(path) >= 2:
            second_node = path[1]
            if graph.has_edge(second_node, treatment):
                backdoor_paths.append(path)

    return backdoor_paths


def identify_node_types(
    graph: nx.DiGraph, treatment: str, outcome: str
) -> dict[str, list[str]]:
    """Identify confounders, mediators, colliders, and instruments."""
    result = {"confounders": [], "mediators": [], "colliders": [], "instruments": []}

    for node in graph.nodes():
        if node in (treatment, outcome):
            continue

        parents = list(graph.predecessors(node))
        children = list(graph.successors(node))

        if len(parents) >= 2:
            result["colliders"].append(node)

        if treatment in children and outcome in children:
            result["confounders"].append(node)

        treatment_ancestors = nx.ancestors(graph, treatment)
        outcome_ancestors = nx.ancestors(graph, outcome)
        if node in treatment_ancestors and node in outcome_ancestors:
            if node not in result["confounders"]:
                result["confounders"].append(node)

        if nx.has_path(graph, treatment, node) and nx.has_path(graph, node, outcome):
            result["mediators"].append(node)

        if node in treatment_ancestors or graph.has_edge(node, treatment):
            graph_without_treatment = graph.copy()
            graph_without_treatment.remove_node(treatment)
            if not nx.has_path(graph_without_treatment, node, outcome):
                if node not in result["confounders"]:
                    result["instruments"].append(node)

    return result


def get_adjustment_sets(graph: nx.DiGraph, treatment: str, outcome: str) -> list[set]:
    """Get valid adjustment sets using DoWhy."""
    nodes = list(graph.nodes())
    df = pd.DataFrame({node: [0, 1] for node in nodes})

    try:
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=graph,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)

        if identified.backdoor_variables:
            return [set(identified.backdoor_variables)]
        return []
    except Exception as e:
        st.warning(f"DoWhy identification failed: {e}")
        return []


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
        placeholder='{\n  "dimensions": [...],\n  "edges": [...]\n}',
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
            <div style="font-weight: 600; margin-top: 10px; margin-bottom: 4px;">Observability</div>
            <div>▢ Observed (solid box)</div>
            <div>◯ Latent (dashed ellipse)</div>
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
            f"**Nodes:** {len(data['dimensions'])} | **Edges:** {len(data['edges'])} — Click a node to inspect"
        )
    else:
        st.info("Paste DAG JSON to visualize")

with col_info:
    if data:
        tab_inspector, tab_analysis = st.tabs(["Inspector", "Analysis"])

        with tab_inspector:
            selected_node = st.session_state.get("selected_node")

            if selected_node:
                dim = next(
                    (d for d in data["dimensions"] if d["name"] == selected_node), None
                )
                if dim:
                    render_dimension_info(dim)

                    st.markdown("**Connected Edges**")
                    for edge in data["edges"]:
                        if edge["cause"] == selected_node or edge["effect"] == selected_node:
                            render_edge_info(edge)
            else:
                st.info("Click a node to inspect")

        with tab_analysis:
            graph = data_to_networkx(data)
            node_names = [d["name"] for d in data["dimensions"]]

            # Find default outcome
            default_outcome_idx = next(
                (i for i, d in enumerate(data["dimensions"]) if d.get("is_outcome")),
                len(node_names) - 1,
            )

            treatment = st.selectbox("Treatment", node_names, index=0)
            outcome = st.selectbox("Outcome", node_names, index=default_outcome_idx)

            if st.button("Run Structural Checks", type="primary"):
                # DAG validity
                is_valid, validity_msg = check_dag_validity(graph)
                if is_valid:
                    st.success(f"✓ {validity_msg}")
                else:
                    st.error(f"✗ {validity_msg}")

                st.markdown("---")

                # Node classification
                st.markdown("**Node Classification**")
                node_types = identify_node_types(graph, treatment, outcome)
                st.markdown(
                    f"- **Confounders:** {', '.join(node_types['confounders']) or 'None'}"
                )
                st.markdown(
                    f"- **Mediators:** {', '.join(node_types['mediators']) or 'None'}"
                )
                st.markdown(
                    f"- **Colliders:** {', '.join(node_types['colliders']) or 'None'}"
                )
                st.markdown(
                    f"- **Instruments:** {', '.join(node_types['instruments']) or 'None'}"
                )

                st.markdown("---")

                # Backdoor paths
                st.markdown("**Backdoor Paths**")
                backdoor_paths = find_backdoor_paths(graph, treatment, outcome)
                if backdoor_paths:
                    st.warning(f"Found {len(backdoor_paths)} backdoor path(s):")
                    for path in backdoor_paths[:5]:
                        st.markdown(f"- {' → '.join(path)}")
                else:
                    st.success("No backdoor paths")

                st.markdown("---")

                # Adjustment sets
                st.markdown("**Adjustment Sets (DoWhy)**")
                adj_sets = get_adjustment_sets(graph, treatment, outcome)
                if adj_sets:
                    for adj_set in adj_sets:
                        st.info(f"Adjust for: {adj_set}")
                else:
                    st.markdown("No adjustment set identified")
    else:
        st.subheader("Inspector")
        st.info("Load a DAG to explore")
