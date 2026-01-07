"""
Interactive DAG Builder with DoWhy Identifiability Analysis.

Run with: uv run streamlit run tools/dag_identifiability.py
"""

import json
from io import BytesIO

import networkx as nx
import streamlit as st
from dowhy import CausalModel
from streamlit_agraph import Config, Edge, Node, agraph

st.set_page_config(page_title="DAG Builder + DoWhy", layout="wide")

# Dark theme CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; }
    h1, h2, h3 { color: #f0f6fc; }
    .result-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
        margin: 8px 0;
    }
    .result-success {
        border-left: 4px solid #238636;
    }
    .result-warning {
        border-left: 4px solid #d29922;
    }
    .result-error {
        border-left: 4px solid #da3633;
    }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
        margin-right: 4px;
    }
    .tag-observed { background: #238636aa; color: #3fb950; }
    .tag-unobserved { background: #6e768166; color: #8b949e; }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = {
    "observed": "#58a6ff",
    "unobserved": "#8b949e",
    "treatment": "#3fb950",
    "outcome": "#a371f7",
    "edge": "#8b949e",
}


def init_session_state():
    """Initialize session state for nodes and edges."""
    if "nodes" not in st.session_state:
        st.session_state.nodes = {}  # name -> {"observed": bool}
    if "edges" not in st.session_state:
        st.session_state.edges = []  # list of (cause, effect)
    if "treatment" not in st.session_state:
        st.session_state.treatment = None
    if "outcome" not in st.session_state:
        st.session_state.outcome = None


def add_node(name: str, observed: bool = True):
    """Add a node to the DAG."""
    if name and name not in st.session_state.nodes:
        st.session_state.nodes[name] = {"observed": observed}
        return True
    return False


def remove_node(name: str):
    """Remove a node and its connected edges."""
    if name in st.session_state.nodes:
        del st.session_state.nodes[name]
        st.session_state.edges = [
            (c, e) for c, e in st.session_state.edges if c != name and e != name
        ]
        if st.session_state.treatment == name:
            st.session_state.treatment = None
        if st.session_state.outcome == name:
            st.session_state.outcome = None


def toggle_observed(name: str):
    """Toggle node observed status."""
    if name in st.session_state.nodes:
        st.session_state.nodes[name]["observed"] = not st.session_state.nodes[name][
            "observed"
        ]


def add_edge(cause: str, effect: str):
    """Add an edge to the DAG."""
    if cause and effect and cause != effect:
        edge = (cause, effect)
        if edge not in st.session_state.edges:
            st.session_state.edges.append(edge)
            return True
    return False


def remove_edge(cause: str, effect: str):
    """Remove an edge from the DAG."""
    edge = (cause, effect)
    if edge in st.session_state.edges:
        st.session_state.edges.remove(edge)


def build_networkx_graph() -> nx.DiGraph:
    """Build NetworkX DiGraph from session state."""
    G = nx.DiGraph()
    for name, props in st.session_state.nodes.items():
        G.add_node(name, observed=props["observed"])
    for cause, effect in st.session_state.edges:
        G.add_edge(cause, effect)
    return G


def graph_to_gml_string(G: nx.DiGraph) -> str:
    """Convert NetworkX graph to GML string for DoWhy."""
    buffer = BytesIO()
    nx.write_gml(G, buffer)
    return buffer.getvalue().decode("utf-8")


def create_agraph_elements() -> tuple[list[Node], list[Edge]]:
    """Create agraph nodes and edges for visualization."""
    nodes = []
    edges = []

    treatment = st.session_state.treatment
    outcome = st.session_state.outcome

    for name, props in st.session_state.nodes.items():
        # Determine color based on role
        if name == outcome:
            color = COLORS["outcome"]
        elif name == treatment:
            color = COLORS["treatment"]
        elif props["observed"]:
            color = COLORS["observed"]
        else:
            color = COLORS["unobserved"]

        is_unobserved = not props["observed"]

        if is_unobserved:
            nodes.append(
                Node(
                    id=name,
                    label=name,
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
                    id=name,
                    label=name,
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

    for cause, effect in st.session_state.edges:
        edges.append(
            Edge(
                source=cause,
                target=effect,
                color=COLORS["edge"],
                width=1.5,
            )
        )

    return nodes, edges


def run_identify_effect() -> tuple[str | None, str | None]:
    """Run DoWhy's identify_effect and return (result, error)."""
    treatment = st.session_state.treatment
    outcome = st.session_state.outcome

    if not treatment or not outcome:
        return None, "Select both treatment and outcome"

    # Treatment and outcome must be observed
    if not st.session_state.nodes[treatment]["observed"]:
        return None, f"Treatment '{treatment}' must be observed"
    if not st.session_state.nodes[outcome]["observed"]:
        return None, f"Outcome '{outcome}' must be observed"

    G = build_networkx_graph()

    if not nx.is_directed_acyclic_graph(G):
        return None, "Graph contains cycles"

    try:
        import pandas as pd

        gml_string = graph_to_gml_string(G)

        # Only include observed nodes in data - DoWhy uses this to determine observability
        observed_nodes = [
            name for name, props in st.session_state.nodes.items()
            if props["observed"]
        ]
        dummy_data = pd.DataFrame({name: [0, 1] for name in observed_nodes})

        model = CausalModel(
            data=dummy_data,
            treatment=treatment,
            outcome=outcome,
            graph=gml_string,
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=False)
        return str(identified_estimand), None

    except Exception as e:
        return None, str(e)


def export_dag() -> str:
    """Export DAG to JSON format."""
    data = {
        "nodes": [
            {"name": name, "observed": props["observed"]}
            for name, props in st.session_state.nodes.items()
        ],
        "edges": [{"cause": c, "effect": e} for c, e in st.session_state.edges],
        "treatment": st.session_state.treatment,
        "outcome": st.session_state.outcome,
    }
    return json.dumps(data, indent=2)


def import_dag(json_str: str) -> tuple[bool, str]:
    """Import DAG from JSON format."""
    try:
        data = json.loads(json_str)

        # Clear current state
        st.session_state.nodes = {}
        st.session_state.edges = []
        st.session_state.treatment = None
        st.session_state.outcome = None

        # Import nodes
        for node in data.get("nodes", []):
            st.session_state.nodes[node["name"]] = {
                "observed": node.get("observed", True)
            }

        # Import edges
        for edge in data.get("edges", []):
            st.session_state.edges.append((edge["cause"], edge["effect"]))

        # Import treatment/outcome
        st.session_state.treatment = data.get("treatment")
        st.session_state.outcome = data.get("outcome")

        return True, "Import successful"
    except Exception as e:
        return False, f"Import failed: {str(e)}"


# =============================================================================
# Main UI
# =============================================================================

init_session_state()

st.title("DAG Builder + DoWhy Identifiability")

col_edit, col_graph, col_analysis = st.columns([1, 2, 1])

with col_edit:
    st.subheader("Edit DAG")

    # Add node section
    st.markdown("**Add Node**")
    col_name, col_obs = st.columns([2, 1])
    with col_name:
        new_node_name = st.text_input("Name", key="new_node_input", label_visibility="collapsed", placeholder="Node name")
    with col_obs:
        new_node_observed = st.checkbox("Observed", value=True, key="new_node_observed")

    if st.button("Add Node", use_container_width=True):
        if add_node(new_node_name, new_node_observed):
            st.rerun()

    # Current nodes
    st.markdown("**Nodes**")
    if st.session_state.nodes:
        for name, props in list(st.session_state.nodes.items()):
            col_n, col_o, col_d = st.columns([2, 1, 1])
            with col_n:
                obs_tag = "observed" if props["observed"] else "unobserved"
                st.markdown(
                    f'<span class="tag tag-{obs_tag}">{obs_tag}</span> {name}',
                    unsafe_allow_html=True,
                )
            with col_o:
                if st.button("Toggle", key=f"toggle_{name}"):
                    toggle_observed(name)
                    st.rerun()
            with col_d:
                if st.button("Del", key=f"del_{name}"):
                    remove_node(name)
                    st.rerun()
    else:
        st.info("No nodes yet")

    st.markdown("---")

    # Add edge section
    st.markdown("**Add Edge**")
    node_names = list(st.session_state.nodes.keys())
    if len(node_names) >= 2:
        col_cause, col_arrow, col_effect = st.columns([2, 1, 2])
        with col_cause:
            cause = st.selectbox("Cause", node_names, key="edge_cause", label_visibility="collapsed")
        with col_arrow:
            st.markdown("<div style='text-align:center; padding-top:8px; color:#8b949e;'>→</div>", unsafe_allow_html=True)
        with col_effect:
            effect = st.selectbox("Effect", node_names, key="edge_effect", label_visibility="collapsed")

        if st.button("Add Edge", use_container_width=True):
            if add_edge(cause, effect):
                st.rerun()
    else:
        st.info("Add at least 2 nodes")

    # Current edges
    st.markdown("**Edges**")
    if st.session_state.edges:
        for cause, effect in list(st.session_state.edges):
            col_e, col_r = st.columns([3, 1])
            with col_e:
                st.text(f"{cause} → {effect}")
            with col_r:
                if st.button("Del", key=f"del_edge_{cause}_{effect}"):
                    remove_edge(cause, effect)
                    st.rerun()
    else:
        st.info("No edges yet")

    st.markdown("---")

    # Import/Export
    st.markdown("**Import/Export**")
    with st.expander("Import JSON"):
        import_json = st.text_area("Paste JSON", key="import_json", height=100)
        if st.button("Import"):
            success, msg = import_dag(import_json)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    if st.button("Export JSON", use_container_width=True):
        st.code(export_dag(), language="json")

with col_graph:
    st.subheader("Graph")

    if st.session_state.nodes:
        nodes, edges = create_agraph_elements()

        config = Config(
            width="100%",
            height=400,
            directed=True,
            hierarchical=True,
            levelSeparation=100,
            nodeSpacing=120,
            treeSpacing=150,
            physics=False,
            nodeHighlightBehavior=True,
            highlightColor="#f0f6fc",
        )

        agraph(nodes=nodes, edges=edges, config=config)

        # Legend
        st.markdown(
            """
            <div style="font-size: 11px; color: #8b949e; margin-top: 8px;">
                <span style="color: #3fb950;">■</span> Treatment &nbsp;
                <span style="color: #a371f7;">■</span> Outcome &nbsp;
                <span style="color: #58a6ff;">■</span> Observed &nbsp;
                <span style="color: #8b949e;">◯</span> Unobserved (dashed)
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show identify_effect result if available
        if "identify_result" in st.session_state:
            st.markdown("---")
            result, error = st.session_state.identify_result
            if error:
                st.error(error)
            else:
                st.code(result, language="text")
    else:
        st.info("Add nodes to visualize the DAG")

with col_analysis:
    st.subheader("Treatment & Outcome")

    node_names = list(st.session_state.nodes.keys())

    if node_names:
        treatment_options = ["(none)"] + node_names
        outcome_options = ["(none)"] + node_names

        treatment_idx = (
            treatment_options.index(st.session_state.treatment)
            if st.session_state.treatment in treatment_options
            else 0
        )
        outcome_idx = (
            outcome_options.index(st.session_state.outcome)
            if st.session_state.outcome in outcome_options
            else 0
        )

        new_treatment = st.selectbox(
            "Treatment",
            treatment_options,
            index=treatment_idx,
            key="treatment_select",
        )
        new_outcome = st.selectbox(
            "Outcome",
            outcome_options,
            index=outcome_idx,
            key="outcome_select",
        )

        st.session_state.treatment = new_treatment if new_treatment != "(none)" else None
        st.session_state.outcome = new_outcome if new_outcome != "(none)" else None

        st.markdown("---")

        if st.button("Run identify_effect()", use_container_width=True, type="primary"):
            st.session_state.identify_result = run_identify_effect()
            st.rerun()
    else:
        st.info("Add nodes first")
