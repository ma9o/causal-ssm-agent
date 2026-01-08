"""
Interactive DAG Builder with DoWhy Identifiability Analysis.

Run with: uv run streamlit run tools/dag_identifiability.py
"""

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import Config, Edge, Node, agraph

from commons import COPY_GRAPH_HTML, run_identify_effect

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
        st.session_state.nodes = {
            "A": {"observed": True},
            "B": {"observed": True},
            "C": {"observed": True},
            "D": {"observed": True},
            "U": {"observed": False},
        }
    if "edges" not in st.session_state:
        st.session_state.edges = [
            ("A", "B"),
            ("B", "C"),
            ("U", "B"),
            ("U", "C"),
            ("U", "D"),
        ]
    if "treatment" not in st.session_state:
        st.session_state.treatment = None
    if "outcome" not in st.session_state:
        st.session_state.outcome = None


def build_networkx_graph() -> nx.DiGraph:
    """Build NetworkX DiGraph from session state."""
    G = nx.DiGraph()
    for name, props in st.session_state.nodes.items():
        G.add_node(name, observed=props["observed"])
    for cause, effect in st.session_state.edges:
        G.add_edge(cause, effect)
    return G


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


def do_identify_effect() -> tuple[str | None, str | None]:
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

    # Get observed nodes for DoWhy
    observed_nodes = [
        name for name, props in st.session_state.nodes.items() if props["observed"]
    ]

    result = run_identify_effect(G, treatment, outcome, observed_nodes)

    if result.error:
        return None, result.error
    return result.estimand_str, None


def generate_mermaid_spec() -> str:
    """Generate Mermaid flowchart spec from current DAG state."""
    lines = ["graph TD"]

    treatment = st.session_state.treatment
    outcome = st.session_state.outcome

    # Add node style definitions
    for name, props in st.session_state.nodes.items():
        if name == treatment:
            lines.append(f"    {name}[{name}]:::treatment")
        elif name == outcome:
            lines.append(f"    {name}[{name}]:::outcome")
        elif props["observed"]:
            lines.append(f"    {name}[{name}]:::observed")
        else:
            lines.append(f"    {name}({name}):::unobserved")

    # Add edges
    for cause, effect in st.session_state.edges:
        lines.append(f"    {cause} --> {effect}")

    # Add style classes
    lines.append("")
    lines.append("    classDef treatment fill:#3fb950,stroke:#238636,color:#fff")
    lines.append("    classDef outcome fill:#a371f7,stroke:#8957e5,color:#fff")
    lines.append("    classDef observed fill:#58a6ff,stroke:#388bfd,color:#fff")
    lines.append("    classDef unobserved fill:#8b949e66,stroke:#8b949e,color:#fff,stroke-dasharray: 5 5")

    return "\n".join(lines)


def parse_mermaid_spec(spec: str) -> tuple[bool, str]:
    """Parse Mermaid spec and update DAG state."""
    import re

    try:
        lines = spec.strip().split("\n")

        new_nodes = {}
        new_edges = []
        new_treatment = None
        new_outcome = None

        for line in lines:
            line = line.strip()

            # Skip empty lines, graph declaration, and classDef lines
            if not line or line.startswith("graph") or line.startswith("classDef"):
                continue

            # Parse edge: A --> B
            edge_match = re.match(r"(\w+)\s*-->\s*(\w+)", line)
            if edge_match:
                cause, effect = edge_match.groups()
                new_edges.append((cause, effect))
                # Ensure both nodes exist
                if cause not in new_nodes:
                    new_nodes[cause] = {"observed": True}
                if effect not in new_nodes:
                    new_nodes[effect] = {"observed": True}
                continue

            # Parse node with class: Name[Name]:::class or Name(Name):::class
            node_match = re.match(r"(\w+)[\[\(]([^\]\)]+)[\]\)](?:::(\w+))?", line)
            if node_match:
                name, _label, node_class = node_match.groups()

                # Determine observed status from shape (parentheses = unobserved)
                is_unobserved = "(" in line and ")" in line and "[" not in line
                observed = not is_unobserved

                # Override with class if specified
                if node_class == "unobserved":
                    observed = False
                elif node_class in ("observed", "treatment", "outcome"):
                    observed = True

                new_nodes[name] = {"observed": observed}

                if node_class == "treatment":
                    new_treatment = name
                elif node_class == "outcome":
                    new_outcome = name
                continue

        # Update session state
        st.session_state.nodes = new_nodes
        st.session_state.edges = new_edges
        st.session_state.treatment = new_treatment
        st.session_state.outcome = new_outcome

        return True, ""
    except Exception as e:
        return False, str(e)


# =============================================================================
# Main UI
# =============================================================================

init_session_state()

st.title("DAG Builder + DoWhy Identifiability")

col_edit, col_graph, col_analysis = st.columns([1, 2, 1])

with col_edit:
    st.subheader("Mermaid Spec")

    # Generate current spec from DAG state
    current_spec = generate_mermaid_spec() if st.session_state.nodes else "graph TD"

    # Use a dynamic key based on DAG state to force text_area refresh
    nodes_tuple = tuple((k, v["observed"]) for k, v in sorted(st.session_state.nodes.items()))
    dag_key = hash((nodes_tuple, tuple(st.session_state.edges),
                    st.session_state.treatment, st.session_state.outcome))

    edited_spec = st.text_area(
        "Edit Mermaid",
        value=current_spec,
        height=350,
        key=f"mermaid_editor_{dag_key}",
        label_visibility="collapsed",
    )

    if st.button("Apply", use_container_width=True, type="primary"):
        success, error = parse_mermaid_spec(edited_spec)
        if success:
            st.rerun()
        else:
            st.error(f"Parse error: {error}")

    st.caption("Use `Name[Label]:::class` for nodes, `A --> B` for edges. Classes: `treatment`, `outcome`, `observed`, `unobserved`")

with col_graph:
    st.subheader("Graph")

    if st.session_state.nodes:
        nodes, edges = create_agraph_elements()

        config = Config(
            width="100%",
            height=400,
            directed=True,
            physics=True,
            nodeHighlightBehavior=True,
            highlightColor="#f0f6fc",
            interaction={"zoomView": False},
        )

        agraph(nodes=nodes, edges=edges, config=config)

        # Legend and copy button
        col_legend, col_copy = st.columns([3, 1])
        with col_legend:
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
        with col_copy:
            components.html(COPY_GRAPH_HTML, height=35)

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
            st.session_state.identify_result = do_identify_effect()
            st.rerun()
    else:
        st.info("Add nodes first")
