"""Common utilities for DAG tools."""

import json
from dataclasses import dataclass
from io import BytesIO

import networkx as nx
import pandas as pd
from dowhy import CausalModel


def parse_dag_json(json_str: str) -> tuple[dict | None, str | None]:
    """Parse DAG JSON - accepts both structural-only and full DSEM formats.

    Structural-only format:
    {
        "structural": {
            "constructs": [...],
            "edges": [...]
        }
    }

    Full DSEM format:
    {
        "structural": {
            "constructs": [...],
            "edges": [...]
        },
        "measurement": {
            "indicators": [...]
        }
    }

    Returns (data, error) tuple - one will always be None.
    Data is normalized to have 'constructs', 'edges', and 'indicators' keys
    (indicators will be empty list if not provided).
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    # Validate structural model format
    if "structural" not in data or "constructs" not in data.get("structural", {}):
        return None, "JSON must have 'structural.constructs'"

    structural = data["structural"]
    normalized = {
        "constructs": structural.get("constructs", []),
        "edges": structural.get("edges", []),
        "indicators": [],  # Always present, defaults to empty
    }

    # Include indicators if measurement model is present
    if "measurement" in data and "indicators" in data.get("measurement", {}):
        normalized["indicators"] = data["measurement"]["indicators"]

    # Validate edges reference valid nodes
    node_names = {c["name"] for c in normalized["constructs"]}
    for edge in normalized["edges"]:
        if edge["cause"] not in node_names:
            return None, f"Edge references unknown cause: '{edge['cause']}'"
        if edge["effect"] not in node_names:
            return None, f"Edge references unknown effect: '{edge['effect']}'"

    return normalized, None


def dag_to_networkx(data: dict) -> nx.DiGraph:
    """Convert DAG JSON to NetworkX DiGraph for DoWhy.

    Expects normalized data with 'constructs' and 'edges' keys
    (as returned by parse_dag_json).
    """
    G = nx.DiGraph()
    for construct in data["constructs"]:
        G.add_node(construct["name"])
    for edge in data["edges"]:
        G.add_edge(edge["cause"], edge["effect"])
    return G


def graph_to_gml_string(G: nx.DiGraph) -> str:
    """Convert NetworkX graph to GML string for DoWhy."""
    buffer = BytesIO()
    nx.write_gml(G, buffer)
    return buffer.getvalue().decode("utf-8")


@dataclass
class IdentifyResult:
    """Result of causal effect identification."""

    identifiable: bool
    method: str | None  # "backdoor", "frontdoor", "iv", or None
    backdoor_vars: list[str]
    frontdoor_vars: list[str]
    iv_vars: list[str]
    estimand_str: str
    error: str | None = None


def run_identify_effect(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    observed_nodes: list[str] | None = None,
) -> IdentifyResult:
    """Run DoWhy's identify_effect on a graph.

    Args:
        graph: NetworkX DiGraph representing the causal structure
        treatment: Name of treatment variable
        outcome: Name of outcome variable
        observed_nodes: List of observed node names. If None, all nodes are observed.

    Returns:
        IdentifyResult with identification details
    """
    if not nx.is_directed_acyclic_graph(graph):
        return IdentifyResult(
            identifiable=False,
            method=None,
            backdoor_vars=[],
            frontdoor_vars=[],
            iv_vars=[],
            estimand_str="",
            error="Graph contains cycles",
        )

    try:
        # Determine observed nodes
        if observed_nodes is None:
            observed_nodes = list(graph.nodes())

        # Create dummy data for DoWhy (only observed nodes)
        dummy_data = pd.DataFrame({name: [0.0] for name in observed_nodes})

        # Convert graph to GML for DoWhy
        gml_string = graph_to_gml_string(graph)

        model = CausalModel(
            data=dummy_data,
            treatment=treatment,
            outcome=outcome,
            graph=gml_string,
        )

        identified = model.identify_effect(proceed_when_unidentifiable=True)

        # Extract variables
        backdoor_vars = identified.get_backdoor_variables() or []
        frontdoor_vars = identified.get_frontdoor_variables() or []
        iv_vars = identified.get_instrumental_variables() or []

        # Determine method
        method = None
        if backdoor_vars:
            method = "backdoor"
        elif frontdoor_vars:
            method = "frontdoor"
        elif iv_vars:
            method = "iv"

        identifiable = bool(identified.estimands and method is not None)

        return IdentifyResult(
            identifiable=identifiable,
            method=method,
            backdoor_vars=list(backdoor_vars),
            frontdoor_vars=list(frontdoor_vars),
            iv_vars=list(iv_vars),
            estimand_str=str(identified),
        )

    except Exception as e:
        return IdentifyResult(
            identifiable=False,
            method=None,
            backdoor_vars=[],
            frontdoor_vars=[],
            iv_vars=[],
            estimand_str="",
            error=str(e),
        )


# HTML/JS for copying graph canvas to clipboard
COPY_GRAPH_HTML = """
<style>
.copy-btn {
    background: #238636;
    color: white;
    border: none;
    padding: 4px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s ease;
}
.copy-btn.success {
    background: #1f6feb;
    transform: scale(1.05);
}
.copy-btn.error {
    background: #da3633;
}
</style>
<button id="copyBtn" class="copy-btn" onclick="copyGraph()">Copy</button>
<script>
async function copyGraph() {
    const btn = document.getElementById('copyBtn');
    const originalText = btn.innerHTML;

    // Search up through parent frames to find all iframes
    let root = window;
    while (root.parent && root.parent !== root) {
        root = root.parent;
    }
    const iframes = root.document.querySelectorAll('iframe');
    let canvas = null;
    for (const iframe of iframes) {
        try {
            canvas = iframe.contentDocument.querySelector('canvas');
            if (canvas) break;
        } catch (e) {}
    }
    if (!canvas) {
        btn.innerHTML = 'Not found';
        btn.classList.add('error');
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove('error');
        }, 1500);
        return;
    }
    try {
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        await navigator.clipboard.write([
            new ClipboardItem({'image/png': blob})
        ]);
        btn.innerHTML = 'Copied!';
        btn.classList.add('success');
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove('success');
        }, 1500);
    } catch (err) {
        // Fallback: download
        const link = root.document.createElement('a');
        link.download = 'dag.png';
        link.href = canvas.toDataURL('image/png');
        root.document.body.appendChild(link);
        link.click();
        root.document.body.removeChild(link);
        btn.innerHTML = 'Downloaded!';
        btn.classList.add('success');
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove('success');
        }, 1500);
    }
}
</script>
"""
