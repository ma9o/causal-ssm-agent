"""Common utilities for DAG tools."""

import json

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


def parse_dag_json(json_str: str) -> tuple[dict | None, str | None]:
    """Parse DAG JSON - accepts both latent-only and full causal model formats.

    Latent-only format:
    {
        "latent": {
            "constructs": [...],
            "edges": [...]
        }
    }

    Full format:
    {
        "latent": {
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

    # Validate latent model format
    if "latent" not in data or "constructs" not in data.get("latent", {}):
        return None, "JSON must have 'latent.constructs'"

    latent = data["latent"]
    normalized = {
        "constructs": latent.get("constructs", []),
        "edges": latent.get("edges", []),
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
