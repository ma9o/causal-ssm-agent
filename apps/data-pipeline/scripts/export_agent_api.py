"""Export the agent HTTP surface as an OpenAPI spec and a curl skill doc.

The tool server's FastAPI app *is* the agent interface (there is no MCP server).
This script dumps its OpenAPI schema to ``packages/api-types/schemas/openapi.json``
and renders that same spec into ``.agents/skills/nof1-episode-api/SKILL.md`` — a
cross-tool Agent Skill (agentskills.io) telling any LLM how to drive the episode
machine with curl. Codex reads ``.agents/skills`` natively; a committed symlink
``.claude/skills/nof1-episode-api`` -> the same dir makes Claude Code load it
too. The narrative comes from the app description and route docstrings, so the
skill never drifts from the API: enrich the docstrings, regenerate.

Usage:
    cd apps/data-pipeline
    uv run python scripts/export_agent_api.py            # write
    uv run python scripts/export_agent_api.py --check    # verify freshness
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.tool_server import app

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENAPI_PATH = REPO_ROOT / "packages" / "api-types" / "schemas" / "openapi.json"
SKILL_PATH = REPO_ROOT / ".agents" / "skills" / "nof1-episode-api" / "SKILL.md"

_BASE_URL = "${TOOL_SERVER_URL:-http://localhost:8100}"
_MAX_EXAMPLE_DEPTH = 8


def _example_from_schema(
    schema: UncheckedJsonObject,
    components: UncheckedJsonObject,
    *,
    depth: int = 0,
    seen: frozenset[str] = frozenset(),
) -> Any:
    """A minimal JSON skeleton satisfying a schema, for a copy-pasteable body.

    Required object properties only, first branch of a union, first enum value.
    The authoritative, hand-written body examples live in the app description;
    this is just a shape hint next to each endpoint.
    """
    if depth > _MAX_EXAMPLE_DEPTH or not isinstance(schema, dict):
        return {}

    ref = schema.get("$ref")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        if name in seen:
            return {}
        return _example_from_schema(
            components.get(name, {}), components, depth=depth, seen=seen | {name}
        )

    for combinator in ("oneOf", "anyOf", "allOf"):
        branches = schema.get(combinator)
        if branches:
            non_null = [b for b in branches if b.get("type") != "null"] or branches
            return _example_from_schema(non_null[0], components, depth=depth + 1, seen=seen)

    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if enum:
        return enum[0]

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        props: UncheckedJsonObject = schema.get("properties", {})
        required = set(schema.get("required", []))
        # Include required fields plus any discriminator/fixed-value field (a
        # `const`, e.g. move `kind`) even when a default makes it non-required,
        # so the skeleton is a valid body. Property order is preserved.
        return {
            name: _example_from_schema(prop, components, depth=depth + 1, seen=seen)
            for name, prop in props.items()
            if name in required or "const" in prop
        }
    if schema_type == "array":
        items = schema.get("items")
        return (
            [_example_from_schema(items, components, depth=depth + 1, seen=seen)] if items else []
        )
    if schema_type == "integer" or schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    return "string"


def _path_with_placeholders(path: str, parameters: list[UncheckedJsonObject]) -> str:
    """Substitute path params with uppercase placeholders for the curl example."""
    result = path
    for param in parameters:
        if param.get("in") == "path":
            name = param["name"]
            result = result.replace("{" + name + "}", name.upper())
    return result


def _curl_block(
    method: str,
    path: str,
    operation: UncheckedJsonObject,
    components: UncheckedJsonObject,
) -> str:
    parameters = operation.get("parameters", [])
    url = f"{_BASE_URL}{_path_with_placeholders(path, parameters)}"
    lines = [f'curl -s "{url}"']
    if method != "get":
        lines[0] += " \\"
        lines.append(f"  -X {method.upper()} \\")
        lines.append("  -H 'Content-Type: application/json' \\")
        body_schema = (
            operation.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        example = _example_from_schema(body_schema, components) if body_schema else {}
        lines.append(f"  -d '{json.dumps(example)}'")
    return "```bash\n" + "\n".join(lines) + "\n```"


def _parameters_block(operation: UncheckedJsonObject) -> str | None:
    parameters = operation.get("parameters", [])
    if not parameters:
        return None
    rows = ["**Parameters**", ""]
    for param in parameters:
        location = param.get("in", "query")
        required = "required" if param.get("required") else "optional"
        description = param.get("description", "").strip().replace("\n", " ")
        suffix = f" — {description}" if description else ""
        rows.append(f"- `{param['name']}` ({location}, {required}){suffix}")
    return "\n".join(rows)


def _skill_frontmatter() -> str:
    """YAML frontmatter for the Agent Skill.

    `name` and `description` satisfy both the Codex requirement and Claude Code's
    trigger heuristic; the description says what the skill does and when to reach
    for it. Kept as a controlled single-line double-quoted scalar for maximal
    cross-parser compatibility.
    """
    description = (
        "Drive or inspect the nof1-causal-lab episode state machine over HTTP with "
        "curl: run pipeline stages, write judgment artifacts (latent structure, "
        "causal design, priors), read episode state/timeline/artifacts, and invoke "
        "stage tools against the tool server. Use when navigating the episode "
        "machine as an external agent instead of the web viewer."
    )
    return f'---\nname: nof1-episode-api\ndescription: "{description}"\n---'


def render_skill(openapi: UncheckedJsonObject) -> str:
    info = openapi.get("info", {})
    components = openapi.get("components", {}).get("schemas", {})
    title = info.get("title", "Agent API")

    blocks: list[str] = [
        _skill_frontmatter(),
        f"# {title} — curl skill",
        (
            "> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI "
            "OpenAPI spec) by `apps/data-pipeline/scripts/export_agent_api.py`. "
            "Edit the route docstrings, not this file."
        ),
        inspect.cleandoc(info.get("description", "")),
        "## Endpoints",
    ]

    for path in sorted(openapi.get("paths", {})):
        methods = openapi["paths"][path]
        for method in sorted(methods):
            operation = methods[method]
            if not isinstance(operation, dict):
                continue
            blocks.append(f"### {method.upper()} `{path}`")
            description = inspect.cleandoc(
                operation.get("description") or operation.get("summary") or ""
            )
            if description:
                blocks.append(description)
            params = _parameters_block(operation)
            if params:
                blocks.append(params)
            blocks.append(_curl_block(method, path, operation, components))

    text = "\n\n".join(block for block in blocks if block)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip("\n") + "\n"


def _write_or_check(path: Path, rendered: str, *, check: bool, changed: list[Path]) -> None:
    if check:
        if not path.exists() or path.read_text() != rendered:
            changed.append(path.relative_to(REPO_ROOT))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def main(*, check: bool = False) -> bool:
    openapi = app.openapi()
    changed: list[Path] = []

    openapi_json = json.dumps(openapi, indent=2) + "\n"
    _write_or_check(OPENAPI_PATH, openapi_json, check=check, changed=changed)

    skill_md = render_skill(openapi)
    _write_or_check(SKILL_PATH, skill_md, check=check, changed=changed)

    if not check:
        n_ops = sum(1 for methods in openapi.get("paths", {}).values() for _ in methods)
        print(f"Exported OpenAPI ({n_ops} operations) to {OPENAPI_PATH}")
        print(f"Rendered curl skill to {SKILL_PATH}")
        return False

    if changed:
        print("Agent API exports are out of date. Run `bun run codegen`.", file=sys.stderr)
        for path in changed:
            print(f"  {path}", file=sys.stderr)
        return True

    print("Agent API exports checked.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Verify generated files without writing."
    )
    args = parser.parse_args()
    if main(check=args.check):
        sys.exit(1)
