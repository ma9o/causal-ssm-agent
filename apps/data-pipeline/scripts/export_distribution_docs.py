"""Export generated docs from the central distribution catalog.

Usage:
    cd apps/data-pipeline
    uv run python scripts/export_distribution_docs.py

Overwrites generated sections in:
  - docs/reference/model-spec/parameters.md  (Supported Prior Families, Common Defaults)
  - docs/reference/model-spec/likelihoods.md (Dtype-to-Distribution Mapping,
    Distribution Families, Link Functions)

Hand-written prose surrounding these sections is preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from causal_ssm_agent.distributions import (
    render_distribution_families_prose,
    render_dtype_likelihood_markdown_table,
    render_link_functions_prose,
    render_parameter_constraint_notes,
    render_parameter_roles_markdown_table,
    render_prior_distribution_markdown_table,
    render_prior_parameter_guidance_markdown_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCS_DIR = _REPO_ROOT / "docs" / "reference" / "model-spec"
_PARAMETERS_PATH = _DOCS_DIR / "parameters.md"
_LIKELIHOODS_PATH = _DOCS_DIR / "likelihoods.md"


def _replace_section(text: str, heading: str, new_body: str, path: Path) -> str:
    """Replace everything between *heading* and the next ``## `` heading."""
    pattern = re.compile(
        rf"({re.escape(heading)}\n)(.*?)(?=\n## |\Z)",
        re.DOTALL,
    )
    replacement = rf"\1\n{new_body}\n"
    result, n = pattern.subn(replacement, text)
    if n == 0:
        raise ValueError(f"Section '{heading}' not found in {path}")
    return result


def _export_parameters(*, check: bool) -> bool:
    """Regenerate the generated sections of parameters.md in-place."""
    original = _PARAMETERS_PATH.read_text()

    roles_body = "\n".join(
        [
            "The [Stage 4 skeleton](../../pipeline/04-model-specification-priors.md) creates exactly "
            "the following parameters from a [`CausalSpec`](../../pipeline/01b-measurement-identifiability.md#causalspec):",
            "",
            render_parameter_roles_markdown_table(),
            "",
            "Constraint notes:",
            "",
            render_parameter_constraint_notes(),
        ]
    )

    families_body = "\n".join(
        [
            render_prior_distribution_markdown_table(),
            "",
            "The `Family` values are the exact canonical strings accepted by Stage 4 prior schemas; aliases are not supported.",
            "The `Use When` column is the authoritative short guidance reused by the Stage 4 prompts.",
        ]
    )

    defaults_body = render_prior_parameter_guidance_markdown_table()

    updated = _replace_section(original, "## Parameter Roles", roles_body, _PARAMETERS_PATH)
    updated = _replace_section(updated, "## Supported Prior Families", families_body, _PARAMETERS_PATH)
    updated = _replace_section(updated, "## Common Defaults", defaults_body, _PARAMETERS_PATH)

    if check:
        changed = updated != original
        if changed:
            print(f"  {_PARAMETERS_PATH.relative_to(_REPO_ROOT)}")
        return changed

    _PARAMETERS_PATH.write_text(updated)
    print(f"  parameters.md  -> {_PARAMETERS_PATH}")
    return updated != original


def _export_likelihoods(*, check: bool) -> bool:
    """Regenerate the generated sections of likelihoods.md in-place."""
    original = _LIKELIHOODS_PATH.read_text()

    dtype_body = "\n".join(
        [
            "Each indicator's [`measurement_dtype`](../../pipeline/01b-measurement-identifiability.md#indicator) "
            "determines the default distribution and link function. "
            "Where the dtype admits only one valid combination, the likelihood is locked "
            "deterministically by the [Stage 4 skeleton](../../pipeline/04-model-specification-priors.md). "
            "Where alternatives exist, the LLM chooses via a decision card.",
            "",
            render_dtype_likelihood_markdown_table(),
        ]
    )

    updated = _replace_section(original, "## Dtype-to-Distribution Mapping", dtype_body, _LIKELIHOODS_PATH)
    updated = _replace_section(updated, "## Distribution Families", render_distribution_families_prose(), _LIKELIHOODS_PATH)
    updated = _replace_section(updated, "## Link Functions", render_link_functions_prose(), _LIKELIHOODS_PATH)

    if check:
        changed = updated != original
        if changed:
            print(f"  {_LIKELIHOODS_PATH.relative_to(_REPO_ROOT)}")
        return changed

    _LIKELIHOODS_PATH.write_text(updated)
    print(f"  likelihoods.md -> {_LIKELIHOODS_PATH}")
    return updated != original


def export_prior_distribution_docs(*, check: bool = False) -> bool:
    """Write the authoritative model-spec reference pages."""
    parameters_changed = _export_parameters(check=check)
    likelihoods_changed = _export_likelihoods(check=check)
    return parameters_changed or likelihoods_changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify generated docs without writing files.")
    args = parser.parse_args()

    has_changes = export_prior_distribution_docs(check=args.check)
    if args.check:
        if has_changes:
            print("Distribution docs codegen is out of date. Run `bun run docs:codegen`.", file=sys.stderr)
            sys.exit(1)
        print("Distribution docs checked.")
