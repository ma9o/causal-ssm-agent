"""Export generated docs from the central distribution catalog.

Usage:
    cd apps/data-pipeline
    uv run python scripts/export_distribution_docs.py

Overwrites generated sections in:
  - docs/reference/statistical-model-spec/parameters.md  (Supported Prior Families, Common Defaults)
  - docs/reference/statistical-model-spec/likelihoods.md (Dtype-to-Distribution Mapping,
    Distribution Families, Link Functions)

Hand-written prose surrounding these sections is preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from nof1_causal_lab.distributions import (
    OBSERVATION_FAMILY_SPECS,
    OBSERVATION_LINK_VALUES_BY_DISTRIBUTION,
    PARAMETER_ROLE_SPECS,
    PRIOR_FAMILY_SPECS,
    VALID_LIKELIHOODS_FOR_DTYPE,
    DistributionFamily,
    render_prior_parameter_guidance_markdown_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCS_DIR = _REPO_ROOT / "docs" / "reference" / "statistical-model-spec"
_PARAMETERS_PATH = _DOCS_DIR / "parameters.md"
_LIKELIHOODS_PATH = _DOCS_DIR / "likelihoods.md"

_DTYPE_ALTERNATIVE_NOTES: dict[tuple[str, DistributionFamily], str] = {
    ("categorical", DistributionFamily.ORDERED_LOGISTIC): (
        "when categories are substantively ordered"
    ),
}


def _render_prior_distribution_markdown_table() -> str:
    lines = [
        "| Family | Signature | Support | Use When |",
        "|---|---|---|---|",
    ]
    for spec in PRIOR_FAMILY_SPECS:
        lines.append(
            f"| `{spec.family.value}` | `{spec.signature}` | `{spec.support}` | {spec.summary} |"
        )
    return "\n".join(lines)


def _format_dist_with_links(dist: DistributionFamily) -> str:
    links = OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[dist]
    if len(links) == 1:
        return f"`{dist.value}` (`{links[0]}`)"
    link_str = " or ".join(f"`{link}`" for link in links)
    return f"`{dist.value}` ({link_str})"


def _render_dtype_likelihood_markdown_table() -> str:
    lines = [
        "| `measurement_dtype` | Default distribution | Link | Alternatives |",
        "|---|---|---|---|",
    ]
    for dtype, valid_dists in VALID_LIKELIHOODS_FOR_DTYPE.items():
        default_dist = valid_dists[0]
        default_links = OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[default_dist]
        alternatives = [f"`{default_dist.value}` with `{link}`" for link in default_links[1:]]
        for dist in valid_dists[1:]:
            entry = _format_dist_with_links(dist)
            note = _DTYPE_ALTERNATIVE_NOTES.get((dtype, dist))
            alternatives.append(f"{entry} {note}" if note else entry)
        alt_str = ", ".join(alternatives) if alternatives else "None"
        lines.append(f"| `{dtype}` | `{default_dist.value}` | `{default_links[0]}` | {alt_str} |")
    return "\n".join(lines)


def _render_distribution_families_prose() -> str:
    names = [f"`{spec.family.value}`" for spec in OBSERVATION_FAMILY_SPECS]
    return (
        "`DistributionFamily` enumerates the valid likelihood distribution names: "
        f"{', '.join(names[:-1])}, and {names[-1]}."
    )


def _render_link_functions_prose() -> str:
    links = list(
        dict.fromkeys(f"`{link}`" for spec in OBSERVATION_FAMILY_SPECS for link in spec.links)
    )
    return (
        "`LinkFunction` enumerates the valid link function names: "
        f"{', '.join(links[:-1])}, and {links[-1]}."
    )


def _render_parameter_roles_markdown_table() -> str:
    lines = [
        "| Role | Symbol | Count | Constraint | SSM location |",
        "|---|---|---|---|---|",
    ]
    for spec in PARAMETER_ROLE_SPECS:
        constraint_cell = f"`{spec.constraint}` `{spec.domain}`"
        if spec.role == "loading":
            constraint_cell = "`positive` or `negative`"
        lines.append(
            f"| `{spec.role}` | `{spec.symbol}` "
            f"| {spec.count} | {constraint_cell} | {spec.ssm_location} |"
        )
    return "\n".join(lines)


def _render_parameter_constraint_notes() -> str:
    notes = [spec for spec in PARAMETER_ROLE_SPECS if spec.note]
    return "\n".join(f"- `{spec.role}`: {spec.note}" for spec in notes)


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
            "The [model-spec skeleton](../../pipeline/statistical-model-spec.md) creates exactly "
            "the following parameters from a [`StructuralPlan`](../../pipeline/measurement-structure.md#structuralplan):",
            "",
            _render_parameter_roles_markdown_table(),
            "",
            "Constraint notes:",
            "",
            _render_parameter_constraint_notes(),
        ]
    )

    families_body = "\n".join(
        [
            _render_prior_distribution_markdown_table(),
            "",
            "The `Family` values are the exact canonical strings accepted by model-spec prior schemas; aliases are not supported.",
            "The `Use When` column is the authoritative short guidance reused by the model-spec prompts.",
        ]
    )

    defaults_body = render_prior_parameter_guidance_markdown_table()

    updated = _replace_section(original, "## Parameter Roles", roles_body, _PARAMETERS_PATH)
    updated = _replace_section(
        updated, "## Supported Prior Families", families_body, _PARAMETERS_PATH
    )
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
            "Each indicator's [`measurement_dtype`](../../pipeline/measurement-structure.md#indicator) "
            "determines the default distribution and link function. "
            "Where the dtype admits only one valid combination, the likelihood is locked "
            "deterministically by the [model-spec skeleton](../../pipeline/statistical-model-spec.md). "
            "Where alternatives exist, the LLM chooses via a decision card.",
            "",
            _render_dtype_likelihood_markdown_table(),
        ]
    )

    updated = _replace_section(
        original, "## Dtype-to-Distribution Mapping", dtype_body, _LIKELIHOODS_PATH
    )
    updated = _replace_section(
        updated,
        "## Distribution Families",
        _render_distribution_families_prose(),
        _LIKELIHOODS_PATH,
    )
    updated = _replace_section(
        updated, "## Link Functions", _render_link_functions_prose(), _LIKELIHOODS_PATH
    )

    if check:
        changed = updated != original
        if changed:
            print(f"  {_LIKELIHOODS_PATH.relative_to(_REPO_ROOT)}")
        return changed

    _LIKELIHOODS_PATH.write_text(updated)
    print(f"  likelihoods.md -> {_LIKELIHOODS_PATH}")
    return updated != original


def export_prior_distribution_docs(*, check: bool = False) -> bool:
    """Write the authoritative statistical-model-spec reference pages."""
    parameters_changed = _export_parameters(check=check)
    likelihoods_changed = _export_likelihoods(check=check)
    return parameters_changed or likelihoods_changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Verify generated docs without writing files."
    )
    args = parser.parse_args()

    has_changes = export_prior_distribution_docs(check=args.check)
    if args.check:
        if has_changes:
            print(
                "Distribution docs codegen is out of date. Run `bun run docs:codegen`.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Distribution docs checked.")
