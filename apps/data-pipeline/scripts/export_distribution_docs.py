"""Export generated docs from the central distribution catalog.

Usage:
    cd apps/data-pipeline
    uv run python scripts/export_distribution_docs.py
"""

from __future__ import annotations

from pathlib import Path

from causal_ssm_agent.distributions import render_prior_distribution_markdown_table

OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "primitives"
    / "model-spec"
    / "prior-distribution-families.md"
)


def export_prior_distribution_docs() -> None:
    """Write the authoritative prior-family reference page."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(
            [
                "# Supported Prior Distribution Families",
                "",
                "This page is generated from `causal_ssm_agent.distributions.PRIOR_FAMILY_SPECS`.",
                "Edit the Python catalog and re-run `uv run python scripts/export_distribution_docs.py` instead of editing this file manually.",
                "",
                render_prior_distribution_markdown_table(),
                "",
                "The `Use When` column is the authoritative short guidance reused by the Stage 4 prompts.",
                "",
            ]
        )
    )
    print(f"Exported prior distribution docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    export_prior_distribution_docs()
