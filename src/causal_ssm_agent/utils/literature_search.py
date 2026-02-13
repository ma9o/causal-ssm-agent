"""Literature search for grounding priors using Exa Research API.

Uses Exa's agentic research API to find empirical effect sizes from published
literature, providing evidence-based context for prior elicitation.
"""

import os
from dataclasses import dataclass
from typing import Any

from exa_py import AsyncExa


@dataclass
class LiteratureEvidence:
    """Evidence from literature for a causal relationship."""

    cause: str
    effect: str
    summary: str
    effect_sizes: list[str]  # Reported effect sizes/ranges from literature
    sources: list[dict[str, str]]  # [{title, url, snippet}]
    confidence: str  # "high", "moderate", "low" based on evidence quality


@dataclass
class LiteratureContext:
    """Literature context for prior elicitation."""

    question_summary: str
    evidence: dict[str, LiteratureEvidence]  # keyed by "cause->effect"
    general_findings: str  # Domain-level findings
    raw_response: dict[str, Any]  # Full Exa response for debugging


def _build_research_instructions(
    question: str,
    edges: list[dict[str, str]],
) -> str:
    """Build research instructions for Exa."""
    edge_descriptions = []
    for edge in edges:
        cause = edge["cause"]
        effect = edge["effect"]
        edge_descriptions.append(f"- {cause} → {effect}")

    edges_text = "\n".join(edge_descriptions)

    return f"""\
Research the following causal relationships to find empirical effect sizes from published literature.

## Research Question
{question}

## Causal Relationships to Research
{edges_text}

## What to Find

For each causal relationship:
1. **Effect sizes**: Standardized coefficients, correlation coefficients, odds ratios, or other effect magnitude measures
2. **Direction**: Whether the relationship is positive or negative
3. **Confidence intervals**: Uncertainty around the estimates
4. **Study context**: Population, sample size, methodology

Focus on:
- Meta-analyses and systematic reviews (highest quality)
- Large-scale longitudinal studies
- Peer-reviewed journal articles
- Recent publications (last 10 years preferred)

Provide specific numerical values when available. If no direct studies exist, note related findings that could inform priors.
"""


def _build_output_schema() -> dict:
    """Build Pydantic-compatible output schema for Exa research."""
    return {
        "type": "object",
        "properties": {
            "question_summary": {
                "type": "string",
                "description": "Brief summary of the research question context",
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cause": {"type": "string"},
                        "effect": {"type": "string"},
                        "summary": {
                            "type": "string",
                            "description": "Summary of evidence for this relationship",
                        },
                        "effect_sizes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Reported effect sizes (e.g., 'r=0.3', 'β=-0.2', 'OR=1.5')",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["positive", "negative", "mixed", "unclear"],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "moderate", "low"],
                            "description": "Confidence based on evidence quality",
                        },
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "snippet": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["cause", "effect", "summary", "effect_sizes", "confidence"],
                },
            },
            "general_findings": {
                "type": "string",
                "description": "Overall domain findings relevant to the research question",
            },
        },
        "required": ["question_summary", "relationships", "general_findings"],
    }


async def search_literature(
    question: str,
    edges: list[dict[str, str]],
    model: str = "exa-research",
    timeout_ms: int = 120000,
) -> LiteratureContext | None:
    """Search for literature evidence to ground priors.

    Args:
        question: The research question
        edges: List of causal edges [{cause, effect}, ...]
        model: Exa research model (exa-research-fast, exa-research, exa-research-pro)
        timeout_ms: Max time to wait for research completion

    Returns:
        LiteratureContext with evidence, or None if search fails/no API key
    """
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return None

    exa = AsyncExa(api_key=api_key)

    instructions = _build_research_instructions(question, edges)
    output_schema = _build_output_schema()

    try:
        # Create research task
        research = await exa.research.create(
            instructions=instructions,
            output_schema=output_schema,
            model=model,
        )

        # Poll until complete
        result = await exa.research.poll_until_finished(
            research.id,
            timeout_ms=timeout_ms,
        )

        if result.status != "completed":
            return None

        # Parse structured output
        data = result.data
        if not data:
            return None

        # Build evidence dict
        evidence = {}
        for rel in data.get("relationships", []):
            key = f"{rel['cause']}->{rel['effect']}"
            evidence[key] = LiteratureEvidence(
                cause=rel["cause"],
                effect=rel["effect"],
                summary=rel.get("summary", ""),
                effect_sizes=rel.get("effect_sizes", []),
                sources=rel.get("sources", []),
                confidence=rel.get("confidence", "low"),
            )

        return LiteratureContext(
            question_summary=data.get("question_summary", ""),
            evidence=evidence,
            general_findings=data.get("general_findings", ""),
            raw_response=data,
        )

    except Exception:
        # Don't fail the pipeline if literature search fails
        return None


def format_literature_for_prompt(
    context: LiteratureContext | None,
    edges: list[dict[str, str]],
) -> str:
    """Format literature context for inclusion in prior elicitation prompt.

    Args:
        context: Literature context from search_literature()
        edges: Original edges to ensure all are covered

    Returns:
        Formatted string for prompt augmentation
    """
    if context is None:
        return ""

    lines = ["## Literature Evidence\n"]

    if context.general_findings:
        lines.append(f"**Domain Context:** {context.general_findings}\n")

    lines.append("### Evidence by Relationship\n")

    for edge in edges:
        key = f"{edge['cause']}->{edge['effect']}"
        cause = edge["cause"]
        effect = edge["effect"]

        if key in context.evidence:
            ev = context.evidence[key]
            lines.append(f"**{cause} → {effect}** (confidence: {ev.confidence})")
            lines.append(f"  {ev.summary}")
            if ev.effect_sizes:
                lines.append(f"  Effect sizes: {', '.join(ev.effect_sizes)}")
            if ev.sources:
                sources_text = ", ".join(s.get("title", "Untitled")[:50] for s in ev.sources[:3])
                lines.append(f"  Sources: {sources_text}")
            lines.append("")
        else:
            lines.append(f"**{cause} → {effect}**")
            lines.append("  No direct literature evidence found. Use domain reasoning.")
            lines.append("")

    return "\n".join(lines)
