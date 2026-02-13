"""Tests for literature search module."""

from causal_ssm_agent.utils.literature_search import (
    LiteratureContext,
    LiteratureEvidence,
    format_literature_for_prompt,
)


class TestFormatLiteratureForPrompt:
    """Tests for formatting literature context into prompts."""

    def test_none_context_returns_empty_string(self):
        """When no literature context, return empty string."""
        edges = [{"cause": "stress", "effect": "mood"}]
        result = format_literature_for_prompt(None, edges)
        assert result == ""

    def test_formats_evidence_for_edges(self):
        """Evidence is formatted per edge."""
        evidence = {
            "stress->mood": LiteratureEvidence(
                cause="stress",
                effect="mood",
                summary="Stress negatively affects mood.",
                effect_sizes=["r=-0.3", "d=-0.5"],
                sources=[
                    {"title": "Meta-analysis 2023", "url": "http://example.com", "snippet": "..."}
                ],
                confidence="high",
            )
        }
        context = LiteratureContext(
            question_summary="Effect of stress on mood",
            evidence=evidence,
            general_findings="Stress is well-established predictor of mood.",
            raw_response={},
        )
        edges = [{"cause": "stress", "effect": "mood"}]

        result = format_literature_for_prompt(context, edges)

        assert "## Literature Evidence" in result
        assert "stress -> mood" in result.lower() or "stress → mood" in result
        assert "high" in result  # confidence
        assert "r=-0.3" in result or "-0.3" in result
        assert "Stress negatively affects mood" in result

    def test_missing_edge_evidence_shows_fallback(self):
        """Edges without evidence show fallback message."""
        context = LiteratureContext(
            question_summary="",
            evidence={},  # No evidence
            general_findings="",
            raw_response={},
        )
        edges = [{"cause": "therapy", "effect": "anxiety"}]

        result = format_literature_for_prompt(context, edges)

        assert "therapy" in result.lower()
        assert "anxiety" in result.lower()
        assert "no direct literature evidence" in result.lower()

    def test_includes_general_findings(self):
        """General findings are included when present."""
        context = LiteratureContext(
            question_summary="",
            evidence={},
            general_findings="Sleep is crucial for mental health.",
            raw_response={},
        )
        edges = [{"cause": "sleep", "effect": "mood"}]

        result = format_literature_for_prompt(context, edges)

        assert "Sleep is crucial for mental health" in result


class TestLiteratureEvidence:
    """Tests for LiteratureEvidence dataclass."""

    def test_basic_instantiation(self):
        """Can create evidence with all fields."""
        ev = LiteratureEvidence(
            cause="X",
            effect="Y",
            summary="X affects Y",
            effect_sizes=["r=0.5"],
            sources=[{"title": "Paper", "url": "http://...", "snippet": "..."}],
            confidence="moderate",
        )

        assert ev.cause == "X"
        assert ev.effect == "Y"
        assert ev.confidence == "moderate"
        assert len(ev.effect_sizes) == 1


class TestLiteratureContext:
    """Tests for LiteratureContext dataclass."""

    def test_basic_instantiation(self):
        """Can create context with all fields."""
        ctx = LiteratureContext(
            question_summary="Testing",
            evidence={},
            general_findings="General info",
            raw_response={"key": "value"},
        )

        assert ctx.question_summary == "Testing"
        assert ctx.evidence == {}
        assert ctx.raw_response == {"key": "value"}
