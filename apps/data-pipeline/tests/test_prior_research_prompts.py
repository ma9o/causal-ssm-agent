"""Tests for prior research prompt formatting functions."""

from causal_ssm_agent.workers.prompts.prior_research import (
    NO_LITERATURE,
    PARAPHRASE_TEMPLATES,
    format_literature_for_parameter,
    generate_paraphrased_prompts,
)

# =============================================================================
# format_literature_for_parameter
# =============================================================================


class TestFormatLiteratureForParameter:
    def test_empty_sources_returns_no_literature(self):
        result = format_literature_for_parameter([])
        assert result == NO_LITERATURE

    def test_single_source(self):
        sources = [
            {
                "title": "Meta-analysis of stress and sleep",
                "url": "https://example.com",
                "snippet": "d=0.5 across 20 studies",
                "effect_size": "d=0.5",
            }
        ]
        result = format_literature_for_parameter(sources)
        assert "Meta-analysis of stress and sleep" in result
        assert "https://example.com" in result
        assert "d=0.5 across 20 studies" in result
        assert "d=0.5" in result
        assert "Source 1" in result

    def test_multiple_sources_numbered(self):
        sources = [
            {"title": "Study A"},
            {"title": "Study B"},
            {"title": "Study C"},
        ]
        result = format_literature_for_parameter(sources)
        assert "Source 1" in result
        assert "Source 2" in result
        assert "Source 3" in result

    def test_missing_optional_fields(self):
        sources = [{"title": "Minimal source"}]
        result = format_literature_for_parameter(sources)
        assert "Minimal source" in result
        assert "URL:" not in result  # no url field

class TestGenerateParaphrasedPrompts:
    def test_cap_at_template_count(self):
        prompts = generate_paraphrased_prompts(
            parameter_name="x",
            parameter_role="r",
            parameter_constraint="c",
            parameter_description="d",
            question="q",
            literature_context="l",
            n_paraphrases=100,
        )
        assert len(prompts) == len(PARAPHRASE_TEMPLATES)

    def test_each_prompt_has_context(self):
        prompts = generate_paraphrased_prompts(
            parameter_name="sigma_Y",
            parameter_role="residual_sd",
            parameter_constraint="positive",
            parameter_description="Residual SD for Y",
            question="What is the noise level?",
            literature_context="Some evidence here.",
            n_paraphrases=3,
        )
        for prompt in prompts:
            assert "sigma_Y" in prompt
            assert "residual_sd" in prompt
            assert "positive" in prompt
            assert "Some evidence here." in prompt
