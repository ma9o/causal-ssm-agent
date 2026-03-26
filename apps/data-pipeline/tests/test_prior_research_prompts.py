"""Tests for prior research prompt formatting functions.

Covers: format_literature_for_parameter, generate_paraphrased_prompts.
"""

from causal_ssm_agent.distributions import render_prior_parameter_guidance_markdown_table
from causal_ssm_agent.workers.prompts.prior_research import (
    NO_LITERATURE,
    PARAPHRASE_TEMPLATES,
    SYSTEM,
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

    def test_header_present(self):
        sources = [{"title": "Any"}]
        result = format_literature_for_parameter(sources)
        assert "Relevant Literature" in result


# =============================================================================
# generate_paraphrased_prompts
# =============================================================================


class TestGenerateParaphrasedPrompts:
    def test_single_paraphrase(self):
        prompts = generate_paraphrased_prompts(
            parameter_name="beta_X",
            parameter_role="fixed_effect",
            parameter_constraint="none",
            parameter_description="Effect of X on Y",
            question="How does X affect Y?",
            literature_context="No literature found.",
            n_paraphrases=1,
        )
        assert len(prompts) == 1
        assert "beta_X" in prompts[0]
        assert "fixed_effect" in prompts[0]

    def test_all_paraphrases(self):
        prompts = generate_paraphrased_prompts(
            parameter_name="rho",
            parameter_role="ar_coefficient",
            parameter_constraint="unit_interval",
            parameter_description="AR(1) coefficient",
            question="test",
            literature_context="test",
            n_paraphrases=10,
        )
        assert len(prompts) == len(PARAPHRASE_TEMPLATES)

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

    def test_prompts_differ(self):
        """Each prompt should use a different paraphrase template."""
        prompts = generate_paraphrased_prompts(
            parameter_name="x",
            parameter_role="r",
            parameter_constraint="c",
            parameter_description="d",
            question="q",
            literature_context="l",
            n_paraphrases=3,
        )
        # The tail (instruction part) should differ
        assert len(set(prompts)) == 3


class TestPromptContracts:
    def test_worker_prompt_mentions_reference_interval_days_for_lagged_beta(self):
        assert "reference_interval_days" in SYSTEM
        assert "authored interval scale" in SYSTEM
        assert "authored on the interval they mean" in SYSTEM

    def test_prior_guidance_table_uses_authored_interval_label_for_beta(self):
        table = render_prior_parameter_guidance_markdown_table()
        assert "Authored interval effect" in table
        assert "`reference_interval_days`" in table
