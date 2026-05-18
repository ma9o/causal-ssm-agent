"""Tests for prior research prompt helper behavior."""

from nof1_causal_lab.workers.prompts.prior_research import (
    generate_paraphrased_prompts,
)


class TestGenerateParaphrasedPrompts:
    def test_cap_at_ten_distinct_prompts(self):
        prompts = generate_paraphrased_prompts(
            parameter_name="x",
            parameter_role="r",
            parameter_constraint="c",
            parameter_description="d",
            question="q",
            literature_context="l",
            n_paraphrases=100,
        )
        assert len(prompts) == 10
        assert len(set(prompts)) == 10

    def test_injects_shared_context_once_per_prompt(self):
        prompts = generate_paraphrased_prompts(
            parameter_name="sigma_Y",
            parameter_role="residual_sd",
            parameter_constraint="positive",
            parameter_description="Residual SD for Y",
            question="What is the noise level?",
            literature_context="Some evidence here.",
            n_paraphrases=3,
        )

        assert len(prompts) == 3
        assert len(set(prompts)) == 3

        for prompt in prompts:
            assert prompt.count("**Name**: sigma_Y") == 1
            assert prompt.count("**Role**: residual_sd") == 1
            assert prompt.count("**Constraint**: positive") == 1
            assert prompt.count("## Research Context") == 1
            assert prompt.count("## Literature Evidence") == 1
            assert "Some evidence here." in prompt
