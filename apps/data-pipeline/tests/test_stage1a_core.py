"""Tests for orchestrator/stage1a.py core logic.

Covers: Stage1aResult properties, Stage1aMessages message building.
"""

from causal_ssm_agent.orchestrator.stage1a import Stage1aMessages, Stage1aResult

# =============================================================================
# Stage1aResult
# =============================================================================


class TestStage1aResult:
    def test_n_constructs(self):
        result = Stage1aResult(
            latent_model={"constructs": [{"name": "A"}, {"name": "B"}], "edges": []},
            outcome_name="",
            treatments=[],
        )
        assert result.n_constructs == 2

    def test_n_edges(self):
        result = Stage1aResult(
            latent_model={
                "constructs": [],
                "edges": [
                    {"cause": "A", "effect": "B"},
                    {"cause": "B", "effect": "C"},
                ],
            },
            outcome_name="",
            treatments=[],
        )
        assert result.n_edges == 2

    def test_empty_model(self):
        result = Stage1aResult(latent_model={}, outcome_name="", treatments=[])
        assert result.n_constructs == 0
        assert result.n_edges == 0

    def test_missing_keys_default_zero(self):
        result = Stage1aResult(latent_model={"constructs": []}, outcome_name="", treatments=[])
        assert result.n_constructs == 0
        assert result.n_edges == 0


# =============================================================================
# Stage1aMessages
# =============================================================================


class TestStage1aMessages:
    def test_proposal_messages_structure(self):
        msgs = Stage1aMessages(question="Does X cause Y?")
        result = msgs.proposal_messages()
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_system_message_content(self):
        msgs = Stage1aMessages(question="test")
        result = msgs.proposal_messages()
        # System prompt should contain key instructions
        system = result[0]["content"]
        assert "causal" in system.lower()
        assert "constructs" in system.lower()
        assert "edges" in system.lower()

    def test_user_message_contains_question(self):
        question = "Does smoking cause cancer?"
        msgs = Stage1aMessages(question=question)
        result = msgs.proposal_messages()
        assert question in result[1]["content"]

    def test_different_questions_produce_different_messages(self):
        m1 = Stage1aMessages(question="Q1").proposal_messages()
        m2 = Stage1aMessages(question="Q2").proposal_messages()
        # System prompts should be identical
        assert m1[0]["content"] == m2[0]["content"]
        # User prompts should differ
        assert m1[1]["content"] != m2[1]["content"]
