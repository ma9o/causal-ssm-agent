"""Test Stage 1b: Measurement Model with Identifiability Fix.

Tests the tripartite flow:
1. Initial measurement model proposal
2. Identifiability check
3. Proxy request (if needed)
"""

import json
import pytest

from dsem_agent.orchestrator.stage1b import (
    run_stage1b,
    Stage1bResult,
    Stage1bMessages,
    _merge_proxies,
    _get_confounders_to_fix,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES: Dummy Data
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_latent_model():
    """Simple chain: Treatment -> Outcome (all observable)."""
    return {
        "constructs": [
            {
                "name": "Treatment",
                "role": "exogenous",
                "description": "The intervention",
                "temporal_status": "time_invariant",
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
            },
        ],
        "edges": [
            {"cause": "Treatment", "effect": "Outcome", "description": "Treatment causes Outcome"},
        ],
    }


@pytest.fixture
def confounded_latent_model():
    """Confounded graph: Treatment -> Outcome, Confounder -> Treatment, Confounder -> Outcome."""
    return {
        "constructs": [
            {
                "name": "Treatment",
                "role": "endogenous",
                "description": "The intervention",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
            },
            {
                "name": "Confounder",
                "role": "exogenous",
                "description": "Unmeasured common cause",
                "temporal_status": "time_invariant",
            },
        ],
        "edges": [
            {"cause": "Treatment", "effect": "Outcome", "description": "Treatment causes Outcome"},
            {"cause": "Confounder", "effect": "Treatment", "description": "Confounder affects Treatment"},
            {"cause": "Confounder", "effect": "Outcome", "description": "Confounder affects Outcome"},
        ],
    }


@pytest.fixture
def measurement_all_observed():
    """Measurement model with indicators for all constructs."""
    return {
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def measurement_missing_confounder():
    """Measurement model missing the confounder (non-identifiable)."""
    return {
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def measurement_with_confounder():
    """Measurement model with confounder indicator (identifiable)."""
    return {
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "confounder_proxy",
                "construct": "Confounder",
                "how_to_measure": "Proxy measurement for the confounder",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def proxy_response_success():
    """Successful proxy response that adds confounder indicator."""
    return {
        "new_proxies": [
            {
                "construct": "Confounder",
                "indicators": ["confounder_proxy"],
                "justification": "This proxy captures the confounder",
            }
        ],
        "unfeasible_confounders": [],
    }


@pytest.fixture
def proxy_response_empty():
    """Empty proxy response (no proxies found)."""
    return {
        "new_proxies": [],
        "unfeasible_confounders": ["Confounder"],
    }


@pytest.fixture
def dummy_chunks():
    """Dummy data chunks for the measurement model proposal."""
    return [
        "Day 1: Patient took 10mg treatment, outcome score was 5.",
        "Day 2: Patient took 15mg treatment, outcome score was 7.",
        "Day 3: Patient took 10mg treatment, outcome score was 6.",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Mock Generate Functions
# ══════════════════════════════════════════════════════════════════════════════


def make_mock_generate(responses: list[str]):
    """Create a mock generate function that returns predefined responses.

    Args:
        responses: List of JSON strings to return in order

    Returns:
        Async function matching OrchestratorGenerateFn signature
    """
    call_count = [0]  # Use list to allow mutation in closure

    async def mock_generate(messages: list, tools: list | None, follow_ups: list[str] | None) -> str:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    return mock_generate


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Helper Functions
# ══════════════════════════════════════════════════════════════════════════════


class TestMergeProxies:
    """Test _merge_proxies helper function."""

    def test_no_proxy_response(self, measurement_missing_confounder):
        """No changes when proxy response is None."""
        result = _merge_proxies(measurement_missing_confounder, None)
        assert result == measurement_missing_confounder

    def test_empty_proxy_response(self, measurement_missing_confounder, proxy_response_empty):
        """No changes when proxy response has no new proxies."""
        result = _merge_proxies(measurement_missing_confounder, proxy_response_empty)
        assert len(result["indicators"]) == len(measurement_missing_confounder["indicators"])

    def test_merge_adds_indicators(self, measurement_missing_confounder, proxy_response_success):
        """Proxy indicators are added to measurement model."""
        result = _merge_proxies(measurement_missing_confounder, proxy_response_success)

        # Should have one more indicator
        assert len(result["indicators"]) == len(measurement_missing_confounder["indicators"]) + 1

        # New indicator should reference the confounder
        new_indicator = result["indicators"][-1]
        assert new_indicator["construct"] == "Confounder"
        assert "confounder_proxy" in new_indicator["name"]


class TestGetConfoundersToFix:
    """Test _get_confounders_to_fix helper function."""

    def test_extracts_confounders(self, confounded_latent_model):
        """Extracts confounders from identifiability result."""
        id_result = {
            "non_identifiable_treatments": {"Treatment"},
            "blocking_confounders": {"Treatment": ["Confounder"]},
        }

        blocking_info, confounders = _get_confounders_to_fix(id_result, confounded_latent_model)

        assert "Confounder" in confounders
        assert "Treatment" in blocking_info
        assert "Confounder" in blocking_info

    def test_filters_unknown_confounders(self, confounded_latent_model):
        """Filters out confounders not in the latent model."""
        id_result = {
            "non_identifiable_treatments": {"Treatment"},
            "blocking_confounders": {"Treatment": ["UnknownNode", "Confounder"]},
        }

        _, confounders = _get_confounders_to_fix(id_result, confounded_latent_model)

        assert "Confounder" in confounders
        assert "UnknownNode" not in confounders


class TestStage1bMessages:
    """Test Stage1bMessages builder."""

    def test_proposal_messages(self, simple_latent_model, dummy_chunks):
        """Proposal messages include question, latent model, and chunks."""
        msgs = Stage1bMessages(
            question="Does treatment improve outcome?",
            latent_model=simple_latent_model,
            chunks=dummy_chunks,
        )

        messages = msgs.proposal_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "treatment improve outcome" in messages[1]["content"]
        assert "Treatment" in messages[1]["content"]

    def test_proxy_messages(self, confounded_latent_model, measurement_missing_confounder, dummy_chunks):
        """Proxy messages include blocking info and confounders."""
        msgs = Stage1bMessages(
            question="Does treatment improve outcome?",
            latent_model=confounded_latent_model,
            chunks=dummy_chunks,
        )

        messages = msgs.proxy_messages(
            blocking_info="Treatment: blocked by Confounder",
            confounders=["Confounder"],
            current_measurement=measurement_missing_confounder,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Confounder" in messages[1]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS: Full Stage 1b Flow
# ══════════════════════════════════════════════════════════════════════════════


class TestStage1bFlow:
    """Integration tests for the full Stage 1b flow."""

    def test_all_identifiable_no_proxy(
        self, simple_latent_model, measurement_all_observed, dummy_chunks
    ):
        """When all effects are identifiable, no proxy request is made."""
        import asyncio

        mock_generate = make_mock_generate([
            json.dumps(measurement_all_observed),  # Initial proposal
        ])

        result = asyncio.run(run_stage1b(
            question="Does treatment improve outcome?",
            latent_model=simple_latent_model,
            chunks=dummy_chunks,
            generate=mock_generate,
        ))

        assert isinstance(result, Stage1bResult)
        assert result.proxy_requested is False
        assert result.proxy_response is None
        assert len(result.final_identifiability["non_identifiable_treatments"]) == 0

    def test_non_identifiable_triggers_proxy_request(
        self, confounded_latent_model, measurement_missing_confounder,
        measurement_with_confounder, proxy_response_success, dummy_chunks
    ):
        """Non-identifiable effects trigger a proxy request."""
        import asyncio

        mock_generate = make_mock_generate([
            json.dumps(measurement_missing_confounder),  # Initial proposal (missing confounder)
            json.dumps(proxy_response_success),  # Proxy response
        ])

        result = asyncio.run(run_stage1b(
            question="Does treatment improve outcome?",
            latent_model=confounded_latent_model,
            chunks=dummy_chunks,
            generate=mock_generate,
        ))

        assert result.proxy_requested is True
        assert result.proxy_response is not None
        # Initial should be non-identifiable
        assert len(result.initial_identifiability["non_identifiable_treatments"]) > 0

    def test_proxy_fixes_identifiability(
        self, confounded_latent_model, measurement_missing_confounder,
        proxy_response_success, dummy_chunks
    ):
        """Proxy response fixes identifiability when it adds the right indicator."""
        import asyncio

        # The proxy adds a confounder indicator
        proxy_with_full_indicator = {
            "new_proxies": [
                {
                    "construct": "Confounder",
                    "indicators": ["confounder_proxy"],
                    "justification": "This proxy captures the confounder",
                }
            ],
            "unfeasible_confounders": [],
        }

        mock_generate = make_mock_generate([
            json.dumps(measurement_missing_confounder),  # Initial proposal
            json.dumps(proxy_with_full_indicator),  # Proxy response
        ])

        result = asyncio.run(run_stage1b(
            question="Does treatment improve outcome?",
            latent_model=confounded_latent_model,
            chunks=dummy_chunks,
            generate=mock_generate,
        ))

        # The measurement model should now include the proxy indicator
        indicator_constructs = {
            ind.get("construct") or ind.get("construct_name")
            for ind in result.measurement_model["indicators"]
        }
        assert "Confounder" in indicator_constructs

        # After proxy, should be identifiable
        assert len(result.final_identifiability["non_identifiable_treatments"]) == 0

    def test_proxy_fails_to_fix(
        self, confounded_latent_model, measurement_missing_confounder,
        proxy_response_empty, dummy_chunks
    ):
        """When proxy fails, identifiability remains non-identifiable."""
        import asyncio

        mock_generate = make_mock_generate([
            json.dumps(measurement_missing_confounder),  # Initial proposal
            json.dumps(proxy_response_empty),  # Empty proxy response
        ])

        result = asyncio.run(run_stage1b(
            question="Does treatment improve outcome?",
            latent_model=confounded_latent_model,
            chunks=dummy_chunks,
            generate=mock_generate,
        ))

        assert result.proxy_requested is True
        # Still non-identifiable after proxy attempt
        assert len(result.final_identifiability["non_identifiable_treatments"]) > 0

    def test_identifiability_status_property(
        self, simple_latent_model, measurement_all_observed, dummy_chunks
    ):
        """The identifiability_status property formats correctly."""
        import asyncio

        mock_generate = make_mock_generate([
            json.dumps(measurement_all_observed),
        ])

        result = asyncio.run(run_stage1b(
            question="Does treatment improve outcome?",
            latent_model=simple_latent_model,
            chunks=dummy_chunks,
            generate=mock_generate,
        ))

        status = result.identifiability_status

        assert "outcome" in status
        assert "all_identifiable" in status
        assert "non_identifiable_treatments" in status
        assert "blocking_confounders" in status
        assert status["all_identifiable"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
