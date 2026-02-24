"""Test Stage 1b: Measurement Model with Identifiability Fix.

Tests the tripartite flow:
1. Initial measurement model proposal
2. Identifiability check
3. Proxy request (if needed)
"""

import asyncio
import json

import pytest

from causal_ssm_agent.orchestrator.stage1b import (
    Stage1bMessages,
    Stage1bResult,
    _get_confounders_to_fix,
    _merge_proxies,
    run_stage1b,
)
from tests.helpers import make_mock_generate

# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Helper Functions
# ══════════════════════════════════════════════════════════════════════════════


class TestMergeProxies:
    """Test _merge_proxies helper function."""

    def test_no_proxy_response(self, stage1b_measurement_missing_confounder):
        """No changes when proxy response is None."""
        result = _merge_proxies(stage1b_measurement_missing_confounder, None)
        assert result == stage1b_measurement_missing_confounder

    def test_empty_proxy_response(
        self, stage1b_measurement_missing_confounder, stage1b_proxy_response_empty
    ):
        """No changes when proxy response has no new proxies."""
        result = _merge_proxies(
            stage1b_measurement_missing_confounder, stage1b_proxy_response_empty
        )
        assert len(result["indicators"]) == len(
            stage1b_measurement_missing_confounder["indicators"]
        )

    def test_merge_adds_indicators(
        self, stage1b_measurement_missing_confounder, stage1b_proxy_response_success
    ):
        """Proxy indicators are added to measurement model."""
        result = _merge_proxies(
            stage1b_measurement_missing_confounder, stage1b_proxy_response_success
        )

        # Should have one more indicator
        assert (
            len(result["indicators"])
            == len(stage1b_measurement_missing_confounder["indicators"]) + 1
        )

        # New indicator should reference the confounder
        new_indicator = result["indicators"][-1]
        assert new_indicator["construct_name"] == "Confounder"
        assert "confounder_proxy" in new_indicator["name"]

    def test_merge_handles_full_indicator_objects(self, stage1b_measurement_missing_confounder):
        """Proxy response with full indicator objects (not just names) is handled correctly."""
        # This is what models sometimes produce - full indicator specs instead of just names
        proxy_response = {
            "new_proxies": [
                {
                    "construct": "Confounder",
                    "indicators": [
                        {
                            "name": "confounder_proxy",
                            "how_to_measure": "Extract proxy value from data",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        }
                    ],
                    "justification": "This proxy captures the confounder",
                }
            ],
        }

        result = _merge_proxies(stage1b_measurement_missing_confounder, proxy_response)

        # Should have one more indicator
        assert (
            len(result["indicators"])
            == len(stage1b_measurement_missing_confounder["indicators"]) + 1
        )

        # New indicator should have all fields from the full object
        new_indicator = result["indicators"][-1]
        assert new_indicator["name"] == "confounder_proxy"
        assert new_indicator["construct_name"] == "Confounder"
        assert new_indicator["measurement_dtype"] == "continuous"
        assert new_indicator["aggregation"] == "mean"
        # how_to_measure should be prepended with proxy justification
        assert "Proxy for Confounder:" in new_indicator["how_to_measure"]


class TestGetConfoundersToFix:
    """Test _get_confounders_to_fix helper function."""

    def test_extracts_confounders(self, stage1b_confounded_latent):
        """Extracts confounders from identifiability result."""
        id_result = {
            "non_identifiable_treatments": {"Treatment": {"confounders": ["Confounder"]}},
        }

        blocking_info, confounders = _get_confounders_to_fix(id_result, stage1b_confounded_latent)

        assert "Confounder" in confounders
        assert "Treatment" in blocking_info
        assert "Confounder" in blocking_info

    def test_filters_unknown_confounders(self, stage1b_confounded_latent):
        """Filters out confounders not in the latent model."""
        id_result = {
            "non_identifiable_treatments": {
                "Treatment": {"confounders": ["UnknownNode", "Confounder"]}
            },
        }

        _, confounders = _get_confounders_to_fix(id_result, stage1b_confounded_latent)

        assert "Confounder" in confounders
        assert "UnknownNode" not in confounders


class TestStage1bMessages:
    """Test Stage1bMessages builder."""

    def test_proposal_messages(self, stage1b_simple_latent, stage1b_dummy_chunks):
        """Proposal messages include question, latent model, and chunks."""
        msgs = Stage1bMessages(
            question="Does treatment improve outcome?",
            latent_model=stage1b_simple_latent,
            chunks=stage1b_dummy_chunks,
        )

        messages = msgs.proposal_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "treatment improve outcome" in messages[1]["content"]
        assert "Treatment" in messages[1]["content"]

    def test_proxy_messages(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_missing_confounder,
        stage1b_dummy_chunks,
    ):
        """Proxy messages include blocking info and confounders."""
        msgs = Stage1bMessages(
            question="Does treatment improve outcome?",
            latent_model=stage1b_confounded_latent,
            chunks=stage1b_dummy_chunks,
        )

        messages = msgs.proxy_messages(
            blocking_info="Treatment: blocked by Confounder",
            confounders=["Confounder"],
            current_measurement=stage1b_measurement_missing_confounder,
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
        self,
        stage1b_simple_latent,
        stage1b_measurement_all_observed,
        stage1b_dummy_chunks,
    ):
        """When all effects are identifiable, no proxy request is made."""
        mock_generate = make_mock_generate(
            [
                json.dumps(stage1b_measurement_all_observed),  # Initial proposal
            ]
        )

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_simple_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        assert isinstance(result, Stage1bResult)
        assert result.proxy_requested is False
        assert result.proxy_response is None
        assert len(result.final_identifiability["non_identifiable_treatments"]) == 0

    def test_non_identifiable_triggers_proxy_request(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_missing_confounder,
        stage1b_measurement_with_confounder,
        stage1b_proxy_response_success,
        stage1b_dummy_chunks,
    ):
        """Non-identifiable effects trigger a proxy request."""
        mock_generate = make_mock_generate(
            [
                json.dumps(
                    stage1b_measurement_missing_confounder
                ),  # Initial proposal (missing confounder)
                json.dumps(stage1b_proxy_response_success),  # Proxy response
            ]
        )

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_confounded_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        assert result.proxy_requested is True
        assert result.proxy_response is not None
        # Initial should be non-identifiable
        assert len(result.initial_identifiability["non_identifiable_treatments"]) > 0

    def test_proxy_fixes_identifiability(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_missing_confounder,
        stage1b_proxy_response_success,
        stage1b_dummy_chunks,
    ):
        """Proxy response fixes identifiability when it adds the right indicator."""
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

        mock_generate = make_mock_generate(
            [
                json.dumps(stage1b_measurement_missing_confounder),  # Initial proposal
                json.dumps(proxy_with_full_indicator),  # Proxy response
            ]
        )

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_confounded_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        # The measurement model should now include the proxy indicator
        indicator_constructs = {
            ind.get("construct_name") for ind in result.measurement_model["indicators"]
        }
        assert "Confounder" in indicator_constructs

        # After proxy, should be identifiable
        assert len(result.final_identifiability["non_identifiable_treatments"]) == 0

    def test_proxy_fails_to_fix(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_missing_confounder,
        stage1b_proxy_response_empty,
        stage1b_dummy_chunks,
    ):
        """When proxy fails, identifiability remains non-identifiable."""
        mock_generate = make_mock_generate(
            [
                json.dumps(stage1b_measurement_missing_confounder),  # Initial proposal
                json.dumps(stage1b_proxy_response_empty),  # Empty proxy response
            ]
        )

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_confounded_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        assert result.proxy_requested is True
        # Still non-identifiable after proxy attempt
        assert len(result.final_identifiability["non_identifiable_treatments"]) > 0

    def test_identifiability_status_property(
        self,
        stage1b_simple_latent,
        stage1b_measurement_all_observed,
        stage1b_dummy_chunks,
    ):
        """The identifiability_status property formats correctly."""
        mock_generate = make_mock_generate(
            [
                json.dumps(stage1b_measurement_all_observed),
            ]
        )

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_simple_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        status = result.identifiability_status

        assert "identifiable_treatments" in status
        assert "non_identifiable_treatments" in status
        assert isinstance(status["identifiable_treatments"], dict)
        assert isinstance(status["non_identifiable_treatments"], dict)
        assert len(status["non_identifiable_treatments"]) == 0

    def test_marginalization_analysis_included(
        self,
        stage1b_confounded_latent,
        stage1b_measurement_missing_confounder,
        stage1b_dummy_chunks,
    ):
        """Marginalization analysis is computed and accessible."""
        proxy_response = {
            "new_proxies": [],
            "unfeasible_confounders": ["Confounder"],
        }

        mock_generate = make_mock_generate(
            [
                json.dumps(stage1b_measurement_missing_confounder),
                json.dumps(proxy_response),
            ]
        )

        result = asyncio.run(
            run_stage1b(
                question="Does treatment improve outcome?",
                latent_model=stage1b_confounded_latent,
                chunks=stage1b_dummy_chunks,
                generate=mock_generate,
            )
        )

        # Marginalization analysis should be present
        assert result.marginalization_analysis is not None
        assert "can_marginalize" in result.marginalization_analysis
        assert "blocking_details" in result.marginalization_analysis

        # Confounder blocks identification, so it needs modeling
        assert "Confounder" in result.needs_modeling
        assert len(result.can_marginalize) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
