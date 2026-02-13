"""Stage 1b: Measurement Model with Identifiability Fix.

Core logic for Stage 1b, decoupled from Prefect/Inspect frameworks.
Uses dependency injection for the LLM generate function.
"""

import json
from dataclasses import dataclass

from causal_ssm_agent.utils.identifiability import (
    analyze_unobserved_constructs,
    check_identifiability,
)
from causal_ssm_agent.utils.llm import (
    OrchestratorGenerateFn,
    make_validate_measurement_model_tool,
    parse_json_response,
)

from .prompts import measurement_model
from .schemas import LatentModel, MeasurementModel


@dataclass
class Stage1bResult:
    """Result of Stage 1b: measurement model with identifiability status."""

    measurement_model: dict
    initial_identifiability: dict
    final_identifiability: dict
    proxy_requested: bool
    proxy_response: dict | None = None
    marginalization_analysis: dict | None = None

    @property
    def identifiability_status(self) -> dict:
        """Format for storing in CausalSpec."""
        status = {
            "identifiable_treatments": self.final_identifiability["identifiable_treatments"],
            "non_identifiable_treatments": self.final_identifiability[
                "non_identifiable_treatments"
            ],
        }
        if "graph_info" in self.final_identifiability:
            status["graph_info"] = self.final_identifiability["graph_info"]
        return status

    @property
    def can_marginalize(self) -> list[str]:
        """Unobserved constructs that can be ignored in causal specification."""
        if self.marginalization_analysis:
            return sorted(self.marginalization_analysis.get("can_marginalize", []))
        return []

    @property
    def needs_modeling(self) -> set[str]:
        """Unobserved constructs that need explicit modeling (block identification)."""
        final_id = self.final_identifiability or {}
        needs = set()
        non_identifiable = final_id.get("non_identifiable_treatments", {})
        for info in non_identifiable.values():
            if not isinstance(info, dict):
                continue
            for conf in info.get("confounders", []):
                if conf:
                    needs.add(conf)
        return needs


@dataclass
class Stage1bMessages:
    """Message builders for Stage 1b prompts."""

    question: str
    latent_model: dict
    chunks: list[str]
    dataset_summary: str = ""

    def proposal_messages(self) -> list[dict]:
        """Build messages for initial measurement proposal."""
        return [
            {"role": "system", "content": measurement_model.SYSTEM},
            {
                "role": "user",
                "content": measurement_model.USER.format(
                    question=self.question,
                    latent_model_json=json.dumps(self.latent_model, indent=2),
                    dataset_summary=self.dataset_summary or "Not provided",
                    chunks="\n".join(self.chunks),
                ),
            },
        ]

    def proxy_messages(
        self,
        blocking_info: str,
        confounders: list[str],
        current_measurement: dict,
    ) -> list[dict]:
        """Build messages for proxy request."""
        return [
            {"role": "system", "content": measurement_model.PROXY_SYSTEM},
            {
                "role": "user",
                "content": measurement_model.PROXY_USER.format(
                    blocking_info=blocking_info,
                    confounders_to_operationalize=", ".join(confounders),
                    latent_model_json=json.dumps(self.latent_model, indent=2),
                    current_measurements_json=json.dumps(current_measurement, indent=2),
                    data_sample="\n".join(self.chunks[:5]),
                ),
            },
        ]


def _merge_proxies(measurement: dict, proxy_response: dict | None) -> dict:
    """Merge proxy indicators into measurement model.

    Handles two formats for proxy indicators:
    1. List of strings (indicator names) - as requested in prompt
    2. List of full indicator objects - what models sometimes produce
    """
    if not proxy_response or not proxy_response.get("new_proxies"):
        return measurement

    # Copy to avoid mutation
    result = dict(measurement)
    result["indicators"] = list(measurement.get("indicators", []))

    for proxy in proxy_response["new_proxies"]:
        for indicator in proxy.get("indicators", []):
            if isinstance(indicator, dict):
                # Model returned full indicator object - use it directly
                # but ensure construct is set correctly
                ind = dict(indicator)
                ind["construct_name"] = proxy["construct"]
                # Prepend proxy justification to how_to_measure if provided
                if proxy.get("justification") and "how_to_measure" in ind:
                    ind["how_to_measure"] = (
                        f"Proxy for {proxy['construct']}: {ind['how_to_measure']}"
                    )
                # Fill in defaults for any missing required fields
                ind.setdefault("measurement_granularity", "finest")
                ind.setdefault("measurement_dtype", "continuous")
                ind.setdefault("aggregation", "mean")
                result["indicators"].append(ind)
            else:
                # Model returned just indicator name (string) - build complete indicator
                # with safe defaults for required fields
                result["indicators"].append(
                    {
                        "name": indicator,
                        "construct_name": proxy["construct"],
                        "how_to_measure": f"Proxy for {proxy['construct']}: {proxy.get('justification', '')}",
                        "measurement_granularity": "finest",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                )

    return result


def _get_confounders_to_fix(
    id_result: dict,
    latent_model: dict,
) -> tuple[str, list[str]]:
    """Extract blocking info and confounders that need proxies.

    Returns:
        Tuple of (blocking_info_string, list_of_confounder_names)
    """
    non_identifiable = id_result.get("non_identifiable_treatments", {})
    all_confounders = set()
    for info in non_identifiable.values():
        if not isinstance(info, dict):
            continue
        all_confounders.update(info.get("confounders", []))

    # Filter to actual constructs (not "unknown" errors)
    construct_names = {c["name"] for c in latent_model["constructs"]}
    confounders_to_fix = [c for c in all_confounders if c in construct_names]

    # Format blocking info
    blocking_lines = []
    for treatment in sorted(non_identifiable.keys()):
        details = non_identifiable[treatment]
        if not isinstance(details, dict):
            continue
        blockers = details.get("confounders", [])
        notes = details.get("notes")
        if blockers:
            blocking_lines.append(f"- {treatment}: blocked by {', '.join(blockers)}")
        elif notes:
            blocking_lines.append(f"- {treatment}: {notes}")
    blocking_info = "\n".join(blocking_lines)

    return blocking_info, confounders_to_fix


async def run_stage1b(
    question: str,
    latent_model: dict,
    chunks: list[str],
    generate: OrchestratorGenerateFn,
    dataset_summary: str = "",
) -> Stage1bResult:
    """
    Run the full Stage 1b flow: proposal → identifiability check → proxy fix.

    This is the core logic, decoupled from any framework. The caller provides
    a `generate` function that handles LLM calls.

    Args:
        question: The causal research question
        latent_model: The latent model dict from Stage 1a
        chunks: Data chunks for operationalization
        generate: Async function (messages, tools, follow_ups) -> completion
        dataset_summary: Optional description of the dataset

    Returns:
        Stage1bResult with measurement model and identifiability info
    """
    msgs = Stage1bMessages(question, latent_model, chunks, dataset_summary)
    latent = LatentModel.model_validate(latent_model)

    # Step 1: Initial proposal with self-review
    proposal_msgs = msgs.proposal_messages()
    tools = [make_validate_measurement_model_tool(latent)]

    completion = await generate(proposal_msgs, tools, [measurement_model.REVIEW])

    # Parse measurement model
    measurement = parse_json_response(completion)
    MeasurementModel.model_validate(measurement)  # Validate schema

    # Step 2: Check identifiability
    initial_id = check_identifiability(latent_model, measurement)

    # Step 3: If non-identifiable, request proxies (one-shot)
    proxy_response = None
    if initial_id["non_identifiable_treatments"]:
        blocking_info, confounders_to_fix = _get_confounders_to_fix(initial_id, latent_model)

        if confounders_to_fix:
            proxy_msgs = msgs.proxy_messages(blocking_info, confounders_to_fix, measurement)

            # Proxy request is single-turn, no tools or follow-ups
            proxy_completion = await generate(proxy_msgs, None, None)

            try:
                proxy_response = parse_json_response(proxy_completion)
            except Exception:
                proxy_response = None

            if proxy_response and proxy_response.get("new_proxies"):
                measurement = _merge_proxies(measurement, proxy_response)
                # Re-validate after merge to catch schema violations
                MeasurementModel.model_validate(measurement)

    # Step 4: Final identifiability check
    final_id = check_identifiability(latent_model, measurement)

    # Step 5: Analyze which unobserved constructs can be marginalized
    marginalization = analyze_unobserved_constructs(latent_model, measurement, final_id)

    return Stage1bResult(
        measurement_model=measurement,
        initial_identifiability=initial_id,
        final_identifiability=final_id,
        proxy_requested=bool(initial_id["non_identifiable_treatments"]),
        proxy_response=proxy_response,
        marginalization_analysis=marginalization,
    )
