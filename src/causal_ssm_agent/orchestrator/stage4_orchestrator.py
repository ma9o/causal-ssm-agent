"""Stage 4 Orchestrator: Model Specification Proposal.

The orchestrator proposes a complete model specification based on the CausalSpec,
enumerating all parameters needing priors with search context for literature.
"""

from causal_ssm_agent.orchestrator.prompts.model_proposal import (
    SYSTEM as MODEL_PROPOSAL_SYSTEM,
)
from causal_ssm_agent.orchestrator.prompts.model_proposal import (
    USER as MODEL_PROPOSAL_USER,
)
from causal_ssm_agent.orchestrator.prompts.model_proposal import (
    format_constructs,
    format_edges,
    format_indicators,
)
from causal_ssm_agent.orchestrator.schemas_model import (
    Stage4OrchestratorResult,
)
from causal_ssm_agent.utils.llm import (
    OrchestratorGenerateFn,
    make_validate_model_spec_tool,
)


async def propose_model_spec(
    causal_spec: dict,
    data_summary: str,
    question: str,
    generate: OrchestratorGenerateFn,
) -> Stage4OrchestratorResult:
    """Orchestrator proposes complete model specification.

    Args:
        causal_spec: The full CausalSpec dict (latent + measurement)
        data_summary: Summary of the data (time points, subjects, etc.)
        question: The research question for context
        generate: Async generate function (messages, tools, follow_ups) -> str

    Returns:
        Stage4OrchestratorResult with ModelSpec
    """
    # Format model components for the prompt
    constructs_str = format_constructs(causal_spec)
    edges_str = format_edges(causal_spec)
    indicators_str = format_indicators(causal_spec)

    # Build messages
    messages = [
        {"role": "system", "content": MODEL_PROPOSAL_SYSTEM},
        {
            "role": "user",
            "content": MODEL_PROPOSAL_USER.format(
                question=question,
                constructs=constructs_str,
                edges=edges_str,
                indicators=indicators_str,
                data_summary=data_summary,
            ),
        },
    ]

    # Generate model specification with validation feedback loop.
    # The tool captures the last valid ModelSpec so we don't need a REVIEW
    # follow-up turn — we just read the captured spec after generate_loop.
    tool, capture = make_validate_model_spec_tool(causal_spec)
    completion = await generate(messages, [tool], None)

    # Use the captured spec from the validation tool (avoids re-parsing LLM output)
    model_spec = capture.get("spec")
    if model_spec is None:
        raise ValueError("Model spec validation never passed. Raw response:\n" + completion[:500])

    return Stage4OrchestratorResult(
        model_spec=model_spec,
        raw_response=completion,
    )


def build_data_summary(measurements_data: dict) -> str:
    """Build a summary of the measurement data for the orchestrator.

    Args:
        measurements_data: Dict of granularity -> polars DataFrame

    Returns:
        Human-readable summary string
    """
    lines = ["### Data Overview\n"]

    for granularity, df in measurements_data.items():
        if granularity == "time_invariant":
            n_indicators = len([c for c in df.columns if c != "time_bucket"])
            lines.append(f"- **Time-invariant**: {n_indicators} indicators")
        else:
            n_rows = df.height
            n_indicators = len([c for c in df.columns if c != "time_bucket"])
            lines.append(f"- **{granularity}**: {n_rows} time points × {n_indicators} indicators")

            # Add basic stats for numeric columns
            if n_rows > 0:
                sample_cols = [c for c in df.columns if c != "time_bucket"][:3]
                for col in sample_cols:
                    try:
                        mean = df[col].mean()
                        std = df[col].std()
                        if mean is not None and std is not None:
                            lines.append(f"    {col}: mean={mean:.2f}, std={std:.2f}")
                    except Exception:
                        pass

    return "\n".join(lines)
