"""Stage 4 Orchestrator: Model Specification Proposal.

The orchestrator proposes a complete Bayesian hierarchical model specification
based on the DSEMModel, enumerating all parameters needing priors with search
context for literature.
"""

from dsem_agent.orchestrator.prompts.model_proposal import (
    SYSTEM as MODEL_PROPOSAL_SYSTEM,
)
from dsem_agent.orchestrator.prompts.model_proposal import (
    USER as MODEL_PROPOSAL_USER,
)
from dsem_agent.orchestrator.prompts.model_proposal import (
    format_constructs,
    format_edges,
    format_indicators,
)
from dsem_agent.orchestrator.schemas_model import (
    ModelSpec,
    Stage4OrchestratorResult,
)
from dsem_agent.utils.llm import OrchestratorGenerateFn, parse_json_response


async def propose_model_spec(
    dsem_model: dict,
    data_summary: str,
    question: str,
    generate: OrchestratorGenerateFn,
) -> Stage4OrchestratorResult:
    """Orchestrator proposes complete model specification.

    Args:
        dsem_model: The full DSEMModel dict (latent + measurement)
        data_summary: Summary of the data (time points, subjects, etc.)
        question: The research question for context
        generate: Async generate function (messages, tools, follow_ups) -> str

    Returns:
        Stage4OrchestratorResult with ModelSpec
    """
    # Format model components for the prompt
    constructs_str = format_constructs(dsem_model)
    edges_str = format_edges(dsem_model)
    indicators_str = format_indicators(dsem_model)

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

    # Generate model specification
    completion = await generate(messages, None, None)

    # Parse JSON response
    model_data = parse_json_response(completion)

    # Validate into ModelSpec
    model_spec = ModelSpec.model_validate(model_data)

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
            n_indicators = len([c for c in df.columns if c not in ("time_bucket", "subject_id")])
            lines.append(f"- **{granularity}**: {n_rows} time points × {n_indicators} indicators")

            # Add basic stats for numeric columns
            if n_rows > 0:
                sample_cols = [c for c in df.columns if c not in ("time_bucket", "subject_id")][:3]
                for col in sample_cols:
                    try:
                        mean = df[col].mean()
                        std = df[col].std()
                        if mean is not None and std is not None:
                            lines.append(f"    {col}: mean={mean:.2f}, std={std:.2f}")
                    except Exception:
                        pass

    return "\n".join(lines)
