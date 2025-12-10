"""DSPy signature and module for structure proposal optimization."""

from pathlib import Path

import dspy

from causal_agent.orchestrator.prompts import STRUCTURE_PROPOSER_SYSTEM

# Base signature (without instructions)
class StructureProposalBase(dspy.Signature):
    """Propose a DSEM structure from a natural language question and sample data."""

    question: str = dspy.InputField(desc="Natural language causal research question")
    data_sample: str = dspy.InputField(desc="Sample chunks from the dataset")
    structure: str = dspy.OutputField(
        desc="JSON with dimensions (name, description, role, observability, temporal_status, causal_granularity, base_dtype, aggregation), "
        "edges (cause, effect, description, lagged, aggregation)"
    )


# Apply detailed instructions from prompts.py
StructureProposal = StructureProposalBase.with_instructions(STRUCTURE_PROPOSER_SYSTEM)

# Default path for optimized program
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
OPTIMIZED_PROGRAM_PATH = MODELS_DIR / "dspy" / "structure_proposer.json"


class StructureProposer(dspy.Module):
    """DSPy module for structure proposal with chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.propose = dspy.ChainOfThought(StructureProposal)

    def forward(self, question: str, data_sample: str) -> dspy.Prediction:
        return self.propose(question=question, data_sample=data_sample)


def load_structure_proposer(optimized_path: Path | str | None = None) -> StructureProposer:
    """Load StructureProposer, using optimized version if available.

    Args:
        optimized_path: Path to optimized program JSON. Defaults to OPTIMIZED_PROGRAM_PATH.

    Returns:
        StructureProposer with optimized prompts if available, otherwise base version.
    """
    program = StructureProposer()

    if optimized_path is None:
        optimized_path = OPTIMIZED_PROGRAM_PATH

    optimized_path = Path(optimized_path)

    if optimized_path.exists():
        program.load(str(optimized_path))

    return program
