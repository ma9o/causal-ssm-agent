"""DSPy signature and module for structure proposal optimization."""

import dspy


class StructureProposal(dspy.Signature):
    """Propose a DSEM structure from a natural language question and sample data.

    Output valid JSON with: dimensions, edges.
    """

    question: str = dspy.InputField(desc="Natural language causal research question")
    data_sample: str = dspy.InputField(desc="Sample chunks from the dataset")
    structure: str = dspy.OutputField(
        desc="JSON with dimensions (name, description, variable_type, causal_granularity, base_dtype, aggregation), "
        "edges (cause, effect, lagged, aggregation)"
    )


class StructureProposer(dspy.Module):
    """DSPy module for structure proposal with chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.propose = dspy.ChainOfThought(StructureProposal)

    def forward(self, question: str, data_sample: str) -> dspy.Prediction:
        return self.propose(question=question, data_sample=data_sample)
