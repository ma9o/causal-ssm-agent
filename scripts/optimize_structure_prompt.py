#!/usr/bin/env python
"""Optimize the structure proposer prompt using DSPy MIPROv2.

Usage:
    uv run python scripts/optimize_structure_prompt.py

Outputs:
    - Optimized prompt printed to stdout
    - Compiled module saved to data/optimization/structure_proposer_optimized.json
"""

import json
import os
from pathlib import Path

import dspy
import networkx as nx
from dotenv import load_dotenv

from causal_agent.orchestrator.dspy_module import StructureProposer
from causal_agent.orchestrator.schemas import ProposedStructure

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

# Paths
EXAMPLES_PATH = Path(__file__).parent.parent / "data/optimization/structure_examples.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data/optimization/structure_proposer_optimized.json"


def load_examples() -> list[dspy.Example]:
    """Load training examples from JSONL file."""
    examples = []
    with open(EXAMPLES_PATH) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                ex = dspy.Example(
                    question=data["question"],
                    data_sample=data["data_sample"],
                    hints=data.get("hints", {}),
                ).with_inputs("question", "data_sample")
                examples.append(ex)
    return examples


def validate_structure(structure_json: str, hints: dict) -> tuple[float, str]:
    """
    Validate a structure proposal.

    Returns:
        (score, reason) where score is 0-1
    """
    # Parse JSON
    try:
        if "```json" in structure_json:
            structure_json = structure_json.split("```json")[1].split("```")[0]
        elif "```" in structure_json:
            structure_json = structure_json.split("```")[1].split("```")[0]
        data = json.loads(structure_json.strip())
    except json.JSONDecodeError as e:
        return 0.0, f"Invalid JSON: {e}"

    # Validate schema
    try:
        structure = ProposedStructure.model_validate(data)
    except Exception as e:
        return 0.0, f"Schema validation failed: {e}"

    # Check DAG is acyclic
    G = structure.to_networkx()
    if not nx.is_directed_acyclic_graph(G):
        return 0.0, "Graph contains cycles"

    # Check edges reference defined dimensions
    dim_names = {d.name for d in structure.dimensions}
    for edge in structure.edges:
        if edge.cause not in dim_names:
            return 0.0, f"Edge cause '{edge.cause}' not in dimensions"
        if edge.effect not in dim_names:
            return 0.0, f"Edge effect '{edge.effect}' not in dimensions"

    # Check no orphan dimensions (every dim must be in at least one edge)
    nodes_in_edges = {e.cause for e in structure.edges} | {e.effect for e in structure.edges}
    orphans = dim_names - nodes_in_edges
    if orphans:
        return 0.0, f"Orphan dimensions not in any edge: {orphans}"

    # Base score for valid structure
    score = 0.5

    # Check hints if provided
    if hints:
        # Must include dimensions
        must_include = set(hints.get("must_include_dimensions", []))
        found_dims = {d.name for d in structure.dimensions}
        # Fuzzy match: check if any dimension name contains the hint
        matched = 0
        for hint in must_include:
            hint_lower = hint.lower()
            if any(hint_lower in d.lower() or d.lower() in hint_lower for d in found_dims):
                matched += 1
        if must_include:
            score += 0.25 * (matched / len(must_include))

        # Must include edges
        must_include_edges = hints.get("must_include_edges", [])
        edge_set = {(e.cause.lower(), e.effect.lower()) for e in structure.edges}
        edge_matched = 0
        for cause, effect in must_include_edges:
            cause_lower, effect_lower = cause.lower(), effect.lower()
            # Fuzzy match edges too
            if any(
                (cause_lower in c or c in cause_lower) and (effect_lower in e or e in effect_lower)
                for c, e in edge_set
            ):
                edge_matched += 1
        if must_include_edges:
            score += 0.15 * (edge_matched / len(must_include_edges))

        # Forbidden dimensions (penalize)
        forbidden = set(hints.get("forbidden_dimensions", []))
        for dim in found_dims:
            if any(f.lower() in dim.lower() for f in forbidden):
                score -= 0.1

    return min(max(score, 0.0), 1.0), "OK"


def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric function for MIPROv2 optimization."""
    hints = getattr(example, "hints", {})
    score, reason = validate_structure(pred.structure, hints)
    if trace is not None and score < 0.5:
        print(f"  Low score ({score:.2f}): {reason}")
    return score


def main():
    # Configure DSPy with OpenRouter
    lm = dspy.LM(
        "openrouter/google/gemini-2.5-flash-preview-05-20",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    dspy.configure(lm=lm)

    # Load examples
    examples = load_examples()
    print(f"Loaded {len(examples)} examples")

    if len(examples) < 2:
        print("Warning: Need more examples for effective optimization (recommend 10+)")

    # Split: 20% train, 80% val (DSPy recommendation)
    split_idx = max(1, len(examples) // 5)
    trainset = examples[:split_idx]
    valset = examples[split_idx:] if len(examples) > split_idx else examples

    print(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Initialize module
    proposer = StructureProposer()

    # Run MIPROv2
    print("\nStarting MIPROv2 optimization...")
    optimizer = dspy.MIPROv2(
        metric=metric,
        auto="light",  # Use "medium" for more thorough search
        num_threads=4,
    )

    optimized = optimizer.compile(
        proposer,
        trainset=trainset,
        valset=valset,
    )

    # Save compiled module
    optimized.save(OUTPUT_PATH)
    print(f"\nSaved optimized module to: {OUTPUT_PATH}")

    # Extract and display the optimized prompt
    print("\n" + "=" * 60)
    print("OPTIMIZED PROMPT")
    print("=" * 60)

    # The optimized prompt is stored in the module's predictor
    if hasattr(optimized.propose, "signature"):
        sig = optimized.propose.signature
        print(f"\nSignature docstring:\n{sig.__doc__}")

    # Show demos if any were generated
    if hasattr(optimized.propose, "demos") and optimized.propose.demos:
        print(f"\nGenerated {len(optimized.propose.demos)} demos")
        for i, demo in enumerate(optimized.propose.demos[:2]):  # Show first 2
            print(f"\n--- Demo {i + 1} ---")
            print(f"Q: {demo.question[:100]}...")

    print("\n" + "=" * 60)
    print("To use this optimized module:")
    print("  from dspy import load")
    print(f'  proposer = load("{OUTPUT_PATH}")')
    print("=" * 60)


if __name__ == "__main__":
    main()
