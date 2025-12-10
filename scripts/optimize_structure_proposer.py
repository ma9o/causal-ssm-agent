#!/usr/bin/env python3
"""Optimize the StructureProposer DSPy module using MIPROv2.

Usage:
    uv run python scripts/optimize_structure_proposer.py --trainset data/trainset.json

The trainset JSON should contain a list of examples:
[
    {
        "question": "How does stress affect my sleep?",
        "data_sample": "...",
        "structure": "{\"dimensions\": [...], \"edges\": [...]}"
    },
    ...
]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import dspy

from causal_agent.orchestrator.dspy_module import (
    OPTIMIZED_PROGRAM_PATH,
    StructureProposer,
)
from causal_agent.orchestrator.scoring import score_structure_proposal


def load_trainset(path: Path) -> list[dspy.Example]:
    """Load training examples from JSON file."""
    with open(path) as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = dspy.Example(
            question=item["question"],
            data_sample=item["data_sample"],
            structure=item["structure"],
        ).with_inputs("question", "data_sample")
        examples.append(example)

    return examples


def split_trainset(
    examples: list[dspy.Example], train_ratio: float = 0.2
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Split examples into train (20%) and validation (80%) sets."""
    split_idx = max(1, int(len(examples) * train_ratio))
    return examples[:split_idx], examples[split_idx:]


def optimize(
    trainset_path: Path,
    output_path: Path | None = None,
    model: str = "openrouter/google/gemini-2.5-pro-preview-06-05",
    auto: str = "medium",
    num_threads: int = 4,
) -> Path:
    """Run MIPROv2 optimization on StructureProposer.

    Args:
        trainset_path: Path to JSON file with training examples.
        output_path: Where to save optimized program. Defaults to OPTIMIZED_PROGRAM_PATH.
        model: LLM model to use.
        auto: MIPROv2 auto mode (light/medium/heavy).
        num_threads: Number of parallel threads.

    Returns:
        Path to saved optimized program.
    """
    if output_path is None:
        output_path = OPTIMIZED_PROGRAM_PATH

    # Configure DSPy
    lm = dspy.LM(model)
    dspy.configure(lm=lm)

    # Load and split data
    all_examples = load_trainset(trainset_path)
    trainset, valset = split_trainset(all_examples)

    print(f"Loaded {len(all_examples)} examples")
    print(f"  Train: {len(trainset)}")
    print(f"  Validation: {len(valset)}")

    # Create optimizer
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=score_structure_proposal,
        prompt_model=model,
        task_model=model,
        auto=auto,
        num_threads=num_threads,
    )

    # Optimize
    program = StructureProposer()
    print(f"\nStarting optimization with MIPROv2 (auto={auto})...")

    optimized = optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
    )

    # Save optimized program
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(output_path))
    print(f"\nSaved optimized program to: {output_path}")

    # Save metadata
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": model,
        "optimizer": "MIPROv2",
        "auto_mode": auto,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
        "created_at": datetime.now().isoformat(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Optimize StructureProposer with DSPy")
    parser.add_argument(
        "--trainset",
        type=Path,
        required=True,
        help="Path to JSON file with training examples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output path for optimized program (default: {OPTIMIZED_PROGRAM_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/google/gemini-2.5-pro-preview-06-05",
        help="LLM model to use",
    )
    parser.add_argument(
        "--auto",
        type=str,
        choices=["light", "medium", "heavy"],
        default="medium",
        help="MIPROv2 optimization intensity",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel threads",
    )

    args = parser.parse_args()

    optimize(
        trainset_path=args.trainset,
        output_path=args.output,
        model=args.model,
        auto=args.auto,
        num_threads=args.threads,
    )


if __name__ == "__main__":
    main()
