#!/usr/bin/env python
"""Sample data chunks for manual LLM testing.

Usage:
    uv run python scripts/sample_data_chunks.py
    uv run python scripts/sample_data_chunks.py -n 3
    uv run python scripts/sample_data_chunks.py -i google_activity_20251208.txt -n 5
    uv run python scripts/sample_data_chunks.py --prompt  # Include system prompt for training data generation
"""

import argparse

from causal_ssm_agent.orchestrator.prompts import latent_model
from causal_ssm_agent.utils.data import (
    CHUNK_SIZE,
    PROCESSED_DIR,
    SAMPLE_CHUNKS,
    get_latest_preprocessed_file,
    sample_chunks,
)

OUTPUT_FILE = PROCESSED_DIR / "orchestrator-samples-manual.txt"
EXCLUDE_FILES = {OUTPUT_FILE.name}


def main():
    parser = argparse.ArgumentParser(description="Sample data chunks for manual testing")
    parser.add_argument("-n", type=int, default=SAMPLE_CHUNKS, help="Number of chunks to sample")
    parser.add_argument("-i", "--input", type=str, help="Input file name (in data/processed/)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--prompt", action="store_true", help="Include system prompt for training data generation")
    args = parser.parse_args()

    # Get input file
    if args.input:
        input_file = PROCESSED_DIR / args.input
    else:
        input_file = get_latest_preprocessed_file(exclude=EXCLUDE_FILES)
        if not input_file:
            raise FileNotFoundError(f"No data files found in {PROCESSED_DIR}")

    print(f"Sampling from: {input_file.name}", file=__import__("sys").stderr)
    print(f"Chunk size: {CHUNK_SIZE} lines per chunk", file=__import__("sys").stderr)

    # Sample chunks
    chunks = sample_chunks(input_file, args.n, args.seed)

    # Build output
    output_parts = []

    if args.prompt:
        output_parts.append(latent_model.SYSTEM)
        output_parts.append("\n---\n")
        output_parts.append("Question: <YOUR QUESTION HERE>\n")
        output_parts.append("\nSample data:\n")

    for i, chunk in enumerate(chunks):
        output_parts.append(f"--- CHUNK {i + 1} ---\n")
        output_parts.append(chunk + "\n\n")

    output = "".join(output_parts)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output)

    print(f"Wrote to: {OUTPUT_FILE}", file=__import__("sys").stderr)
    if args.prompt:
        print("(Includes system prompt - ready for LLM paste)", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
