#!/usr/bin/env python
"""Sample real Stage 2 worker chunks from a workspace for manual testing.

Usage:
    uv run python evals/scripts/sample_data_chunks.py
    uv run python evals/scripts/sample_data_chunks.py -n 3
    uv run python evals/scripts/sample_data_chunks.py --workspace-id SMALLGOLDEN
    uv run python evals/scripts/sample_data_chunks.py --prompt
"""

import argparse
import sys
from pathlib import Path

from evaluation.inspect_evals.common import DEFAULT_EVAL_WORKSPACE_ID, get_stage2_eval_chunks

from nof1_causal_lab.workers.core import _format_indicators, _get_outcome_description
from nof1_causal_lab.workers.prompts.extraction import SYSTEM, USER

OUTPUT_FILE = Path(__file__).resolve().parents[4] / "scratchpad" / "worker-chunks-manual.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Stage 2 worker chunks for manual testing")
    parser.add_argument("-n", type=int, default=5, help="Number of chunks to sample")
    parser.add_argument(
        "--workspace-id",
        type=str,
        default=DEFAULT_EVAL_WORKSPACE_ID,
        help=f"Workspace to sample from (default: {DEFAULT_EVAL_WORKSPACE_ID})",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Include the exact worker system + user prompt for each sampled chunk",
    )
    args = parser.parse_args()

    stage2_inputs = get_stage2_eval_chunks(args.n, args.seed or 42, args.workspace_id)
    causal_spec = stage2_inputs["causal_spec"]
    chunks = stage2_inputs["sampled_chunk_texts"]

    print(f"Sampling from workspace: {stage2_inputs['workspace_id']}", file=sys.stderr)
    print(f"Question: {stage2_inputs['question']}", file=sys.stderr)
    print(f"Chunks: {len(chunks)}", file=sys.stderr)

    output_parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        if args.prompt:
            user_prompt = USER.format(
                question=stage2_inputs["question"],
                outcome_description=_get_outcome_description(causal_spec),
                indicators=_format_indicators(causal_spec),
                chunk=chunk,
            )
            output_parts.append(f"--- CHUNK {i} SYSTEM ---\n{SYSTEM}\n")
            output_parts.append(f"--- CHUNK {i} USER ---\n{user_prompt}\n")
        else:
            output_parts.append(f"--- CHUNK {i} ---\n{chunk}\n")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(output_parts))
    print(f"Wrote to: {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
