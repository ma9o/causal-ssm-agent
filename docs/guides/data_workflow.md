# Data Workflow

This page covers workspace layout and file placement. For the practitioner-facing dataset contract, including timestamps, missingness, indicator modes, and minimum viable dataset shape, see [data_contract.md](data_contract.md).

## Directory Structure

```text
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Raw uploaded files for stage 0
│   ├── query.txt          # Materialized research question
│   ├── session.json       # Per-workspace run lineage metadata
│   └── run/               # Persisted stage JSON + artifacts
├── DEFAULT/               # Tracked mock fixture workspace
├── DOCTOLIB/              # Tracked mock fixture workspace
├── GOLDEN/                # Default tracked workspace for evals and manual sampling
├── MEDICAL_SEMANTICS/     # Tracked medical archive fixture for stage 0-2 golden tests
└── SMALLGOLDEN/           # Smaller tracked workspace for quicker eval iteration
```

## Workspace Runs

For app or pipeline runs, place raw exports directly into a workspace:

```bash
WORKSPACE_ID="T3ST42"
mkdir -p data/$WORKSPACE_ID/input
cp /path/to/export.json data/$WORKSPACE_ID/input/
```

Stage 0 scans `data/{workspace_id}/input/` and ingests the most recent non-hidden file.
The question is stored in `data/{workspace_id}/query.txt`, run lineage is stored in
`data/{workspace_id}/session.json`, and stage outputs land in `data/{workspace_id}/run/`.

## Tracked Fixture Workspaces

Evals and manual prompt-sampling tools read the same persisted workspace artifacts
that the pipeline uses. By default they load `data/GOLDEN/`, but you can point
them at any workspace with compatible `query.txt` and `run/stage-*.json` outputs.

This keeps evaluation inputs aligned with the main app and pipeline contracts
instead of maintaining a separate preprocessed-text lane.

## Manual Worker Prompt Sampling

To inspect representative Stage 2 semantic worker chunks from a workspace:

```bash
cd apps/data-pipeline

uv run python evals/scripts/sample_data_chunks.py -n 20
uv run python evals/scripts/sample_data_chunks.py --workspace-id SMALLGOLDEN -n 5

# Include the exact worker system + user prompts for copy/paste experiments
uv run python evals/scripts/sample_data_chunks.py --prompt
```

Output goes to `scratchpad/worker-chunks-manual.txt`.
