# Data Workflow

The repo currently has two distinct data lanes:

1. **Workspace runs** for the web app and Prefect pipeline.
2. **Preprocessed chunk files** for evals and manual prompt testing.

## Directory Structure

```
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Raw uploaded files for stage 0
│   ├── query.txt          # Materialized research question
│   ├── session.json       # Per-workspace run lineage metadata
│   └── run/               # Persisted stage JSON + artifacts
├── DEFAULT/               # Tracked mock fixture workspace
├── DOCTOLIB/              # Tracked mock fixture workspace
├── MEDICAL_SEMANTICS/     # Tracked medical archive fixture for stage 0-2 golden tests
├── GOLDEN/                # Golden input dataset submodule
└── processed/             # Preprocessed text chunks for eval/manual tools (gitignored)
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

## Preprocessed Chunk Workflow

Some eval and manual utilities still consume newline-delimited text chunk files
from `data/processed/`. If you already have a preprocessed text file there, you
can sample representative chunks for prompt testing with:

```bash
cd apps/data-pipeline

uv run python evals/scripts/sample_data_chunks.py -n 20

# Sample from a specific preprocessed file
uv run python evals/scripts/sample_data_chunks.py -i google_activity_20251208.txt -n 5

# Include the orchestrator system prompt for copy/paste experiments
uv run python evals/scripts/sample_data_chunks.py --prompt
```

Output goes to `data/processed/orchestrator-samples-manual.txt`. The `data/processed/` directory is created at runtime (gitignored).
