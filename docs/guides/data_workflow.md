# Data Workflow

The repo currently has two distinct data lanes:

1. **User workspaces** for the web app and Prefect pipeline.
2. **Preprocessed chunk files** for evals and manual prompt testing.

## Directory Structure

```
data/
├── <USER_ID>/             # User workspace
│   ├── input/             # Raw uploaded files for stage 0
│   ├── query.txt          # Materialized research question
│   └── run/               # Persisted stage JSON + artifacts
├── DEFAULT/               # Tracked mock user fixture
├── DOCTOLIB/              # Tracked mock user fixture
├── GOLDEN/                # Golden input dataset submodule
├── processed/             # Preprocessed text chunks for eval/manual tools (gitignored)
├── sessions.seed.json     # Tracked fixture run metadata keyed by user ID
└── sessions.json          # Runtime run metadata keyed by user ID (gitignored)
```

## User Runs

For app or pipeline runs, place raw exports directly into a user workspace:

```bash
USER_ID="T3ST42"
mkdir -p data/$USER_ID/input
cp /path/to/export.json data/$USER_ID/input/
```

Stage 0 scans `data/{user_id}/input/` and ingests the most recent non-hidden file.
The question is stored in `data/{user_id}/query.txt`, and stage outputs land in
`data/{user_id}/run/`.

## Preprocessed Chunk Workflow

Some eval and manual utilities still consume newline-delimited text chunk files
from `data/processed/`. If you already have a preprocessed text file there, you
can sample representative chunks for prompt testing with:

```bash
uv run python evals/scripts/sample_data_chunks.py -n 20

# Sample from a specific preprocessed file
uv run python evals/scripts/sample_data_chunks.py -i google_activity_20251208.txt -n 5

# Include the orchestrator system prompt for copy/paste experiments
uv run python evals/scripts/sample_data_chunks.py --prompt
```

Output goes to `data/processed/orchestrator-samples-manual.txt`.
