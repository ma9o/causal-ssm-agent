# Data Workflow

The repo currently has two distinct data lanes:

1. **Session workspaces** for the web app and Prefect pipeline.
2. **Preprocessed chunk files** for evals and manual prompt testing.

## Directory Structure

```
data/
├── <CODE>/                # Session workspace
│   ├── input/             # Raw uploaded files for stage 0
│   ├── query.txt          # Materialized research question
│   └── run/               # Persisted stage JSON + artifacts
├── DEFAULT/               # Tracked mock session fixture
├── DOCTOLIB/              # Tracked mock session fixture
├── GOLDEN/                # Golden input dataset submodule
├── processed/             # Preprocessed text chunks for eval/manual tools (gitignored)
├── sessions.seed.json     # Tracked fixture session metadata
└── sessions.json          # Runtime session metadata (gitignored)
```

## Session Runs

For app or pipeline runs, place raw exports directly into a session workspace:

```bash
CODE="T3ST42"
mkdir -p data/$CODE/input
cp /path/to/export.json data/$CODE/input/
```

Stage 0 scans `data/{code}/input/` and ingests the most recent non-hidden file.
The question is stored in `data/{code}/query.txt`, and stage outputs land in
`data/{code}/run/`.

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
