# Data Workflow

The pipeline is **input-agnostic** and operates on preprocessed text chunks. Raw data sources are converted to a standardized format before running the causal inference pipeline.

## Directory Structure

```
data/
├── raw/           # Raw input data (gitignored)
├── processed/     # Converted text files (gitignored)
├── queries/       # Test queries for pipeline
└── eval/          # Eval data (questions/{N}_{name}/ with latent_model.json, causal_spec.json)
```

## Preprocessing

1. Place raw data exports in `data/raw/`
2. Run the preprocessing script to convert to text chunks:

```bash
# Process all zips in data/raw/
uv run python evals/scripts/preprocess_google_takeout.py

# Process a specific file
uv run python evals/scripts/preprocess_google_takeout.py -i data/raw/export.zip
```

This outputs `data/processed/google_activity_<timestamp>.txt` with one text chunk per line.

## Manual Testing

Sample contiguous data chunks for testing graph construction with external LLMs:

```bash
uv run python evals/scripts/sample_data_chunks.py -n 20

# Include system prompt for generating training examples
uv run python evals/scripts/sample_data_chunks.py --prompt
```

Output goes to `data/processed/orchestrator-samples-manual.txt`.
