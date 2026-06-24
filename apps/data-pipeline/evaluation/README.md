# evaluation/

Capability benchmarks for the causal-SSM pipeline. Top-level (peer of `src/`)
because it *consumes* `nof1_causal_lab` and nothing in the pipeline imports it
back. Imported as `from evaluation... import ...` (repo root on `sys.path`).

## Layout

- **spine** — `contracts` · `registry` · `seeds` · `scenarios/` · `scorers/` ·
  `fixtures/` · `recovery/`: the importable library. A `Scenario` carries inputs
  + truth, a `StageRunner` drives the live core, a `StageScorer` grades. Cells
  are tagged `stage × mode × cost × cadence × kind × capability`.
- **surfaces** — thin entrypoints: `inspect_evals/` (LLM evals via Inspect),
  `benchmarks/` (CLI/Modal recovery).
- **data** — `data/questions/` (static fixtures), `data/cache/` (fetched
  datasets, gitignored), `data/datasets.py` (downloader).

## Invariants (don't break these)

1. **Scoring is single-source.** Grading logic lives only in `scorers/`;
   surfaces import it, never re-implement. Re-implementing is what silently
   broke the old `orchestrator`-based evals.
2. **Two frameworks, one scoring.** Statistical/deterministic benchmarks are
   **registry rows** (run via `evaluate()`); LLM evals are **Inspect tasks** —
   separate runners, but both call the same `scorers/`.
3. **Code vs data.** Generators are code → `fixtures/`. Static fixtures +
   fetched datasets → `data/` (cache gitignored). Generated outputs (CSVs) are
   gitignored, never committed to the tree.

## Adding one

- **Statistical benchmark** → a `Scenario` + `StageScorer` (+ `StageRunner` if
  it drives a live core), registered in `seeds.py`.
- **LLM eval** → an Inspect `@task` in `inspect_evals/` that imports a `scorers/`
  function — do not write a new scorer there.
- **External dataset** → `cached_download(url, name)` into `data/cache/`.
