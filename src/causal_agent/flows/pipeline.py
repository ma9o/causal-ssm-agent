from pathlib import Path

from prefect import flow, task
from prefect.cache_policies import INPUTS
from typing import Any

from causal_agent.utils.data import (
    load_text_chunks as load_text_chunks_util,
    resolve_input_path,
    load_query,
)


@task(cache_policy=INPUTS)
def load_text_chunks(input_path: Path) -> list[str]:
    """Stage 0: Load preprocessed text chunks from file."""
    return load_text_chunks_util(input_path)


@task(retries=2, retry_delay_seconds=30, cache_policy=INPUTS)
def propose_structure(question: str, data_sample: list[str]) -> dict:
    """Stage 1: Orchestrator proposes dimensions, autocorrelations, time granularities, DAG."""
    from causal_agent.orchestrator.agents import propose_structure as propose_structure_agent

    return propose_structure_agent(question, data_sample)


@task(
    retries=2,
    retry_delay_seconds=10,
    task_run_name="populate-chunk-{chunk_id}",
)
def populate_dimensions(chunk: str, chunk_id: int, schema: dict) -> dict:
    """Stage 2: Workers populate candidate dimensions for each chunk."""
    pass


@task(retries=1, cache_policy=INPUTS)
def merge_suggestions(base_schema: dict, worker_suggestions: list[dict]) -> dict:
    """Stage 2b: Orchestrator performs 3-way merge and backfill."""
    pass


@task(cache_policy=INPUTS)
def check_identifiability(dag: dict, target_effects: list[str]) -> bool:
    """Stage 3: Run DoWhy identifiability checks."""
    pass


@task
def run_sensitivity_analysis(dag: dict, naive_model: Any) -> dict:
    """Stage 3b: Cinelli-Hazlett sensitivity analysis if unidentifiable."""
    pass


@task(retries=1, cache_policy=INPUTS)
def specify_model(dag: dict, schema: dict) -> dict:
    """Stage 4: Orchestrator specifies GLM in PyMC."""
    pass


@task(retries=2, retry_delay_seconds=10, task_run_name="elicit-priors")
def elicit_priors(model_spec: dict) -> dict:
    """Stage 4b: Workers provide priors."""
    pass


@task(timeout_seconds=3600, retries=1)
def fit_model(model_spec: dict, priors: dict, data: list[str]) -> Any:
    """Stage 5: Fit PyMC model."""
    pass


@task
def run_interventions(fitted_model: Any, interventions: list[str]) -> list[dict]:
    """Stage 5b: Run interventions and counterfactuals, rank by effect size."""
    pass


@flow(log_prints=True)
def causal_inference_pipeline(
    query_file: str,
    target_effects: list[str],
    input_file: str | None = None,
):
    """
    Main causal inference pipeline.

    Args:
        query_file: Filename in data/test-queries/ (e.g., 'smoking-cancer')
        target_effects: Causal effects to estimate
        input_file: Filename in data/preprocessed/ (default: latest file)
    """
    # Stage 0: Load question and data chunks
    question = load_query(query_file)
    print(f"Query: {query_file}")
    print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")

    input_path = resolve_input_path(input_file)
    print(f"Using input file: {input_path.name}")

    chunks = load_text_chunks(input_path)
    print(f"Loaded {len(chunks)} chunks")

    # Stage 1: Propose structure from sample
    schema = propose_structure(question, chunks[:3])

    # Stage 2: Parallel dimension population
    worker_results = populate_dimensions.map(
        chunks,
        chunk_id=list(range(len(chunks))),
        schema=schema,
    )
    schema = merge_suggestions(schema, worker_results)

    # Stage 3: Identifiability
    identifiable = check_identifiability(schema["dag"], target_effects)
    # TODO: conditional logic for sensitivity analysis

    # Stage 4: Model specification
    model_spec = specify_model(schema["dag"], schema)
    priors = elicit_priors(model_spec)

    # Stage 5: Fit and intervene
    fitted = fit_model(model_spec, priors, chunks)
    results = run_interventions(fitted, target_effects)

    return results


if __name__ == "__main__":
    # Serve the flow for UI access
    causal_inference_pipeline.serve(
        name="causal-inference",
        tags=["causal", "llm"],
    )
