"""Main causal inference pipeline.

Orchestrates all stages from structure proposal to intervention analysis.
"""

from prefect import flow

from causal_agent.utils.data import (
    resolve_input_path,
    load_query,
    SAMPLE_CHUNKS,
)
from .stages import (
    # Stage 1
    load_orchestrator_chunks,
    propose_structure,
    # Stage 2
    load_worker_chunks,
    populate_dimensions,
    merge_suggestions,
    # Stage 3
    check_identifiability,
    # Stage 4
    specify_model,
    elicit_priors,
    # Stage 5
    fit_model,
    run_interventions,
)


@flow(log_prints=True)
def causal_inference_pipeline(
    query_file: str,
    target_effects: list[str],
    input_file: str | None = None,
):
    """
    Main causal inference pipeline.

    Args:
        query_file: Filename in data/queries/ (e.g., 'smoking-cancer')
        target_effects: Causal effects to estimate
        input_file: Filename in data/processed/ (default: latest file)
    """
    # Stage 0: Load question and resolve input path
    question = load_query(query_file)
    print(f"Query: {query_file}")
    print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")

    input_path = resolve_input_path(input_file)
    print(f"Using input file: {input_path.name}")

    # Stage 1: Propose structure from sample (orchestrator chunk size)
    orchestrator_chunks = load_orchestrator_chunks(input_path)
    print(f"Loaded {len(orchestrator_chunks)} orchestrator chunks")
    schema = propose_structure(question, orchestrator_chunks[:SAMPLE_CHUNKS])

    # Stage 2: Parallel dimension population (worker chunk size)
    worker_chunks = load_worker_chunks(input_path)
    print(f"Loaded {len(worker_chunks)} worker chunks")
    worker_results = populate_dimensions.map(
        worker_chunks,
        question=question,
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
    fitted = fit_model(model_spec, priors, worker_chunks)
    results = run_interventions(fitted, target_effects)

    return results


if __name__ == "__main__":
    # Serve the flow for UI access
    causal_inference_pipeline.serve(
        name="causal-inference",
        tags=["causal", "llm"],
    )
