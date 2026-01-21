"""Main causal inference pipeline.

Orchestrates all stages from structure proposal to intervention analysis.

Two-stage specification following Anderson & Gerbing (1988):
- Stage 1a: Structural model (theory-driven, no data)
- Stage 1b: Measurement model (data-driven operationalization)
"""

from prefect import flow
from prefect.utilities.annotations import unmapped

from causal_agent.utils.data import (
    SAMPLE_CHUNKS,
    load_query,
    resolve_input_path,
)

from .stages import (
    # Stage 1a
    propose_structural_model,
    # Stage 1b
    build_dsem_model,
    load_orchestrator_chunks,
    propose_measurement_model,
    # Stage 2
    aggregate_measurements,
    load_worker_chunks,
    populate_indicators,
    # Stage 3
    check_identifiability,
    # Stage 4
    elicit_priors,
    specify_model,
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

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1a: Propose structural model (theory only, no data)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 1a: Structural Model ===")
    structural_model = propose_structural_model(question)
    n_constructs = len(structural_model["constructs"])
    n_edges = len(structural_model["edges"])
    print(f"Proposed {n_constructs} constructs with {n_edges} causal edges")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1b: Propose measurement model (with data)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 1b: Measurement Model ===")
    orchestrator_chunks = load_orchestrator_chunks(input_path)
    print(f"Loaded {len(orchestrator_chunks)} orchestrator chunks")

    measurement_model = propose_measurement_model(
        question,
        structural_model,
        orchestrator_chunks[:SAMPLE_CHUNKS],
    )
    n_indicators = len(measurement_model["indicators"])
    print(f"Proposed {n_indicators} indicators")

    # Combine into full DSEM model
    dsem_model = build_dsem_model(structural_model, measurement_model)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: Parallel indicator population (worker chunk size)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 2: Worker Extraction ===")
    worker_chunks = load_worker_chunks(input_path)
    print(f"Loaded {len(worker_chunks)} worker chunks")

    worker_results = populate_indicators.map(
        worker_chunks,
        question=unmapped(question),
        dsem_model=unmapped(dsem_model),
    )

    # Stage 2b: Aggregate measurements into time-series by causal_granularity
    measurements = aggregate_measurements(worker_results, dsem_model)
    for granularity, df in measurements.items():
        n_indicators = len([c for c in df.columns if c != "time_bucket"])
        if granularity == "time_invariant":
            print(f"  {granularity}: {n_indicators} indicators")
        else:
            print(f"  {granularity}: {df.height} time points × {n_indicators} indicators")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3: Identifiability
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 3: Identifiability ===")
    # TODO: Update to work with new DSEMModel - use structural.edges
    identifiable = check_identifiability(dsem_model["structural"], target_effects)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: Model specification
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 4: Model Specification ===")
    # TODO: Update to work with new DSEMModel
    model_spec = specify_model(dsem_model["structural"], dsem_model)
    priors = elicit_priors(model_spec)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5: Fit and intervene
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 5: Inference ===")
    fitted = fit_model(model_spec, priors, worker_chunks)
    results = run_interventions(fitted, target_effects)

    return results


if __name__ == "__main__":
    # Serve the flow for UI access
    causal_inference_pipeline.serve(
        name="causal-inference",
        tags=["causal", "llm"],
    )
