"""Main causal inference pipeline.

Orchestrates all stages from structure proposal to intervention analysis.

Two-stage specification following Anderson & Gerbing (1988):
- Stage 1a: Latent model (theory-driven, no data)
- Stage 1b: Measurement model (data-driven operationalization)
"""

from prefect import flow
from prefect.utilities.annotations import unmapped

from dsem_agent.utils.data import (
    SAMPLE_CHUNKS,
    load_query,
    resolve_input_path,
)
from dsem_agent.utils.effects import (
    get_all_treatments,
    get_outcome_from_latent_model,
)

from .stages import (
    # Stage 1a
    propose_latent_model,
    # Stage 1b
    build_dsem_model,
    load_orchestrator_chunks,
    propose_measurement_with_identifiability_fix,
    # Stage 2
    aggregate_measurements,
    load_worker_chunks,
    populate_indicators,
    # Stage 3
    validate_extraction,
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
    input_file: str | None = None,
):
    """
    Main causal inference pipeline.

    Automatically identifies the outcome from the question and estimates
    effects of all potential treatments, ranking them by effect size.

    Args:
        query_file: Filename in data/queries/ (e.g., 'resolve-errors')
        input_file: Filename in data/processed/ (default: latest file)
    """
    # Stage 0: Load question and resolve input path
    question = load_query(query_file)
    print(f"Query: {query_file}")
    print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")

    input_path = resolve_input_path(input_file)
    print(f"Using input file: {input_path.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1a: Propose latent model (theory only, no data)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 1a: Latent Model ===")
    latent_model = propose_latent_model(question)
    n_constructs = len(latent_model["constructs"])
    n_edges = len(latent_model["edges"])
    print(f"Proposed {n_constructs} constructs with {n_edges} causal edges")

    # Identify the outcome and all potential treatments
    outcome = get_outcome_from_latent_model(latent_model)
    if not outcome:
        raise ValueError("No outcome identified in latent model (missing is_outcome=true)")
    print(f"Outcome variable: {outcome}")

    treatments = get_all_treatments(latent_model)
    print(f"Potential treatments: {len(treatments)} constructs with paths to {outcome}")
    for t in treatments[:5]:
        print(f"  - {t}")
    if len(treatments) > 5:
        print(f"  ... and {len(treatments) - 5} more")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1b: Propose measurement model (with identifiability check)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 1b: Measurement Model with Identifiability ===")
    orchestrator_chunks = load_orchestrator_chunks(input_path)
    print(f"Loaded {len(orchestrator_chunks)} orchestrator chunks")

    # Propose measurements and check identifiability
    measurement_result = propose_measurement_with_identifiability_fix(
        question,
        latent_model,
        orchestrator_chunks[:SAMPLE_CHUNKS],
    )

    measurement_model = measurement_result['measurement_model']
    identifiability_status = measurement_result['identifiability_status']

    n_indicators = len(measurement_model["indicators"])
    print(f"Final model has {n_indicators} indicators")

    # Report non-identifiable treatments
    non_identifiable = identifiability_status.get('non_identifiable_treatments', {})
    if non_identifiable:
        print("\n⚠️  NON-IDENTIFIABLE TREATMENT EFFECTS:")
        for treatment in sorted(non_identifiable.keys()):
            details = non_identifiable[treatment]
            blockers = details.get('confounders', []) if isinstance(details, dict) else []
            notes = details.get('notes') if isinstance(details, dict) else None
            if blockers:
                print(f"  - {treatment} → {outcome} (blocked by: {', '.join(blockers)})")
            elif notes:
                print(f"  - {treatment} → {outcome} ({notes})")
            else:
                print(f"  - {treatment} → {outcome}")
        print("These effects will be flagged in the final ranking.")

    # Combine into full DSEM model with identifiability status
    dsem_model = build_dsem_model(latent_model, measurement_model, identifiability_status)

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
    # Stage 3: Validate Extraction
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 3: Extraction Validation ===")
    validation_report = validate_extraction(dsem_model, measurements)
    # TODO: Handle validation failures - revalidate DAG if proxies missing

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: Model specification
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 4: Model Specification ===")
    model_spec = specify_model(dsem_model["latent"], dsem_model)
    priors = elicit_priors(model_spec)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5: Fit and intervene (with identifiability awareness)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 5: Inference ===")
    print(f"Estimating effects of {len(treatments)} treatments on {outcome}")
    fitted = fit_model(model_spec, priors, worker_chunks)

    # Run interventions for all treatments
    results = run_interventions(fitted, treatments, dsem_model)

    # TODO: Rank by effect size
    print(f"\n=== Treatment Ranking by Effect Size ===")
    print("(To be implemented: ranking of all treatments by their effect on the outcome)")

    return results


if __name__ == "__main__":
    # Serve the flow for UI access
    causal_inference_pipeline.serve(
        name="causal-inference",
        tags=["causal", "llm"],
    )
