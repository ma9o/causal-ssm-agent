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
    stage4_orchestrated_flow,
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
    measurements_task = aggregate_measurements(worker_results, dsem_model)
    measurements_data = measurements_task.result() if hasattr(measurements_task, "result") else measurements_task

    for granularity, df in measurements_data.items():
        n_indicators = len([c for c in df.columns if c != "time_bucket"])
        if granularity == "time_invariant":
            print(f"  {granularity}: {n_indicators} indicators")
        else:
            print(f"  {granularity}: {df.height} time points × {n_indicators} indicators")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3: Validate Extraction
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 3: Extraction Validation ===")
    validation_task = validate_extraction(dsem_model, measurements_data)
    validation_report = validation_task.result() if hasattr(validation_task, "result") else validation_task

    if validation_report:
        issues = validation_report.get("issues", [])
        if not validation_report.get("is_valid", True):
            print("⚠️  Stage 3 validation errors detected:")
            for issue in issues:
                print(f"    - {issue['indicator']}: {issue['issue_type']} ({issue['severity']}) {issue['message']}")
        elif issues:
            print("⚠️  Stage 3 validation warnings:")
            for issue in issues:
                print(f"    - {issue['indicator']}: {issue['issue_type']} ({issue['severity']}) {issue['message']}")

    constructs = dsem_model.get("latent", {}).get("constructs", [])
    construct_granularity = {c.get("name"): c.get("causal_granularity") for c in constructs}
    missing_indicators: list[str] = []

    for indicator in dsem_model.get("measurement", {}).get("indicators", []):
        ind_name = indicator.get("name")
        if not ind_name:
            continue

        construct_name = indicator.get("construct") or indicator.get("construct_name")
        granularity = construct_granularity.get(construct_name)
        gran_key = granularity if granularity else "time_invariant"

        df = measurements_data.get(gran_key)
        if df is None or ind_name not in df.columns:
            missing_indicators.append(ind_name)

    if missing_indicators:
        joined = ", ".join(sorted(set(missing_indicators)))
        print(f"⚠️  Aggregation produced no data for indicators: {joined}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: Model Specification (Orchestrator-Worker Architecture)
    # ══════════════════════════════════════════════════════════════════════════
    from dsem_agent.utils.config import get_config

    config = get_config()

    print("\n=== Stage 4: Model Specification ===")
    stage4_result = stage4_orchestrated_flow(
        dsem_model=dsem_model,
        question=question,
        measurements_data=measurements_data,
        enable_literature=config.stage4_prior_elicitation.literature_search.enabled,
    )

    model_spec = stage4_result.get("model_spec", {})
    print(f"Model clock: {model_spec.get('model_clock', 'unknown')}")
    print(f"Parameters: {len(model_spec.get('parameters', []))} total")

    # Report validation issues
    validation = stage4_result.get("validation", {})
    if not validation.get("is_valid", True):
        issues = validation.get("issues", [])
        print(f"⚠️  Stage 4 prior validation failed ({len(issues)} issues):")
        for issue in issues:
            print(f"    - {issue.get('parameter')}: {issue.get('issue')}")

    model_info = stage4_result.get("model_info", {})
    if not model_info.get("model_built", True):
        print(f"⚠️  Stage 4 model build failed: {model_info.get('error')}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5: Fit and intervene (with identifiability awareness)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 5: Inference ===")
    print(f"Estimating effects of {len(treatments)} treatments on {outcome}")
    fitted = fit_model(stage4_result, worker_chunks)

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
