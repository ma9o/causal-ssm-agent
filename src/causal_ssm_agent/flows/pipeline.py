"""Main causal inference pipeline.

Orchestrates all stages from structure proposal to intervention analysis.

Two-stage specification following Anderson & Gerbing (1988):
- Stage 1a: Latent model (theory-driven, no data)
- Stage 1b: Measurement model (data-driven operationalization)
"""

from pathlib import Path

from prefect import flow
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.utilities.annotations import unmapped

from causal_ssm_agent.utils.aggregations import flatten_aggregated_data
from causal_ssm_agent.utils.data import get_sample_chunks, load_query
from causal_ssm_agent.utils.effects import (
    get_all_treatments,
    get_outcome_from_latent_model,
)

from .stages import (
    # Stage 3
    aggregate_measurements,
    # Stage 1b
    build_causal_spec,
    combine_worker_results,
    # Stage 5
    fit_model,
    load_orchestrator_chunks,
    # Stage 2
    load_worker_chunks,
    populate_indicators,
    # Stage 0
    preprocess_raw_input,
    # Stage 1a
    propose_latent_model,
    propose_measurement_with_identifiability_fix,
    run_interventions,
    run_power_scaling,
    run_ppc,
    # Stage 4
    stage4_orchestrated_flow,
    # Stage 4b
    stage4b_parametric_id_flow,
    validate_extraction,
)

RESULT_STORAGE = Path("results")


@flow(
    log_prints=True,
    persist_result=True,
    result_storage=RESULT_STORAGE,
    result_serializer="pickle",
)
async def causal_inference_pipeline(
    query_file: str,
    user_id: str = "test_user",
    inference_method: str | None = None,
    enable_literature: bool | None = None,
):
    """
    Main causal inference pipeline.

    Automatically identifies the outcome from the question and estimates
    effects of all potential treatments, ranking them by effect size.

    Args:
        query_file: Filename in data/queries/ (e.g., 'procrastination-patterns')
        user_id: User subdirectory under data/raw/ (default: test_user)
        inference_method: Override inference method ("svi" or "nuts", default from config)
        enable_literature: Override literature search (default from config)
    """
    # ══════════════════════════════════════════════════════════════════════════
    # Stage 0: Preprocess raw input and load question
    # ══════════════════════════════════════════════════════════════════════════
    question = load_query(query_file)
    print(f"Query: {query_file}")
    print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")

    print(f"\n=== Stage 0: Preprocess (user: {user_id}) ===")
    lines = preprocess_raw_input(user_id)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1a: Propose latent model (theory only, no data)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 1a: Latent Model ===")
    latent_model = await propose_latent_model(question)
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
    orchestrator_chunks = load_orchestrator_chunks(lines)
    print(f"Loaded {len(orchestrator_chunks)} orchestrator chunks")

    # Propose measurements and check identifiability
    measurement_result = await propose_measurement_with_identifiability_fix(
        question,
        latent_model,
        orchestrator_chunks[: get_sample_chunks()],
    )

    measurement_model = measurement_result["measurement_model"]
    identifiability_status = measurement_result["identifiability_status"]

    n_indicators = len(measurement_model["indicators"])
    print(f"Final model has {n_indicators} indicators")

    # Hard gate: filter out non-identifiable treatments
    non_identifiable = identifiability_status.get("non_identifiable_treatments", {})
    if non_identifiable:
        print("\n⚠️  NON-IDENTIFIABLE TREATMENT EFFECTS (excluded from analysis):")
        for treatment in sorted(non_identifiable.keys()):
            details = non_identifiable[treatment]
            blockers = details.get("confounders", []) if isinstance(details, dict) else []
            notes = details.get("notes") if isinstance(details, dict) else None
            if blockers:
                print(f"  - {treatment} → {outcome} (blocked by: {', '.join(blockers)})")
            elif notes:
                print(f"  - {treatment} → {outcome} ({notes})")
            else:
                print(f"  - {treatment} → {outcome}")
        treatments = [t for t in treatments if t not in non_identifiable]
        print(f"Continuing with {len(treatments)} identifiable treatments")
        if not treatments:
            raise RuntimeError(
                "No identifiable treatment effects remain after filtering. "
                "All treatments are blocked by unobserved confounders."
            )

    create_markdown_artifact(
        key="causal-spec",
        markdown=f"## Causal Specification\n\n"
        f"- **Constructs**: {n_constructs}\n"
        f"- **Edges**: {n_edges}\n"
        f"- **Indicators**: {n_indicators}\n"
        f"- **Non-identifiable treatments**: "
        f"{list(non_identifiable.keys()) if non_identifiable else 'none'}\n",
    )

    # Combine into full causal spec with identifiability status
    causal_spec = build_causal_spec(latent_model, measurement_model, identifiability_status)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: Parallel indicator population (worker chunk size)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 2: Worker Extraction ===")
    worker_chunks = load_worker_chunks(lines)
    print(f"Loaded {len(worker_chunks)} worker chunks")

    worker_results = populate_indicators.map(
        worker_chunks,
        question=unmapped(question),
        causal_spec=unmapped(causal_spec),
    )

    # Combine raw worker results
    raw_data = combine_worker_results(worker_results)
    raw_data_result = raw_data.result() if hasattr(raw_data, "result") else raw_data
    n_observations = len(raw_data_result)
    n_unique_indicators = raw_data_result["indicator"].n_unique() if n_observations > 0 else 0
    print(f"  Combined {n_observations} observations across {n_unique_indicators} indicators")

    # Aggregate to measurement_granularity
    aggregated = aggregate_measurements(causal_spec, worker_results)
    aggregated_result = aggregated.result() if hasattr(aggregated, "result") else aggregated
    if aggregated_result:
        data_for_model = flatten_aggregated_data(aggregated_result)
        n_agg = len(data_for_model)
        print(
            f"  Aggregated to {n_agg} observations across {list(aggregated_result.keys())} granularities"
        )
    else:
        data_for_model = raw_data_result
        print("  No aggregation applied (using raw data)")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3: Validate Extraction
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 3: Extraction Validation ===")
    validation_task = validate_extraction(causal_spec, worker_results)
    validation_report = (
        validation_task.result() if hasattr(validation_task, "result") else validation_task
    )

    if validation_report:
        issues = validation_report.get("issues", [])
        if not validation_report.get("is_valid", True):
            print("⚠️  Stage 3 validation errors detected:")
            for issue in issues:
                print(
                    f"    - {issue['indicator']}: {issue['issue_type']} ({issue['severity']}) {issue['message']}"
                )
        elif issues:
            print("⚠️  Stage 3 validation warnings:")
            for issue in issues:
                print(
                    f"    - {issue['indicator']}: {issue['issue_type']} ({issue['severity']}) {issue['message']}"
                )

    if validation_report and validation_report.get("issues"):
        create_table_artifact(
            key="validation-issues",
            table=[
                {
                    "indicator": i["indicator"],
                    "type": i["issue_type"],
                    "severity": i["severity"],
                    "message": i["message"],
                }
                for i in validation_report["issues"]
            ],
            description="Stage 3 extraction validation issues",
        )

    # Hard gate: abort if no usable data
    if validation_report and not validation_report.get("is_valid", True):
        n_data = len(data_for_model) if hasattr(data_for_model, "__len__") else 0
        if n_data == 0:
            raise RuntimeError(
                "Stage 3 validation failed and no usable data remains. "
                "Cannot proceed to model specification."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: Model Specification (Orchestrator-Worker Architecture)
    # ══════════════════════════════════════════════════════════════════════════
    from causal_ssm_agent.utils.config import get_config

    config = get_config()
    lit_enabled = (
        enable_literature
        if enable_literature is not None
        else config.stage4_prior_elicitation.literature_search.enabled
    )

    print("\n=== Stage 4: Model Specification ===")
    stage4_result = await stage4_orchestrated_flow(
        causal_spec=causal_spec,
        question=question,
        raw_data=data_for_model,
        enable_literature=lit_enabled,
    )

    model_spec = stage4_result.get("model_spec", {})
    print(f"Parameters: {len(model_spec.get('parameters', []))} total")

    # Report validation issues
    validation = stage4_result.get("validation", {})
    if not validation.get("is_valid", True):
        issues = validation.get("issues", [])
        print(f"⚠️  Stage 4 prior validation failed ({len(issues)} issues):")
        for issue in issues:
            if isinstance(issue, dict):
                print(f"    - {issue.get('parameter')}: {issue.get('issue')}")
            else:
                print(f"    - {issue}")

    model_info = stage4_result.get("model_info", {})
    if not model_info.get("model_built", True):
        print(f"⚠️  Stage 4 model build failed: {model_info.get('error')}")

    create_markdown_artifact(
        key="model-spec",
        markdown=f"## Model Specification\n\n"
        f"- **Parameters**: {len(model_spec.get('parameters', []))}\n"
        f"- **Priors valid**: {validation.get('is_valid', 'unknown')}\n"
        f"- **Model built**: {model_info.get('model_built', 'unknown')}\n",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4b: Parametric Identifiability Diagnostics
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 4b: Parametric Identifiability ===")
    stage4_result = stage4b_parametric_id_flow(stage4_result, raw_data=data_for_model)

    param_id = stage4_result.get("parametric_id", {})
    if param_id.get("checked", False):
        t_rule = param_id.get("t_rule", {})
        if not t_rule.get("satisfies", True):
            raise RuntimeError(
                f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
                f"> {t_rule.get('n_moments')} moment conditions. "
                "Model is provably non-identified. Halting pipeline."
            )
        summary = param_id.get("summary", {})
        if summary.get("structural_issues"):
            print("⚠️  STRUCTURAL non-identifiability detected — some parameters unconstrained")
        elif summary.get("boundary_issues"):
            print("⚠️  Boundary identifiability issues at some prior draws")
        else:
            print("Parametric identifiability OK")
        weak = summary.get("weak_params", [])
        if weak:
            print(f"  Weak parameters (low contraction): {weak}")
    else:
        print(f"  Skipped: {param_id.get('error', 'unknown')}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5: Fit and intervene (with identifiability awareness)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 5: Inference ===")
    print(f"Estimating effects of {len(treatments)} treatments on {outcome}")
    sampler_config = (
        config.inference.to_sampler_config(method_override=inference_method)
        if inference_method
        else None
    )

    if config.inference.gpu:
        # ── GPU path: dispatch all stage 5 tasks to Modal ──
        from causal_ssm_agent.flows.gpu_inference import run_stage5_gpu

        print(f"Dispatching to Modal ({config.inference.gpu} GPU)...")
        gpu_result = run_stage5_gpu(
            stage4_result=stage4_result,
            raw_data=data_for_model,
            sampler_config=sampler_config,
            treatments=treatments,
            outcome=outcome,
            causal_spec=causal_spec,
            gpu=config.inference.gpu,
        )
        ps_result = gpu_result["ps_result"]
        ppc_result = gpu_result.get("ppc_result", {"checked": False})
        intervention_results = gpu_result["intervention_results"]
    else:
        # ── Local path: run stage 5 tasks via Prefect ──
        fitted = fit_model(stage4_result, data_for_model, sampler_config=sampler_config)

        # Post-fit power-scaling sensitivity diagnostic
        power_scaling = run_power_scaling(fitted, data_for_model)
        ps_result = power_scaling.result() if hasattr(power_scaling, "result") else power_scaling

        # Posterior predictive checks
        ppc_task = run_ppc(fitted, data_for_model)
        ppc_result = ppc_task.result() if hasattr(ppc_task, "result") else ppc_task

        # Run interventions for all treatments (with PPC + power-scaling warnings)
        results = run_interventions(
            fitted, treatments, outcome, causal_spec, ppc_result, ps_result=ps_result
        )
        intervention_results = results.result() if hasattr(results, "result") else results

    # Print power-scaling results (shared by both paths)
    print("\n--- Power-Scaling Sensitivity ---")
    if ps_result.get("checked", False):
        diagnosis = ps_result.get("diagnosis", {})
        prior_dominated = [k for k, v in diagnosis.items() if v == "prior_dominated"]
        conflicts = [k for k, v in diagnosis.items() if v == "prior_data_conflict"]
        if prior_dominated:
            print(f"  Prior-dominated parameters: {prior_dominated}")
        if conflicts:
            print(f"  Prior-data conflicts: {conflicts}")
        if not prior_dominated and not conflicts:
            print("  All parameters well-identified")
    else:
        print(f"  Skipped: {ps_result.get('error', 'unknown')}")

    # Print PPC results
    print("\n--- Posterior Predictive Checks ---")
    if ppc_result.get("checked", False):
        ppc_warnings = ppc_result.get("warnings", [])
        if ppc_warnings:
            print(f"  {len(ppc_warnings)} warning(s):")
            for w in ppc_warnings:
                print(f"    - {w['variable']}: {w['message']}")
        else:
            print("  All checks passed")
    else:
        print(f"  Skipped: {ppc_result.get('error', 'unknown')}")

    # Print ranked results table
    print(f"\n=== Treatment Ranking by Effect on {outcome} ===")
    if intervention_results:
        print(f"{'Rank':<5} {'Treatment':<30} {'Effect':>10} {'95% CI':>22} {'P(>0)':>8} {'ID':>4}")
        print("-" * 81)
        for rank, entry in enumerate(intervention_results, 1):
            name = entry["treatment"]
            effect = entry.get("effect_size")
            ci = entry.get("credible_interval")
            prob = entry.get("prob_positive")
            ident = "yes" if entry.get("identifiable", True) else "NO"

            if effect is not None:
                ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
                prob_str = f"{prob:.2f}" if prob is not None else ""
                line = (
                    f"{rank:<5} {name:<30} {effect:>+10.4f} {ci_str:>22} {prob_str:>8} {ident:>4}"
                )
                if entry.get("prior_sensitivity_warning"):
                    line += "  *"
                print(line)
            else:
                warning = entry.get("warning", "no estimate")
                print(f"{rank:<5} {name:<30} {'—':>10} {'':>22} {'':>8} {ident:>4}  ({warning})")

        # Print prior-sensitivity footnotes
        ps_entries = [e for e in intervention_results if e.get("prior_sensitivity_warning")]
        if ps_entries:
            print("\n  * Prior-dominated effects:")
            for e in ps_entries:
                print(f"    {e['treatment']}: {e['prior_sensitivity_warning']}")

    if intervention_results:
        create_table_artifact(
            key="treatment-ranking",
            table=[
                {
                    "rank": i + 1,
                    "treatment": r["treatment"],
                    "effect": (
                        f"{r['effect_size']:+.4f}" if r.get("effect_size") is not None else "—"
                    ),
                    "95% CI": (
                        f"[{r['credible_interval'][0]:+.3f}, {r['credible_interval'][1]:+.3f}]"
                        if r.get("credible_interval")
                        else ""
                    ),
                    "P(>0)": (
                        f"{r['prob_positive']:.2f}" if r.get("prob_positive") is not None else ""
                    ),
                    "identifiable": "yes" if r.get("identifiable", True) else "NO",
                }
                for i, r in enumerate(intervention_results)
            ],
            description="Final treatment effect ranking",
        )

    return {"intervention_results": intervention_results, "ppc": ppc_result}


if __name__ == "__main__":
    # Serve the flow for UI access
    causal_inference_pipeline.serve(
        name="causal-inference",
        tags=["causal", "llm"],
    )
