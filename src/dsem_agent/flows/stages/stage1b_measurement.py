"""Stage 1b: Measurement Model Proposal (Orchestrator).

The orchestrator proposes indicators to operationalize the theoretical
constructs from Stage 1a, using sample data to inform operationalization.

This follows the Anderson & Gerbing (1988) two-step approach where the
measurement model is specified after the latent model.
"""

from pathlib import Path

from prefect import task
from prefect.cache_policies import INPUTS

from dsem_agent.orchestrator.agents import (
    build_dsem_model as build_dsem_model_agent,
    propose_measurement_model as propose_measurement_model_agent,
    request_proxy_measurements,
)
from dsem_agent.utils.data import (
    get_orchestrator_chunk_size,
    load_text_chunks as load_text_chunks_util,
)
from dsem_agent.utils.identifiability import (
    check_identifiability,
    format_identifiability_report,
)


@task(cache_policy=INPUTS)
def load_orchestrator_chunks(input_path: Path) -> list[str]:
    """Load chunks sized for orchestrator (stage 1b)."""
    return load_text_chunks_util(input_path, chunk_size=get_orchestrator_chunk_size())


@task(cache_policy=INPUTS)
def build_dsem_model(
    latent_model: dict,
    measurement_model: dict,
    identifiability_status: dict | None = None
) -> dict:
    """Combine latent and measurement models into full DSEMModel with identifiability.

    Args:
        latent_model: The latent model dict from Stage 1a
        measurement_model: The measurement model dict from Stage 1b
        identifiability_status: Status of effect identifiability

    Returns:
        DSEMModel as a dictionary with 'latent', 'measurement', and 'identifiability'
    """
    dsem = build_dsem_model_agent(latent_model, measurement_model)
    dsem['identifiability'] = identifiability_status
    return dsem


@task(cache_policy=INPUTS)
def propose_measurement_with_identifiability_fix(
    question: str,
    latent_model: dict,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """Propose measurements and attempt to fix identifiability issues.

    This is the enhanced Stage 1b that:
    1. Proposes initial measurements
    2. Checks identifiability for all treatment→outcome effects
    3. Requests proxies for blocking confounders if needed (one-shot)
    4. Returns final measurements with identifiability status

    Args:
        question: The causal research question
        latent_model: The latent model dict from Stage 1a
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset

    Returns:
        Dict with 'measurement_model' and 'identifiability_status'
    """
    # Step 1: Initial measurement proposal
    print("Proposing initial measurements...")
    measurement = propose_measurement_model_agent(
        question, latent_model, data_sample, dataset_summary
    )

    # Step 2: Check identifiability
    print("\nChecking identifiability for all treatment effects...")
    id_result = check_identifiability(latent_model, measurement)
    print(format_identifiability_report(id_result))

    # Step 3: If issues, request proxies (one-shot)
    if id_result['non_identifiable_treatments']:
        print("\nRequesting proxy measurements for blocking confounders...")

        # Get unique confounders across all non-identifiable treatments
        all_confounders = set()
        for blockers in id_result['blocking_confounders'].values():
            all_confounders.update(blockers)

        # Filter to actual constructs (not "unknown" errors)
        construct_names = {c['name'] for c in latent_model['constructs']}
        confounders_to_fix = [c for c in all_confounders if c in construct_names]

        if confounders_to_fix:
            # Format blocking info for LLM
            blocking_info = "\n".join([
                f"- {treatment}: blocked by {', '.join(id_result['blocking_confounders'][treatment])}"
                for treatment in sorted(id_result['non_identifiable_treatments'])
                if treatment in id_result['blocking_confounders']
            ])

            # Request proxies
            proxy_response = request_proxy_measurements(
                question,
                latent_model,
                measurement,
                blocking_info,
                confounders_to_fix,
                data_sample,
            )

            # Merge new proxies into measurement model
            if proxy_response.get('new_proxies'):
                print(f"Found proxies for {len(proxy_response['new_proxies'])} confounders")
                for proxy in proxy_response['new_proxies']:
                    for indicator_name in proxy['indicators']:
                        measurement['indicators'].append({
                            'name': indicator_name,
                            'construct': proxy['construct'],
                            'how_to_measure': f"Proxy for {proxy['construct']}: {proxy['justification']}",
                        })

                # Re-check identifiability
                print("\nRe-checking identifiability after adding proxies...")
                id_result = check_identifiability(latent_model, measurement)
                print(format_identifiability_report(id_result))

            # Report unfeasible confounders
            if proxy_response.get('unfeasible_confounders'):
                print("\nCould not find proxies for:")
                for unf in proxy_response['unfeasible_confounders']:
                    print(f"  - {unf['construct']}: {unf['reason']}")

    # Return measurement with identifiability status
    return {
        'measurement_model': measurement,
        'identifiability_status': {
            'outcome': id_result['outcome'],
            'all_identifiable': len(id_result['non_identifiable_treatments']) == 0,
            'non_identifiable_treatments': id_result['non_identifiable_treatments'],
            'blocking_confounders': id_result['blocking_confounders'],
        }
    }
