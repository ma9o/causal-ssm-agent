"""Stage 5: Bayesian inference and intervention analysis.

Fits the DSEM model and runs counterfactual interventions to
estimate treatment effects, ranked by effect size.

TODO: Implement using DSEMModelBuilder from Stage 4.
"""

from typing import Any

from prefect import task


@task
def fit_model(stage4_result: dict, data: list[str]) -> Any:
    """Fit the PyMC model to data.

    Args:
        stage4_result: Result from stage4_orchestrated_flow containing
            model_spec, priors, and model_info
        data: Raw data chunks

    Returns:
        Fitted model (TODO: implement)

    TODO: Use DSEMModelBuilder.fit() from stage4_result
    """
    pass


@task
def run_interventions(
    fitted_model: Any,
    treatments: list[str],
    dsem_model: dict | None = None,
) -> list[dict]:
    """Run interventions and rank treatments by effect size.

    Args:
        fitted_model: The fitted PyMC model
        treatments: List of treatment construct names
        dsem_model: Optional DSEM model with identifiability status

    Returns:
        List of intervention results, sorted by effect size (descending)

    TODO: Implement intervention analysis and ranking.
    """
    results = []

    # Get identifiability status
    id_status = dsem_model.get('identifiability') if dsem_model else None
    non_identifiable: set[str] = set()
    blocker_details: dict[str, list[str]] = {}
    if id_status:
        non_identifiable_map = id_status.get('non_identifiable_treatments', {})
        non_identifiable = set(non_identifiable_map.keys())
        blocker_details = {
            treatment: details.get('confounders', [])
            for treatment, details in non_identifiable_map.items()
            if isinstance(details, dict)
        }

    for treatment in treatments:
        result = {
            'treatment': treatment,
            'effect_size': None,  # TODO: compute from fitted model
            'credible_interval': None,
            'identifiable': treatment not in non_identifiable,
        }

        if treatment in non_identifiable:
            blockers = blocker_details.get(treatment, [])
            if blockers:
                result['warning'] = f"Effect not identifiable (blocked by: {', '.join(blockers)})"
            else:
                result['warning'] = "Effect not identifiable (missing proxies)"

        results.append(result)

    # TODO: Sort by effect size once computed
    # results.sort(key=lambda x: x['effect_size'] or 0, reverse=True)

    return results
