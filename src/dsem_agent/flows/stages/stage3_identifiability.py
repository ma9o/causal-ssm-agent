"""Stage 3: Identifiability checking.

Reports on which treatment effects are identifiable based on the
identifiability status computed in Stage 1b.
"""

from prefect import task
from prefect.cache_policies import INPUTS


@task(cache_policy=INPUTS)
def check_identifiability(dsem_model: dict) -> dict:
    """Report identifiability status from Stage 1b.

    Args:
        dsem_model: The full DSEM model with identifiability status

    Returns:
        Dict with identifiability report
    """
    id_status = dsem_model.get('identifiability')

    if not id_status:
        return {
            'status': 'unknown',
            'message': 'No identifiability check was performed',
            'non_identifiable_treatments': set(),
        }

    outcome = id_status.get('outcome', 'unknown')

    if id_status['all_identifiable']:
        return {
            'status': 'identifiable',
            'message': f'All treatment effects on {outcome} are identifiable',
            'non_identifiable_treatments': set(),
        }

    # Report non-identifiable treatments
    non_id = id_status['non_identifiable_treatments']
    return {
        'status': 'not_identifiable',
        'message': f"{len(non_id)} treatment effects on {outcome} are NOT identifiable",
        'non_identifiable_treatments': non_id,
        'blocking_confounders': id_status.get('blocking_confounders', {}),
        'recommendation': 'Consider sensitivity analysis for these effects',
    }


@task
def run_sensitivity_analysis(dsem_model: dict, treatment: str) -> dict:
    """Run sensitivity analysis for a non-identifiable treatment effect.

    TODO: Implement Cinelli-Hazlett sensitivity analysis.
    """
    pass
