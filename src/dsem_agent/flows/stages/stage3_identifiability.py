"""Stage 3: Identifiability checking.

Reports on which treatment effects are identifiable based on the
identifiability status computed in Stage 1b.
"""

from prefect import task
from prefect.cache_policies import INPUTS

from dsem_agent.utils.effects import get_outcome_from_latent_model


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
            'non_identifiable_treatments': {},
        }

    outcome = get_outcome_from_latent_model(dsem_model.get('latent', {})) or 'unknown'
    non_id = id_status.get('non_identifiable_treatments', {})

    if not non_id:
        return {
            'status': 'identifiable',
            'message': f'All treatment effects on {outcome} are identifiable',
            'non_identifiable_treatments': {},
        }

    # Report non-identifiable treatments
    return {
        'status': 'not_identifiable',
        'message': f"{len(non_id)} treatment effects on {outcome} are NOT identifiable",
        'non_identifiable_treatments': non_id,
        'recommendation': 'Consider sensitivity analysis for these effects',
    }


@task
def run_sensitivity_analysis(dsem_model: dict, treatment: str) -> dict:
    """Run sensitivity analysis for a non-identifiable treatment effect.

    TODO: Implement Cinelli-Hazlett sensitivity analysis.
    """
    pass
