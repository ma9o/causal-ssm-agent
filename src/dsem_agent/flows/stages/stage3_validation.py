"""Stage 3: Extraction Validation.

Validates that worker extraction (Stage 2) produced usable data:
1. Check that each proxy/indicator actually has data in the extracted dataframes
2. If proxies are missing or empty, flag for DAG revalidation
3. Report coverage statistics

This stage catches extraction failures before model specification.
"""

from prefect import task
from prefect.cache_policies import INPUTS


@task(cache_policy=INPUTS)
def validate_extraction(
    dsem_model: dict,
    extracted_data: dict,
) -> dict:
    """Validate that extraction produced data for all indicators.

    Args:
        dsem_model: The full DSEM model with measurement model
        extracted_data: Dict of granularity -> polars DataFrame from Stage 2

    Returns:
        Dict with:
            - valid: bool - whether extraction is usable
            - coverage: dict - indicator -> bool (has data)
            - missing_indicators: list[str] - indicators with no data
            - recommendation: str - what to do if invalid

    TODO: Implement validation logic:
        1. For each indicator in measurement_model, check if it exists in extracted_data
        2. Check for null/empty columns
        3. If critical indicators missing, suggest DAG modifications
        4. Return validation report
    """
    pass


@task(cache_policy=INPUTS)
def revalidate_dag_for_missing_proxies(
    dsem_model: dict,
    missing_indicators: list[str],
) -> dict:
    """Suggest DAG modifications when proxies have no data.

    If extraction failed for certain indicators, we may need to:
    - Remove the indicator and find alternatives
    - Remove the construct if no valid proxies exist
    - Adjust the causal structure

    TODO: Implement revalidation logic
    """
    pass
