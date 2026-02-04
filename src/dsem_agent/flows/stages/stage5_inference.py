"""Stage 5: Bayesian inference and intervention analysis.

Fits the CT-SEM model and runs counterfactual interventions to
estimate treatment effects, ranked by effect size.

NOTE: Uses CTSEMModelBuilder with raw timestamped data.
No upfront aggregation - CT-SEM handles irregular time intervals directly.
"""

from typing import Any

import polars as pl
from prefect import task


@task
def fit_model(stage4_result: dict, raw_data: pl.DataFrame) -> Any:
    """Fit the CT-SEM model to data.

    Args:
        stage4_result: Result from stage4_orchestrated_flow containing
            glmm_spec, priors, and model_info
        raw_data: Raw timestamped data (indicator, value, timestamp)

    Returns:
        Fitted model results

    NOTE: Implementation will be merged from numpyro-ctsem.
    """
    from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

    glmm_spec = stage4_result.get("glmm_spec", {})
    priors = stage4_result.get("priors", {})

    try:
        builder = CTSEMModelBuilder(glmm_spec=glmm_spec, priors=priors)

        # Convert raw data to wide format
        if raw_data.is_empty():
            return {"fitted": False, "error": "No data available"}

        wide_data = (
            raw_data
            .with_columns(pl.col("value").cast(pl.Float64, strict=False))
            .pivot(on="indicator", index="timestamp", values="value")
            .sort("timestamp")
        )

        X = wide_data.to_pandas()
        if "timestamp" in X.columns:
            X = X.rename(columns={"timestamp": "time"})

        # Fit the model
        mcmc_result = builder.fit(X)

        return {
            "fitted": True,
            "mcmc": mcmc_result,
            "builder": builder,
        }

    except NotImplementedError:
        # Expected until numpyro-ctsem is merged
        return {
            "fitted": False,
            "error": "CT-SEM implementation pending merge from numpyro-ctsem",
        }
    except Exception as e:
        return {
            "fitted": False,
            "error": str(e),
        }


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
