"""Stage 4: Model Specification & Prior Elicitation.

Translates the causal DAG (topological structure) into a PyMC-ready
functional specification. Combines rule-based constraints with
LLM-assisted prior elicitation.

See docs/modeling/functional_spec.md for design rationale.
"""

from prefect import task
from prefect.cache_policies import INPUTS

from dsem_agent.orchestrator.schemas import (
    DSEMModel,
    Role,
    TemporalStatus,
)


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED SPECIFICATION
# ══════════════════════════════════════════════════════════════════════════════

# Link functions determined by measurement dtype
DTYPE_TO_DISTRIBUTION = {
    "continuous": {"dist": "Normal", "link": "identity"},
    "binary": {"dist": "Bernoulli", "link": "logit"},
    "count": {"dist": "Poisson", "link": "log"},
    "ordinal": {"dist": "OrderedLogistic", "link": "cumulative_logit"},
    "categorical": {"dist": "Categorical", "link": "softmax"},
}

# Default priors for parameters (can be overridden by LLM elicitation)
DEFAULT_PRIORS = {
    # AR(1) coefficient: uniform on [0, 1] for stability
    "ar": {"dist": "Uniform", "lower": 0.0, "upper": 1.0},
    # Cross-lag coefficients: weakly informative normal
    "beta": {"dist": "Normal", "mean": 0.0, "std": 0.5},
    # Residual standard deviation: half-normal
    "sigma": {"dist": "HalfNormal", "sigma": 1.0},
    # Factor loadings (for multi-indicator): half-normal
    "loading": {"dist": "HalfNormal", "sigma": 1.0},
}


def _get_model_clock(dsem_model: DSEMModel) -> str:
    """Determine the model clock (finest endogenous outcome granularity).

    The model operates at the finest temporal resolution among
    endogenous time-varying constructs.
    """
    from dsem_agent.orchestrator.schemas import GRANULARITY_HOURS

    finest_hours = float("inf")
    finest_gran = "daily"  # default

    for construct in dsem_model.latent.constructs:
        if (
            construct.role == Role.ENDOGENOUS
            and construct.temporal_status == TemporalStatus.TIME_VARYING
            and construct.causal_granularity
        ):
            hours = GRANULARITY_HOURS.get(construct.causal_granularity, 24)
            if hours < finest_hours:
                finest_hours = hours
                finest_gran = construct.causal_granularity

    return finest_gran


def _build_construct_specs(dsem_model: DSEMModel) -> dict:
    """Build specification for each construct."""
    specs = {}

    for construct in dsem_model.latent.constructs:
        spec = {
            "name": construct.name,
            "role": construct.role.value,
            "temporal_status": construct.temporal_status.value,
            "is_outcome": construct.is_outcome,
        }

        if construct.temporal_status == TemporalStatus.TIME_VARYING:
            spec["granularity"] = construct.causal_granularity

            # Endogenous time-varying constructs get AR(1)
            if construct.role == Role.ENDOGENOUS:
                spec["ar_prior"] = DEFAULT_PRIORS["ar"].copy()

        # Residual variance for endogenous constructs
        if construct.role == Role.ENDOGENOUS:
            spec["sigma_prior"] = DEFAULT_PRIORS["sigma"].copy()

        specs[construct.name] = spec

    return specs


def _build_edge_specs(dsem_model: DSEMModel) -> dict:
    """Build specification for each causal edge (cross-lag coefficient)."""
    specs = {}

    for edge in dsem_model.latent.edges:
        param_name = f"beta_{edge.effect}_{edge.cause}"

        specs[param_name] = {
            "cause": edge.cause,
            "effect": edge.effect,
            "lagged": edge.lagged,
            "lag_hours": dsem_model.get_edge_lag_hours(edge),
            "prior": DEFAULT_PRIORS["beta"].copy(),
        }

    return specs


def _build_measurement_specs(dsem_model: DSEMModel) -> dict:
    """Build specification for measurement model (indicators)."""
    specs = {}

    # Group indicators by construct
    indicators_by_construct: dict[str, list] = {}
    for indicator in dsem_model.measurement.indicators:
        if indicator.construct_name not in indicators_by_construct:
            indicators_by_construct[indicator.construct_name] = []
        indicators_by_construct[indicator.construct_name].append(indicator)

    for indicator in dsem_model.measurement.indicators:
        construct_indicators = indicators_by_construct[indicator.construct_name]
        is_single = len(construct_indicators) == 1

        dist_info = DTYPE_TO_DISTRIBUTION.get(
            indicator.measurement_dtype,
            DTYPE_TO_DISTRIBUTION["continuous"],
        )

        spec = {
            "name": indicator.name,
            "construct": indicator.construct_name,
            "dtype": indicator.measurement_dtype,
            "distribution": dist_info["dist"],
            "link": dist_info["link"],
            "granularity": indicator.measurement_granularity,
            "aggregation": indicator.aggregation,
        }

        # Single-indicator: loading fixed to 1 (absorbed measurement error)
        # Multi-indicator: estimate loadings (first fixed to 1 for identification)
        if is_single:
            spec["loading"] = 1.0  # fixed
            spec["loading_prior"] = None
        else:
            first_indicator = construct_indicators[0]
            if indicator.name == first_indicator.name:
                spec["loading"] = 1.0  # reference indicator
                spec["loading_prior"] = None
            else:
                spec["loading"] = None  # estimated
                spec["loading_prior"] = DEFAULT_PRIORS["loading"].copy()

        specs[indicator.name] = spec

    return specs


def _build_identifiability_specs(dsem_model: DSEMModel) -> dict:
    """Extract identifiability status for flagging in results."""
    id_status = dsem_model.identifiability

    if id_status is None:
        return {
            "identifiable": [],
            "non_identifiable": [],
        }

    return {
        "identifiable": list(id_status.identifiable_treatments.keys()),
        "non_identifiable": list(id_status.non_identifiable_treatments.keys()),
    }


@task(retries=1, cache_policy=INPUTS)
def specify_model(latent_dict: dict, dsem_dict: dict) -> dict:
    """Generate PyMC-ready model specification from DSEM model.

    This is the rule-based component of Stage 4. It determines:
    - Model clock (finest temporal resolution)
    - AR(1) structure for time-varying endogenous constructs
    - Link functions based on indicator dtypes
    - Measurement model structure (single vs. multi-indicator)
    - Default priors for all parameters

    Args:
        latent_dict: Latent model dict (unused, kept for API compatibility)
        dsem_dict: Full DSEM model dict

    Returns:
        ModelSpec dict ready for LLM prior refinement and PyMC construction
    """
    # Parse into validated model
    dsem_model = DSEMModel.model_validate(dsem_dict)

    model_spec = {
        "time_index": _get_model_clock(dsem_model),
        "constructs": _build_construct_specs(dsem_model),
        "edges": _build_edge_specs(dsem_model),
        "measurement": _build_measurement_specs(dsem_model),
        "identifiability": _build_identifiability_specs(dsem_model),
    }

    return model_spec


# ══════════════════════════════════════════════════════════════════════════════
# LLM-ASSISTED PRIOR ELICITATION
# ══════════════════════════════════════════════════════════════════════════════


@task(retries=2, retry_delay_seconds=10, task_run_name="elicit-priors")
def elicit_priors(model_spec: dict) -> dict:
    """Elicit domain-informed priors from LLM for model parameters.

    Uses AutoElicit-style paraphrased prompting to handle LLM uncertainty:
    1. Generate N paraphrased task descriptions
    2. Elicit prior parameters (mean, std) for each paraphrase
    3. Aggregate into mixture or pooled prior

    Args:
        model_spec: Output from specify_model()

    Returns:
        Refined priors dict with LLM-informed hyperparameters

    TODO: Implement LLM elicitation following docs/modeling/functional_spec.md
        - Use orchestrator LLM to propose effect size priors
        - Paraphrase prompts N times for robustness
        - Aggregate responses into final priors
        - Fall back to defaults if elicitation fails
    """
    # For now, return the default priors from model_spec unchanged
    # LLM elicitation will refine these
    priors = {}

    # Extract AR priors
    for name, spec in model_spec.get("constructs", {}).items():
        if "ar_prior" in spec and spec["ar_prior"]:
            priors[f"rho_{name}"] = spec["ar_prior"]
        if "sigma_prior" in spec and spec["sigma_prior"]:
            priors[f"sigma_{name}"] = spec["sigma_prior"]

    # Extract edge priors
    for param_name, spec in model_spec.get("edges", {}).items():
        priors[param_name] = spec["prior"]

    # Extract loading priors
    for name, spec in model_spec.get("measurement", {}).items():
        if spec.get("loading_prior"):
            priors[f"lambda_{name}"] = spec["loading_prior"]

    return priors
