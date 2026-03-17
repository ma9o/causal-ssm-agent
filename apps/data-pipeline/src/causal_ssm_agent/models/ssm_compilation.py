"""Public pure-compilation surface for turning ModelSpec + priors into SSM inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causal_ssm_agent.models.ssm_compilation_common import (
    PriorIndexMaps,
    normalize_prior_params,
    split_compound_name,
)
from causal_ssm_agent.models.ssm_prior_compilation import (
    bind_parameters,
    check_drift_lag_consistency,
    compile_priors,
    warn_first_order_approximation,
)
from causal_ssm_agent.models.ssm_prior_indexing import build_prior_index_maps
from causal_ssm_agent.models.ssm_spec_translation import (
    build_masks_from_causal_spec,
    get_construct_dt_days,
    get_structural_latent_layout,
    translate_spec,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec
    from causal_ssm_agent.orchestrator.schemas_model import ModelSpec


def compile_ssm_inputs(
    model_spec: ModelSpec | dict | None = None,
    priors: dict[str, dict] | None = None,
    *,
    ssm_spec: SSMSpec | None = None,
    ssm_priors: SSMPriors | None = None,
    causal_spec: dict | None = None,
) -> tuple[SSMSpec, SSMPriors, list[dict[str, object]]]:
    """Resolve executable SSM inputs from either semantic specs or precompiled state."""
    edge_lag_days: dict[tuple[int, int], float] = {}
    if ssm_spec is None:
        if model_spec is None:
            raise ValueError("Cannot compile SSM inputs without model_spec or ssm_spec")
        ssm_spec, edge_lag_days = translate_spec(model_spec, causal_spec)

    index_maps = None
    if ssm_priors is None:
        ssm_priors, index_maps = compile_priors(
            priors or {},
            model_spec,
            ssm_spec,
            edge_lag_days=edge_lag_days,
            causal_spec=causal_spec,
        )

    bindings = bind_parameters(
        model_spec,
        ssm_spec,
        index_maps=index_maps,
        causal_spec=causal_spec,
    )
    return ssm_spec, ssm_priors, bindings


__all__ = [
    "PriorIndexMaps",
    "bind_parameters",
    "build_masks_from_causal_spec",
    "build_prior_index_maps",
    "check_drift_lag_consistency",
    "compile_priors",
    "compile_ssm_inputs",
    "get_construct_dt_days",
    "get_structural_latent_layout",
    "normalize_prior_params",
    "split_compound_name",
    "translate_spec",
    "warn_first_order_approximation",
]
