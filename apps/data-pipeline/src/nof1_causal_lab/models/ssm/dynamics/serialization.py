"""Serializable composite-spec config: the bridge between Stage 4's
dict-config priors and the runtime ``CompositeSpec`` consumed by
``compile_composite``.

Composite specs store prior config using the same canonical
``{"family": ..., "params": {...}}`` shape as the site prior registry,
so Stage 4 can emit nonlinear dynamics specs that round-trip through
JSON / pydantic without distribution objects.

Schema
------

Each component is a single dict with a ``kind`` discriminator plus
component-specific indices and a ``priors`` sub-dict mapping each
parameter name to a ``{"family": <PriorDistributionFamily>, "params":
{...}}`` config.

Example::

    {
        "n_latent": 5,
        "components": [
            {"kind": "DiagonalDecay",
             "priors": {"decay": {"family": "Gamma",
                                  "params": {"concentration": 2.0, "rate": 4.0}}}},
            {"kind": "HillEdge",
             "source": 3, "target": 4,
             "priors": {
                 "Emax": {"family": "LogNormal", "params": {"mu": 0.0, "sigma": 0.5}},
                 "EC50": {"family": "LogNormal", "params": {"mu": 0.0, "sigma": 0.5}},
                 "n":    {"family": "TruncatedNormal",
                          "params": {"mu": 2.0, "sigma": 0.5, "lower": 1.0, "upper": 4.0}}}}
        ]
    }
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .composite import (
    CompositeSpec,
    DenseLinearSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    InterceptSpec,
    LinearEdgeSpec,
    MultiplicativeEdgeSpec,
    StructuralDenseLinearSpec,
    StructuralInterceptSpec,
)

_COMPONENT_KIND_REGISTRY: dict[str, str] = {
    "DenseLinear": "dense_linear",
    "DiagonalDecay": "diagonal_decay",
    "Intercept": "intercept",
    "LinearEdge": "linear_edge",
    "HillEdge": "hill_edge",
    "MultiplicativeEdge": "multiplicative_edge",
    "StructuralDenseLinear": "structural_dense_linear",
    "StructuralIntercept": "structural_intercept",
}


def _prior_to_dict(prior: Any) -> Any:
    """Return the dict-config form of a prior for JSON serialisation.

    Priors are stored canonically as dict-configs on
    :class:`CompositeSpec` components. A ``numpyro.Distribution`` here
    indicates the spec was constructed for runtime use (research / tests)
    and cannot round-trip through JSON — convert to a dict-config before
    serialising.
    """
    if prior is None:
        return None
    if isinstance(prior, dict):
        return prior
    raise TypeError(
        "Cannot serialise prior as dict-config: got "
        f"{type(prior).__name__}. Construct the spec component with a "
        "dict-config prior ({'family': ..., 'params': {...}}) when JSON "
        "round-trip is required."
    )


def composite_spec_to_dict(spec: CompositeSpec) -> dict[str, Any]:
    """Inverse of :func:`composite_spec_from_dict`. Round-trips dict-config
    priors. Distribution-typed priors are passed through as-is (callers
    that need JSON serialisation must store priors as dict-configs).
    """
    from .composite import (
        DenseLinearSpec,
        DiagonalDecaySpec,
        HillEdgeSpec,
        InterceptSpec,
        LinearEdgeSpec,
        MultiplicativeEdgeSpec,
        StructuralDenseLinearSpec,
        StructuralInterceptSpec,
    )

    components: list[dict[str, Any]] = []
    for component in spec.components:
        if isinstance(component, DenseLinearSpec):
            entry: dict[str, Any] = {
                "kind": "DenseLinear",
                "priors": {"drift": _prior_to_dict(component.drift_prior)},
            }
            if component.cint_prior is not None:
                entry["priors"]["cint"] = _prior_to_dict(component.cint_prior)
        elif isinstance(component, DiagonalDecaySpec):
            entry = {
                "kind": "DiagonalDecay",
                "priors": {"decay": _prior_to_dict(component.decay_prior)},
            }
        elif isinstance(component, InterceptSpec):
            entry = {
                "kind": "Intercept",
                "priors": {"cint": _prior_to_dict(component.cint_prior)},
            }
        elif isinstance(component, LinearEdgeSpec):
            entry = {
                "kind": "LinearEdge",
                "source": int(component.source),
                "target": int(component.target),
                "priors": {"weight": _prior_to_dict(component.weight_prior)},
            }
        elif isinstance(component, HillEdgeSpec):
            entry = {
                "kind": "HillEdge",
                "source": int(component.source),
                "target": int(component.target),
                "priors": {
                    "Emax": _prior_to_dict(component.emax_prior),
                    "EC50": _prior_to_dict(component.ec50_prior),
                    "n": _prior_to_dict(component.n_prior),
                },
            }
        elif isinstance(component, MultiplicativeEdgeSpec):
            entry = {
                "kind": "MultiplicativeEdge",
                "source_a": int(component.source_a),
                "source_b": int(component.source_b),
                "target": int(component.target),
                "priors": {"weight": _prior_to_dict(component.weight_prior)},
            }
        elif isinstance(component, StructuralDenseLinearSpec):
            entry = {
                "kind": "StructuralDenseLinear",
                "n_latent": int(component.n_latent),
                "drift_diag_mask": np.asarray(component.drift_diag_mask).tolist(),
                "drift_offdiag_mask": np.asarray(component.drift_offdiag_mask).tolist(),
                "drift_template": np.asarray(component.drift_template).tolist(),
                "stability_margin": float(component.stability_margin),
                "time_invariant_mask": (
                    np.asarray(component.time_invariant_mask).tolist()
                    if component.time_invariant_mask is not None
                    else None
                ),
                "priors": {},
            }
            if component.base_decay_prior is not None:
                entry["priors"]["base_decay"] = _prior_to_dict(component.base_decay_prior)
            if component.offdiag_prior is not None:
                entry["priors"]["offdiag"] = _prior_to_dict(component.offdiag_prior)
        elif isinstance(component, StructuralInterceptSpec):
            entry = {
                "kind": "StructuralIntercept",
                "n_latent": int(component.n_latent),
                "cint_mask": np.asarray(component.cint_mask).tolist(),
                "cint_template": np.asarray(component.cint_template).tolist(),
                "priors": {},
            }
            if component.cint_prior is not None:
                entry["priors"]["cint"] = _prior_to_dict(component.cint_prior)
        else:
            raise ValueError(
                f"composite_spec_to_dict: unsupported component type "
                f"{type(component).__name__}"
            )
        components.append(entry)
    return {"n_latent": int(spec.n_latent), "components": components}


def _spec_from_component_dict(component: dict[str, Any]) -> Any:
    """Build a component spec from a dict-config.

    Priors are stored on the spec as the original dict-configs (not
    materialised). Materialisation to ``numpyro.Distribution`` happens
    lazily inside ``sample_params`` via ``resolve_prior_distribution``,
    so ``composite_spec_to_dict`` can round-trip the spec back to JSON.
    """
    kind = component["kind"]
    priors = component.get("priors", {})

    def _required(name: str) -> dict[str, Any]:
        if name not in priors:
            raise ValueError(
                f"Composite component {kind!r} missing required prior {name!r}; "
                f"available: {sorted(priors)}"
            )
        return priors[name]

    def _optional(name: str) -> dict[str, Any] | None:
        return priors.get(name)

    if kind == "DenseLinear":
        return DenseLinearSpec(
            drift_prior=_required("drift"),
            cint_prior=_optional("cint"),
        )
    if kind == "DiagonalDecay":
        return DiagonalDecaySpec(decay_prior=_required("decay"))
    if kind == "Intercept":
        return InterceptSpec(cint_prior=_required("cint"))
    if kind == "LinearEdge":
        return LinearEdgeSpec(
            source=int(component["source"]),
            target=int(component["target"]),
            weight_prior=_required("weight"),
        )
    if kind == "HillEdge":
        return HillEdgeSpec(
            source=int(component["source"]),
            target=int(component["target"]),
            emax_prior=_required("Emax"),
            ec50_prior=_required("EC50"),
            n_prior=_required("n"),
        )
    if kind == "MultiplicativeEdge":
        return MultiplicativeEdgeSpec(
            source_a=int(component["source_a"]),
            source_b=int(component["source_b"]),
            target=int(component["target"]),
            weight_prior=_required("weight"),
        )
    if kind == "StructuralDenseLinear":
        time_inv = component.get("time_invariant_mask")
        return StructuralDenseLinearSpec(
            n_latent=int(component["n_latent"]),
            drift_diag_mask=np.asarray(component["drift_diag_mask"], dtype=bool),
            drift_offdiag_mask=np.asarray(component["drift_offdiag_mask"], dtype=bool),
            drift_template=jnp.asarray(component["drift_template"]),
            stability_margin=float(component.get("stability_margin", 0.05)),
            time_invariant_mask=(
                np.asarray(time_inv, dtype=bool) if time_inv is not None else None
            ),
            base_decay_prior=_optional("base_decay"),
            offdiag_prior=_optional("offdiag"),
        )
    if kind == "StructuralIntercept":
        return StructuralInterceptSpec(
            n_latent=int(component["n_latent"]),
            cint_mask=np.asarray(component["cint_mask"], dtype=bool),
            cint_template=jnp.asarray(component["cint_template"]),
            cint_prior=_optional("cint"),
        )
    raise ValueError(
        f"Unknown composite component kind {kind!r}; "
        f"known: {sorted(_COMPONENT_KIND_REGISTRY)}"
    )


def composite_spec_from_dict(config: dict[str, Any]) -> CompositeSpec:
    """Build a ``CompositeSpec`` from a nested dict-config (see module docstring)."""
    if "n_latent" not in config:
        raise ValueError("composite_spec_from_dict requires 'n_latent'")
    components = tuple(
        _spec_from_component_dict(c) for c in config.get("components", ())
    )
    return CompositeSpec(n_latent=int(config["n_latent"]), components=components)
