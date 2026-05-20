"""Serializable composite-spec config: the bridge between Stage 4's
dict-config priors and the runtime ``CompositeSpec`` consumed by
``compile_composite``.

Closes the integration gap where ``dynamics/priors.py`` returns
``Distribution`` objects directly while the rest of the framework
(``SSMPriors``, ``PriorProposal``) speaks dict-config. With this module a
composite spec round-trips through JSON / pydantic exactly like the
linear-path priors, so anywhere Stage 4 already emits a prior
dict-config it can now emit a composite spec instead.

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
import numpyro.distributions as ndist

from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_kind_from_index,
    get_real_runtime_kind_from_index,
)

from .compilation import (
    CompositeSpec,
    DenseLinearSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    InterceptSpec,
    LinearEdgeSpec,
    MultiplicativeEdgeSpec,
    compile_composite,
)


def _materialize_legacy_prior(prior: dict[str, Any]) -> ndist.Distribution:
    """Materialise the legacy SSMPriors flat dict-config format.

    Ports the original ``models/ssm/model.py:_make_prior_dist`` logic
    here so the linear-path priors share a single materialiser with
    the composite-path priors. Accepts:

    - ``{"mu": ..., "sigma": ...}`` → Normal (or TruncatedNormal if
      ``lower``/``upper`` present).
    - ``{"family": <int>, "mu": ..., "sigma": ..., "lower"?, "upper"?}``
      → real-valued family via ``get_real_runtime_kind_from_index``.
    - ``{"family": <int>, "concentration": ..., "rate": ...}`` etc.
      → positive-valued family via ``get_positive_runtime_kind_from_index``.
    """
    family = prior.get("family", 0)
    if isinstance(family, list):
        unique_families = {int(value) for value in family}
        if len(unique_families) != 1:
            raise ValueError("Mixed prior families within a single SSM field are unsupported")
        family = unique_families.pop()
    if "mu" in prior or "lower" in prior or "upper" in prior:
        if "family" in prior:
            runtime_kind = get_real_runtime_kind_from_index(int(family))
            if runtime_kind == PriorDistributionFamily.NORMAL:
                return ndist.Normal(jnp.asarray(prior["mu"]), jnp.asarray(prior["sigma"]))
            if runtime_kind == PriorDistributionFamily.TRUNCATED_NORMAL:
                return ndist.TruncatedNormal(
                    loc=jnp.asarray(prior["mu"]),
                    scale=jnp.asarray(prior["sigma"]),
                    low=jnp.asarray(prior["lower"]),
                    high=jnp.asarray(prior["upper"]),
                )
            if runtime_kind == PriorDistributionFamily.UNIFORM:
                return ndist.Uniform(
                    low=jnp.asarray(prior["lower"]),
                    high=jnp.asarray(prior["upper"]),
                )
            raise ValueError(f"Unsupported serialized real prior runtime kind {runtime_kind!r}")
        if "lower" in prior and "upper" in prior:
            return ndist.TruncatedNormal(
                loc=jnp.asarray(prior["mu"]),
                scale=jnp.asarray(prior["sigma"]),
                low=jnp.asarray(prior["lower"]),
                high=jnp.asarray(prior["upper"]),
            )
        return ndist.Normal(jnp.asarray(prior["mu"]), jnp.asarray(prior["sigma"]))
    if "family" in prior:
        runtime_kind = get_positive_runtime_kind_from_index(int(family))
        if runtime_kind == PriorDistributionFamily.HALF_NORMAL:
            return ndist.HalfNormal(jnp.asarray(prior["sigma"]))
        if runtime_kind == PriorDistributionFamily.GAMMA:
            return ndist.Gamma(
                concentration=jnp.asarray(prior.get("concentration", 2.0)),
                rate=jnp.asarray(prior.get("rate", 1.0)),
            )
        if runtime_kind == PriorDistributionFamily.LOG_NORMAL:
            return ndist.LogNormal(
                loc=jnp.asarray(prior.get("loc", 0.0)),
                scale=jnp.asarray(prior.get("sigma", 1.0)),
            )
        if runtime_kind == PriorDistributionFamily.EXPONENTIAL:
            return ndist.Exponential(rate=jnp.asarray(prior.get("rate", 1.0)))
        if runtime_kind == PriorDistributionFamily.DELTA:
            return ndist.Delta(jnp.asarray(prior["value"]))
        raise ValueError(f"Unsupported serialized positive prior runtime kind {runtime_kind!r}")
    if {"concentration", "rate"} <= set(prior):
        return ndist.Gamma(
            concentration=jnp.asarray(prior.get("concentration", 2.0)),
            rate=jnp.asarray(prior.get("rate", 1.0)),
        )
    return ndist.HalfNormal(jnp.asarray(prior["sigma"]))

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
    from .compilation import (
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
                "bare_site_names": bool(component.bare_site_names),
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
                "bare_site_names": bool(component.bare_site_names),
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


def materialize_prior(prior_cfg: dict[str, Any]) -> ndist.Distribution:
    """Materialize a NumPyro ``Distribution`` from a dict-config.

    Accepts both prior-config formats used across the codebase:

    - **New format** (Stage 4 / composite-spec): nested ``params``
      sub-dict — ``{"family": "Normal", "params": {"mu": 0.0, "sigma":
      1.0}, "shape": [...]}``. ``family`` is a string family name.
      ``shape`` controls broadcasting.

    - **Legacy format** (``SSMPriors``): flat dict with no ``params``
      sub-dict — ``{"mu": 0.0, "sigma": 1.0}`` (assumed Normal) or
      ``{"family": <int_idx>, "concentration": ..., "rate": ...}``
      (positive families). ``family`` is the integer index from
      :mod:`nof1_causal_lab.distributions` runtime registries.

    The two formats coexist for historical reasons (linear-path
    ``SSMPriors`` predates the composite path's dict-config layer).
    This unified materialiser dispatches on the presence of ``params``;
    legacy callers go through here too via :func:`_make_legacy_prior_dist`.
    """
    if "params" not in prior_cfg and (
        "mu" in prior_cfg or "lower" in prior_cfg or "upper" in prior_cfg
        or "concentration" in prior_cfg or ("family" in prior_cfg and isinstance(prior_cfg["family"], int))
        or "sigma" in prior_cfg
    ):
        return _materialize_legacy_prior(prior_cfg)

    family_raw = prior_cfg.get("family", "Normal")
    family = PriorDistributionFamily(family_raw)
    params = prior_cfg.get("params", {})
    shape = tuple(prior_cfg.get("shape", ()))

    def _bcast(scalar: float, shape: tuple[int, ...]):
        if not shape:
            return jnp.asarray(scalar)
        return jnp.broadcast_to(jnp.asarray(scalar), shape)

    if family is PriorDistributionFamily.NORMAL:
        return ndist.Normal(_bcast(params.get("mu", 0.0), shape),
                            _bcast(params.get("sigma", 1.0), shape))
    if family is PriorDistributionFamily.HALF_NORMAL:
        return ndist.HalfNormal(_bcast(params.get("sigma", 1.0), shape))
    if family is PriorDistributionFamily.TRUNCATED_NORMAL:
        return ndist.TruncatedNormal(
            loc=_bcast(params.get("mu", 0.0), shape),
            scale=_bcast(params.get("sigma", 1.0), shape),
            low=params.get("lower", -float("inf")),
            high=params.get("upper", float("inf")),
        )
    if family is PriorDistributionFamily.LOG_NORMAL:
        return ndist.LogNormal(_bcast(params.get("mu", 0.0), shape),
                               _bcast(params.get("sigma", 1.0), shape))
    if family is PriorDistributionFamily.GAMMA:
        return ndist.Gamma(_bcast(params.get("concentration", 2.0), shape),
                           _bcast(params.get("rate", 1.0), shape))
    if family is PriorDistributionFamily.EXPONENTIAL:
        return ndist.Exponential(_bcast(params.get("rate", 1.0), shape))
    if family is PriorDistributionFamily.BETA:
        return ndist.Beta(_bcast(params.get("alpha", 2.0), shape),
                          _bcast(params.get("beta", 2.0), shape))
    if family is PriorDistributionFamily.UNIFORM:
        return ndist.Uniform(_bcast(params.get("lower", 0.0), shape),
                             _bcast(params.get("upper", 1.0), shape))
    if family is PriorDistributionFamily.DELTA:
        return ndist.Delta(_bcast(params.get("value", 0.0), shape))
    raise ValueError(f"Unsupported prior family for composite: {family_raw!r}")


def _spec_from_component_dict(component: dict[str, Any]) -> Any:
    """Build a component spec from a dict-config.

    Priors are stored on the spec as the original dict-configs (not
    materialised). Materialisation to ``numpyro.Distribution`` happens
    lazily inside ``sample_params`` via the polymorphic ``_resolve_prior``
    helper — this lets ``composite_spec_to_dict`` round-trip the spec
    back to JSON without needing an inverse of ``materialize_prior``.
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
        from .compilation import StructuralDenseLinearSpec

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
            bare_site_names=bool(component.get("bare_site_names", True)),
        )
    if kind == "StructuralIntercept":
        from .compilation import StructuralInterceptSpec

        return StructuralInterceptSpec(
            n_latent=int(component["n_latent"]),
            cint_mask=np.asarray(component["cint_mask"], dtype=bool),
            cint_template=jnp.asarray(component["cint_template"]),
            cint_prior=_optional("cint"),
            bare_site_names=bool(component.get("bare_site_names", True)),
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


def compile_composite_from_dict(
    config: dict[str, Any], *, prefix: str = "vf"
):
    """One-shot: dict-config → ``CompiledComposite`` ready for inference.

    Convenience wrapper around ``composite_spec_from_dict`` +
    ``compile_composite``. Stage 4 emits the dict; inference consumes
    the compiled object.
    """
    spec = composite_spec_from_dict(config)
    return compile_composite(spec, prefix=prefix)
