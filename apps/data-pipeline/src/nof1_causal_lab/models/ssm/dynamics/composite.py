"""Spec → ``CompositeVectorField`` compiler.

Bridges between a *declarative* description of the SSM dynamics (the
kind of structure Stage 4 will eventually emit from the LLM) and the
runtime ``CompositeVectorField`` + ``args.params`` tuple that the
simulator, root-finder, and (eventually) Corenflos auxiliary samplers
consume.

Design:

- ``ComponentSpec`` is a Protocol. Each concrete spec carries
  *structure* (source / target indices, latent dimensionality
  expectation) and the *priors* on its parameters.
- ``compile_composite(spec)`` walks the components, builds the
  ``CompositeVectorField``, and returns a ``CompiledComposite`` whose
  ``sample_params`` is a NumPyro-callable function: invoked inside a
  ``numpyro`` model context, it draws every parameter and packs them
  into the per-component tuple shape that the vector field expects in
  ``args.params``.

Concrete specs cover every primitive currently in the library:
``DenseLinearSpec``, ``DiagonalDecaySpec``, ``InterceptSpec``,
``LinearEdgeSpec``, ``HillEdgeSpec``, ``MultiplicativeEdgeSpec``. Adding
a new primitive is: a new ``DriftComponent`` (already in ``edges.py``)
plus a new ``ComponentSpec`` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpyro

from nof1_causal_lab.models.ssm.priors import resolve_prior_distribution
from nof1_causal_lab.models.ssm.structure.sites import (
    PriorAuthoringTransform,
    SemanticBinding,
    SiteKind,
    SupportClass,
    make_site,
)

from .edges import (
    DenseLinear,
    DiagonalDecay,
    HillEdge,
    Intercept,
    LinearEdge,
    MultiplicativeEdge,
)
from .vector_field import CompositeVectorField

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import jax.numpy as jnp
    import numpy as np
    from jax import Array

    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

    from .edges import DriftComponent


@runtime_checkable
class ComponentSpec(Protocol):
    """Declarative description of one drift component plus its priors."""

    def build(self) -> DriftComponent: ...

    def iter_sites(self, prefix: str, *, n_latent: int | None = None) -> Iterator[SiteDescriptor]:
        """Yield canonical sample-site descriptors for this component."""
        ...

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        """Yield component-owned semantic bindings for prior compilation."""
        ...

    def with_runtime_priors(self, prior_fn, *, prefix: str):
        """Return this component with site-keyed runtime priors bound."""
        ...

    def sample_params(self, prefix: str) -> dict[str, Array]:
        """Call inside a NumPyro model; returns this component's param slice."""
        ...


def _require_n_latent(component_name: str, n_latent: int | None) -> int:
    if n_latent is None:
        raise ValueError(f"{component_name}.iter_sites requires n_latent.")
    return int(n_latent)


# ---------------------------------------------------------------------------
# Full-vector component specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DenseLinearSpec:
    """``DenseLinear`` component spec. Priors over ``A`` (``(n, n)``) and
    optionally ``c`` (``(n,)``).

    Priors may be supplied as a ``numpyro.Distribution`` or as a dict-config
    ``{"family": "...", "params": {...}, "shape": [...]}``.
    """

    drift_prior: Any
    cint_prior: Any = None

    def build(self) -> DenseLinear:
        return DenseLinear()

    def drift_site_name(self, prefix: str) -> str:
        return f"{prefix}_drift"

    def cint_site_name(self, prefix: str) -> str:
        return f"{prefix}_cint"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,
    ) -> Iterator[SiteDescriptor]:
        n = _require_n_latent(type(self).__name__, n_latent)
        yield make_site(
            self.drift_site_name(prefix),
            (n, n),
            SupportClass.REAL,
            "dynamics",
            SiteKind.DENSE_DRIFT,
            priors_field="dense_drift",
        )
        if self.cint_prior is not None:
            yield make_site(
                self.cint_site_name(prefix),
                (n,),
                SupportClass.REAL,
                "dynamics",
                SiteKind.DYNAMICS_CINT,
                priors_field="dynamics_cint",
            )

    def iter_semantic_bindings(
        self,
        prefix: str,  # noqa: ARG002
        *,
        latent_names: tuple[str, ...],  # noqa: ARG002
        component_index: int,  # noqa: ARG002
    ) -> Iterator[SemanticBinding]:
        return iter(())

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> DenseLinearSpec:
        return replace(
            self,
            drift_prior=prior_fn(self.drift_site_name(prefix)),
            cint_prior=(
                prior_fn(self.cint_site_name(prefix)) if self.cint_prior is not None else None
            ),
        )

    def sample_params(self, prefix: str) -> dict[str, Array]:
        drift_dist = resolve_prior_distribution(self.drift_prior)
        cint_dist = resolve_prior_distribution(self.cint_prior)
        out: dict[str, Array] = {
            "drift": numpyro.sample(self.drift_site_name(prefix), drift_dist),
        }
        if cint_dist is not None:
            out["cint"] = numpyro.sample(self.cint_site_name(prefix), cint_dist)
        return out


@dataclass(frozen=True, eq=False)
class StructuralDenseLinearSpec:
    """Dense linear drift with structural sparsity + stability-by-construction.

    Self-contained: carries the structural masks, template, stability margin,
    and time-invariant mask directly. The dynamics framework owns the
    linear drift sampling end-to-end; ``SSMParameterLayout`` only derives
    site positions and sizes.

    With this component a ``CompositeSpec`` expresses a linear SSM as a
    single-component composite. Pair with
    :class:`StructuralInterceptSpec` when a constant intercept is needed.

    """

    n_latent: int
    drift_diag_mask: np.ndarray
    drift_offdiag_mask: np.ndarray
    drift_template: jnp.ndarray
    stability_margin: float = 0.05
    time_invariant_mask: np.ndarray | None = None
    base_decay_prior: Any = None
    offdiag_prior: Any = None

    @property
    def drift_base_decay_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import (
            drift_base_decay_positions as _positions,
        )

        return _positions(self.drift_diag_mask, self.n_latent)

    @property
    def offdiag_positions(self) -> list[tuple[int, int]]:
        from nof1_causal_lab.models.ssm.structure.assembly import (
            drift_offdiag_positions as _positions,
        )

        return _positions(self.drift_offdiag_mask, self.n_latent)

    @property
    def n_drift_base_decay(self) -> int:
        return len(self.drift_base_decay_positions)

    @property
    def n_drift_offdiag(self) -> int:
        return len(self.offdiag_positions)

    def base_decay_site_name(self, prefix: str) -> str:
        return f"{prefix}_base_decay"

    def offdiag_site_name(self, prefix: str) -> str:
        return f"{prefix}_offdiag"

    def drift_deterministic_name(self, prefix: str) -> str:
        return f"{prefix}_drift"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,  # noqa: ARG002 - structural spec owns n_latent
    ) -> Iterator[SiteDescriptor]:
        if self.n_drift_base_decay > 0:
            yield make_site(
                self.base_decay_site_name(prefix),
                (self.n_drift_base_decay,),
                SupportClass.POSITIVE,
                "drift",
                SiteKind.DRIFT_BASE_DECAY,
                positions=tuple(self.drift_base_decay_positions),
                deterministic_name=self.drift_deterministic_name(prefix),
                fixed_spec_field="drift",
                priors_field="drift_base_decay",
            )
        if self.n_drift_offdiag > 0:
            yield make_site(
                self.offdiag_site_name(prefix),
                (self.n_drift_offdiag,),
                SupportClass.REAL,
                "drift",
                SiteKind.DRIFT_OFFDIAG,
                positions=tuple(self.offdiag_positions),
                deterministic_name=self.drift_deterministic_name(prefix),
                fixed_spec_field="drift",
                priors_field="drift_offdiag",
            )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        base_decay_site_name = self.base_decay_site_name(prefix)
        for flat_index, latent_idx in enumerate(self.drift_base_decay_positions):
            latent_name = latent_names[latent_idx]
            yield SemanticBinding(
                parameter_name=f"rho_{latent_name}",
                site_name=base_decay_site_name,
                flat_index=flat_index,
                site_kind=SiteKind.DRIFT_BASE_DECAY,
                transform=PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
                prior_field="drift_base_decay",
                construct_names=(latent_name,),
                component_index=component_index,
            )

        offdiag_site_name = self.offdiag_site_name(prefix)
        for flat_index, (effect_idx, cause_idx) in enumerate(self.offdiag_positions):
            cause_name = latent_names[cause_idx]
            effect_name = latent_names[effect_idx]
            yield SemanticBinding(
                parameter_name=f"beta_{cause_name}_{effect_name}",
                site_name=offdiag_site_name,
                flat_index=flat_index,
                site_kind=SiteKind.DRIFT_OFFDIAG,
                transform=PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
                prior_field="drift_offdiag",
                construct_names=(cause_name, effect_name),
                component_index=component_index,
                effect_idx=effect_idx,
                cause_idx=cause_idx,
            )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> StructuralDenseLinearSpec:
        return replace(
            self,
            base_decay_prior=(
                prior_fn(self.base_decay_site_name(prefix))
                if self.n_drift_base_decay > 0
                else self.base_decay_prior
            ),
            offdiag_prior=(
                prior_fn(self.offdiag_site_name(prefix))
                if self.n_drift_offdiag > 0
                else self.offdiag_prior
            ),
        )

    def assemble_drift(
        self,
        base_decay_free: jnp.ndarray | None,
        offdiag_free: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """Stability-by-construction drift assembly.

        Delegates to the shared :func:`assemble_dense_linear_drift` in
        ``structure.assembly`` — single canonical implementation shared
        with :class:`SSMParameterLayout`.
        """
        from nof1_causal_lab.models.ssm.structure.assembly import (
            assemble_dense_linear_drift,
        )

        return assemble_dense_linear_drift(
            drift_template=self.drift_template,
            base_decay_positions=self.drift_base_decay_positions,
            offdiag_positions_list=self.offdiag_positions,
            base_decay_free=base_decay_free,
            offdiag_free=offdiag_free,
            stability_margin=self.stability_margin,
            time_invariant_mask=self.time_invariant_mask,
        )

    def build(self) -> DenseLinear:
        return DenseLinear()

    def sample_params(self, prefix: str) -> dict[str, Array]:
        if self.n_drift_base_decay > 0:
            base_decay_dist = resolve_prior_distribution(self.base_decay_prior)
            if base_decay_dist is None:
                raise ValueError(
                    "StructuralDenseLinearSpec requires base_decay_prior when "
                    f"n_drift_base_decay={self.n_drift_base_decay} > 0."
                )
            base_decay_free = numpyro.sample(self.base_decay_site_name(prefix), base_decay_dist)
        else:
            base_decay_free = None

        if self.n_drift_offdiag > 0:
            offdiag_dist = resolve_prior_distribution(self.offdiag_prior)
            if offdiag_dist is None:
                raise ValueError(
                    "StructuralDenseLinearSpec requires offdiag_prior when "
                    f"n_drift_offdiag={self.n_drift_offdiag} > 0."
                )
            offdiag_free = numpyro.sample(self.offdiag_site_name(prefix), offdiag_dist)
        else:
            offdiag_free = None

        drift = self.assemble_drift(base_decay_free, offdiag_free)
        numpyro.deterministic(self.drift_deterministic_name(prefix), drift)

        return {"drift": drift}


@dataclass(frozen=True, eq=False)
class StructuralInterceptSpec:
    """Sparse-element-sampled constant intercept ``c`` with structural mask.

    Self-contained sibling of :class:`StructuralDenseLinearSpec`. Ports the
    sparse intercept assembly algorithm inline.
    """

    n_latent: int
    cint_mask: np.ndarray
    cint_template: jnp.ndarray
    cint_prior: Any = None

    @property
    def cint_free_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import (
            cint_free_positions as _positions,
        )

        return _positions(self.cint_mask, self.n_latent)

    @property
    def n_cint(self) -> int:
        return len(self.cint_free_positions)

    def cint_site_name(self, prefix: str) -> str:
        return f"{prefix}_cint"

    def cint_deterministic_name(self, prefix: str) -> str:
        return f"{prefix}_cint_full"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,  # noqa: ARG002 - structural spec owns n_latent
    ) -> Iterator[SiteDescriptor]:
        if self.n_cint > 0:
            yield make_site(
                self.cint_site_name(prefix),
                (self.n_cint,),
                SupportClass.REAL,
                "cint",
                SiteKind.CINT,
                positions=tuple(self.cint_free_positions),
                deterministic_name=self.cint_deterministic_name(prefix),
                fixed_spec_field="cint",
                priors_field="cint",
            )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        site_name = self.cint_site_name(prefix)
        for flat_index, latent_idx in enumerate(self.cint_free_positions):
            latent_name = latent_names[latent_idx]
            yield SemanticBinding(
                parameter_name=f"cint_{latent_name}",
                site_name=site_name,
                flat_index=flat_index,
                site_kind=SiteKind.CINT,
                prior_field="cint",
                construct_names=(latent_name,),
                component_index=component_index,
            )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> StructuralInterceptSpec:
        return replace(
            self,
            cint_prior=prior_fn(self.cint_site_name(prefix))
            if self.n_cint > 0
            else self.cint_prior,
        )

    def assemble_cint(self, cint_free: jnp.ndarray | None) -> jnp.ndarray:
        """Sparse-element cint assembly.

        Delegates to the shared :func:`assemble_intercept_cint` in
        ``structure.assembly`` — single canonical implementation shared
        with :class:`SSMParameterLayout`.
        """
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_intercept_cint

        return assemble_intercept_cint(
            cint_template=self.cint_template,
            free_positions=self.cint_free_positions,
            cint_free=cint_free,
        )

    def build(self) -> Intercept:
        return Intercept()

    def sample_params(self, prefix: str) -> dict[str, Array]:
        if self.n_cint > 0:
            cint_dist = resolve_prior_distribution(self.cint_prior)
            if cint_dist is None:
                raise ValueError(
                    f"StructuralInterceptSpec requires cint_prior when n_cint={self.n_cint} > 0."
                )
            cint_free = numpyro.sample(self.cint_site_name(prefix), cint_dist)
        else:
            cint_free = None

        cint = self.assemble_cint(cint_free)
        numpyro.deterministic(self.cint_deterministic_name(prefix), cint)

        return {"cint": cint}


@dataclass(frozen=True, eq=False)
class DiagonalDecaySpec:
    """``DiagonalDecay`` component. Prior over ``(n_latent,)`` rate vector
    (must be positive)."""

    decay_prior: Any

    def build(self) -> DiagonalDecay:
        return DiagonalDecay()

    def decay_site_name(self, prefix: str) -> str:
        return f"{prefix}_decay"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,
    ) -> Iterator[SiteDescriptor]:
        n = _require_n_latent(type(self).__name__, n_latent)
        yield make_site(
            self.decay_site_name(prefix),
            (n,),
            SupportClass.POSITIVE,
            "dynamics",
            SiteKind.DYNAMICS_DECAY,
            priors_field="dynamics_decay",
        )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        site_name = self.decay_site_name(prefix)
        for flat_index, latent_name in enumerate(latent_names):
            yield SemanticBinding(
                parameter_name=f"rho_{latent_name}",
                site_name=site_name,
                flat_index=flat_index,
                site_kind=SiteKind.DYNAMICS_DECAY,
                transform=PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
                prior_field="dynamics_decay",
                construct_names=(latent_name,),
                component_index=component_index,
            )
            yield SemanticBinding(
                parameter_name=f"decay_{latent_name}",
                site_name=site_name,
                flat_index=flat_index,
                site_kind=SiteKind.DYNAMICS_DECAY,
                transform=PriorAuthoringTransform.POSITIVE_IDENTITY,
                prior_field="dynamics_decay",
                construct_names=(latent_name,),
                component_index=component_index,
            )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> DiagonalDecaySpec:
        return replace(self, decay_prior=prior_fn(self.decay_site_name(prefix)))

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {
            "decay": numpyro.sample(
                self.decay_site_name(prefix),
                resolve_prior_distribution(self.decay_prior),
            )
        }


@dataclass(frozen=True, eq=False)
class InterceptSpec:
    """``Intercept`` component. Prior over ``(n_latent,)`` intercept vector."""

    cint_prior: Any

    def build(self) -> Intercept:
        return Intercept()

    def cint_site_name(self, prefix: str) -> str:
        return f"{prefix}_cint"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,
    ) -> Iterator[SiteDescriptor]:
        n = _require_n_latent(type(self).__name__, n_latent)
        yield make_site(
            self.cint_site_name(prefix),
            (n,),
            SupportClass.REAL,
            "dynamics",
            SiteKind.DYNAMICS_CINT,
            priors_field="dynamics_cint",
        )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        site_name = self.cint_site_name(prefix)
        for flat_index, latent_name in enumerate(latent_names):
            yield SemanticBinding(
                parameter_name=f"cint_{latent_name}",
                site_name=site_name,
                flat_index=flat_index,
                site_kind=SiteKind.DYNAMICS_CINT,
                prior_field="dynamics_cint",
                construct_names=(latent_name,),
                component_index=component_index,
            )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> InterceptSpec:
        return replace(self, cint_prior=prior_fn(self.cint_site_name(prefix)))

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {
            "cint": numpyro.sample(
                self.cint_site_name(prefix),
                resolve_prior_distribution(self.cint_prior),
            )
        }


# ---------------------------------------------------------------------------
# Single-target edge specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LinearEdgeSpec:
    """``LinearEdge`` component spec. Prior over scalar ``weight``."""

    source: int
    target: int
    weight_prior: Any

    def build(self) -> LinearEdge:
        return LinearEdge(source=self.source, target=self.target)

    def weight_site_name(self, prefix: str) -> str:
        return f"{prefix}_weight"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,  # noqa: ARG002 - scalar edge parameter
    ) -> Iterator[SiteDescriptor]:
        yield make_site(
            self.weight_site_name(prefix),
            (),
            SupportClass.REAL,
            "dynamics",
            SiteKind.DYNAMICS_WEIGHT,
            positions=((self.source, self.target),),
            priors_field="linear_edge_weight",
        )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        cause_name = latent_names[self.source]
        effect_name = latent_names[self.target]
        site_name = self.weight_site_name(prefix)
        yield SemanticBinding(
            parameter_name=f"beta_{cause_name}_{effect_name}",
            site_name=site_name,
            flat_index=0,
            site_kind=SiteKind.DYNAMICS_WEIGHT,
            transform=PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
            prior_field="linear_edge_weight",
            construct_names=(cause_name, effect_name),
            component_index=component_index,
            effect_idx=self.target,
            cause_idx=self.source,
        )
        yield SemanticBinding(
            parameter_name=f"linear_weight_{cause_name}_{effect_name}",
            site_name=site_name,
            flat_index=0,
            site_kind=SiteKind.DYNAMICS_WEIGHT,
            prior_field="linear_edge_weight",
            construct_names=(cause_name, effect_name),
            component_index=component_index,
            effect_idx=self.target,
            cause_idx=self.source,
        )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> LinearEdgeSpec:
        return replace(self, weight_prior=prior_fn(self.weight_site_name(prefix)))

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {
            "weight": numpyro.sample(
                self.weight_site_name(prefix),
                resolve_prior_distribution(self.weight_prior),
            )
        }


@dataclass(frozen=True, eq=False)
class HillEdgeSpec:
    """``HillEdge`` component spec. Priors over ``Emax``, ``EC50``, ``n``.

    All three should be over positive supports — typical choices are
    ``LogNormal`` for ``Emax`` and ``EC50``, ``TruncatedNormal`` for
    ``n`` (Hill coefficient, biologically ≥ 1, rarely > 4).
    """

    source: int
    target: int
    emax_prior: Any
    ec50_prior: Any
    n_prior: Any

    def build(self) -> HillEdge:
        return HillEdge(source=self.source, target=self.target)

    def emax_site_name(self, prefix: str) -> str:
        return f"{prefix}_Emax"

    def ec50_site_name(self, prefix: str) -> str:
        return f"{prefix}_EC50"

    def n_site_name(self, prefix: str) -> str:
        return f"{prefix}_n"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,  # noqa: ARG002 - scalar edge parameters
    ) -> Iterator[SiteDescriptor]:
        positions = ((self.source, self.target),)
        yield make_site(
            self.emax_site_name(prefix),
            (),
            SupportClass.POSITIVE,
            "dynamics",
            SiteKind.HILL_EMAX,
            positions=positions,
            priors_field="hill_emax",
        )
        yield make_site(
            self.ec50_site_name(prefix),
            (),
            SupportClass.POSITIVE,
            "dynamics",
            SiteKind.HILL_EC50,
            positions=positions,
            priors_field="hill_ec50",
        )
        yield make_site(
            self.n_site_name(prefix),
            (),
            SupportClass.REAL,
            "dynamics",
            SiteKind.HILL_N,
            positions=positions,
            priors_field="hill_n",
        )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        cause_name = latent_names[self.source]
        effect_name = latent_names[self.target]
        for parameter_prefix, site_name, site_kind, transform in (
            (
                "hill_emax",
                self.emax_site_name(prefix),
                SiteKind.HILL_EMAX,
                PriorAuthoringTransform.POSITIVE_IDENTITY,
            ),
            (
                "hill_ec50",
                self.ec50_site_name(prefix),
                SiteKind.HILL_EC50,
                PriorAuthoringTransform.POSITIVE_IDENTITY,
            ),
            (
                "hill_n",
                self.n_site_name(prefix),
                SiteKind.HILL_N,
                PriorAuthoringTransform.IDENTITY,
            ),
        ):
            yield SemanticBinding(
                parameter_name=f"{parameter_prefix}_{cause_name}_{effect_name}",
                site_name=site_name,
                flat_index=0,
                site_kind=site_kind,
                transform=transform,
                prior_field=parameter_prefix,
                construct_names=(cause_name, effect_name),
                component_index=component_index,
                effect_idx=self.target,
                cause_idx=self.source,
            )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> HillEdgeSpec:
        return replace(
            self,
            emax_prior=prior_fn(self.emax_site_name(prefix)),
            ec50_prior=prior_fn(self.ec50_site_name(prefix)),
            n_prior=prior_fn(self.n_site_name(prefix)),
        )

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {
            "Emax": numpyro.sample(
                self.emax_site_name(prefix),
                resolve_prior_distribution(self.emax_prior),
            ),
            "EC50": numpyro.sample(
                self.ec50_site_name(prefix),
                resolve_prior_distribution(self.ec50_prior),
            ),
            "n": numpyro.sample(
                self.n_site_name(prefix),
                resolve_prior_distribution(self.n_prior),
            ),
        }


@dataclass(frozen=True, eq=False)
class MultiplicativeEdgeSpec:
    """``MultiplicativeEdge`` component spec. Prior over scalar ``weight``."""

    source_a: int
    source_b: int
    target: int
    weight_prior: Any

    def build(self) -> MultiplicativeEdge:
        return MultiplicativeEdge(
            source_a=self.source_a, source_b=self.source_b, target=self.target
        )

    def weight_site_name(self, prefix: str) -> str:
        return f"{prefix}_weight"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,  # noqa: ARG002 - scalar edge parameter
    ) -> Iterator[SiteDescriptor]:
        yield make_site(
            self.weight_site_name(prefix),
            (),
            SupportClass.REAL,
            "dynamics",
            SiteKind.DYNAMICS_WEIGHT,
            positions=((self.source_a, self.source_b, self.target),),
            priors_field="multiplicative_weight",
        )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        source_a = latent_names[self.source_a]
        source_b = latent_names[self.source_b]
        target = latent_names[self.target]
        yield SemanticBinding(
            parameter_name=f"multiplicative_weight_{source_a}_{source_b}_{target}",
            site_name=self.weight_site_name(prefix),
            flat_index=0,
            site_kind=SiteKind.DYNAMICS_WEIGHT,
            prior_field="multiplicative_weight",
            construct_names=(source_a, source_b, target),
            component_index=component_index,
            effect_idx=self.target,
            cause_idx=self.source_a,
        )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> MultiplicativeEdgeSpec:
        return replace(self, weight_prior=prior_fn(self.weight_site_name(prefix)))

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {
            "weight": numpyro.sample(
                self.weight_site_name(prefix),
                resolve_prior_distribution(self.weight_prior),
            )
        }


# ---------------------------------------------------------------------------
# Composite spec + compiler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CompositeSpec:
    """Declarative SSM dynamics: latent dimension + tuple of component specs."""

    n_latent: int
    components: tuple[ComponentSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True, eq=False)
class CompiledComposite:
    """Output of ``compile_composite``: ready-to-fit vector field plus a
    NumPyro-callable that produces the matching ``args.params`` tuple."""

    spec: CompositeSpec
    vector_field: CompositeVectorField
    sample_params: Callable[[], tuple[dict[str, Array], ...]]
    site_registry: tuple[SiteDescriptor, ...]
    site_prefix: str = "vf"


def compile_composite(spec: CompositeSpec, *, prefix: str = "vf") -> CompiledComposite:
    """Compile a ``CompositeSpec`` into a ``CompositeVectorField`` and a
    NumPyro-callable parameter sampler.

    The ``prefix`` is prepended to every NumPyro sample site name so
    nested compositions or multiple SSMs in one model stay disambiguated.
    """
    components = tuple(component_spec.build() for component_spec in spec.components)
    vector_field = CompositeVectorField(n_latent=spec.n_latent, components=components)
    component_specs = spec.components
    site_registry = tuple(
        site
        for i, component_spec in enumerate(component_specs)
        for site in component_spec.iter_sites(
            prefix=f"{prefix}_{i}",
            n_latent=spec.n_latent,
        )
    )

    def _sample_all_params() -> tuple[dict[str, Array], ...]:
        return tuple(
            component_spec.sample_params(prefix=f"{prefix}_{i}")
            for i, component_spec in enumerate(component_specs)
        )

    return CompiledComposite(
        spec=spec,
        vector_field=vector_field,
        sample_params=_sample_all_params,
        site_registry=site_registry,
        site_prefix=prefix,
    )


def iter_component_semantic_bindings(
    spec: CompositeSpec,
    *,
    latent_names: tuple[str, ...],
    prefix: str = "vf",
) -> Iterator[SemanticBinding]:
    """Yield semantic prior bindings owned by drift component specs."""
    for component_index, component_spec in enumerate(spec.components):
        yield from component_spec.iter_semantic_bindings(
            prefix=f"{prefix}_{component_index}",
            latent_names=latent_names,
            component_index=component_index,
        )


def pack_component_params_from_samples(
    spec: CompositeSpec,
    samples: dict[str, Array],
    deterministics: dict[str, Array],
    *,
    prefix: str = "vf",
) -> tuple[dict[str, Array], ...]:
    """Pack flat sample-site values into the vector-field param tuple.

    This mirrors ``CompiledComposite.sample_params`` without entering a NumPyro
    sampling context, so post-fit likelihood evaluators can rebuild the same
    runtime dynamics object from constrained parameter draws.
    """
    packed: list[dict[str, Array]] = []
    for idx, component_spec in enumerate(spec.components):
        site_prefix = f"{prefix}_{idx}"
        if isinstance(component_spec, StructuralDenseLinearSpec):
            drift_name = component_spec.drift_deterministic_name(site_prefix)
            drift = deterministics.get(drift_name)
            if drift is None:
                drift = component_spec.assemble_drift(
                    samples.get(component_spec.base_decay_site_name(site_prefix)),
                    samples.get(component_spec.offdiag_site_name(site_prefix)),
                )
            packed.append({"drift": drift})
        elif isinstance(component_spec, StructuralInterceptSpec):
            cint_name = component_spec.cint_deterministic_name(site_prefix)
            cint = deterministics.get(cint_name)
            if cint is None:
                cint = component_spec.assemble_cint(
                    samples.get(component_spec.cint_site_name(site_prefix))
                )
            packed.append({"cint": cint})
        elif isinstance(component_spec, DenseLinearSpec):
            params = {"drift": samples[component_spec.drift_site_name(site_prefix)]}
            cint_name = component_spec.cint_site_name(site_prefix)
            if cint_name in samples:
                params["cint"] = samples[cint_name]
            packed.append(params)
        elif isinstance(component_spec, DiagonalDecaySpec):
            packed.append({"decay": samples[component_spec.decay_site_name(site_prefix)]})
        elif isinstance(component_spec, InterceptSpec):
            packed.append({"cint": samples[component_spec.cint_site_name(site_prefix)]})
        elif isinstance(component_spec, LinearEdgeSpec):
            packed.append({"weight": samples[component_spec.weight_site_name(site_prefix)]})
        elif isinstance(component_spec, HillEdgeSpec):
            packed.append(
                {
                    "Emax": samples[component_spec.emax_site_name(site_prefix)],
                    "EC50": samples[component_spec.ec50_site_name(site_prefix)],
                    "n": samples[component_spec.n_site_name(site_prefix)],
                }
            )
        elif isinstance(component_spec, MultiplicativeEdgeSpec):
            packed.append({"weight": samples[component_spec.weight_site_name(site_prefix)]})
        else:
            raise TypeError(
                f"Unsupported drift component spec for parameter packing: "
                f"{type(component_spec).__name__}"
            )
    return tuple(packed)
