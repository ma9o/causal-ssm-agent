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
``StateDecaySpec``, ``DiagonalDecaySpec``, ``StateInterceptSpec``,
``InterceptSpec``, ``LinearEdgeSpec``, ``HillEdgeSpec``,
``MultiplicativeEdgeSpec``. Adding
a new primitive is: a new ``VectorFieldComponent`` (already in ``edges.py``)
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
    DiagonalDecay,
    HillEdge,
    Intercept,
    LinearEdge,
    MultiplicativeEdge,
    StateDecay,
    StateIntercept,
)
from .vector_field import CompositeVectorField

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from jax import Array

    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

    from .edges import VectorFieldComponent


@runtime_checkable
class ComponentSpec(Protocol):
    """Declarative description of one vector-field component plus its priors."""

    def build(self) -> VectorFieldComponent: ...

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
# Scalar target-owned component specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StateDecaySpec:
    """Single-latent relaxation component with one positive decay parameter."""

    target: int
    decay_prior: Any

    def build(self) -> StateDecay:
        return StateDecay(target=self.target)

    def decay_site_name(self, prefix: str) -> str:
        return f"{prefix}_decay"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,  # noqa: ARG002 - scalar target parameter
    ) -> Iterator[SiteDescriptor]:
        yield make_site(
            self.decay_site_name(prefix),
            (),
            SupportClass.POSITIVE,
            "dynamics",
            SiteKind.DYNAMICS_DECAY,
            positions=(self.target,),
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
        latent_name = latent_names[self.target]
        yield SemanticBinding(
            parameter_name=f"rho_{latent_name}",
            site_name=site_name,
            flat_index=0,
            site_kind=SiteKind.DYNAMICS_DECAY,
            transform=PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
            prior_field="dynamics_decay",
            construct_names=(latent_name,),
            component_index=component_index,
        )
        yield SemanticBinding(
            parameter_name=f"decay_{latent_name}",
            site_name=site_name,
            flat_index=0,
            site_kind=SiteKind.DYNAMICS_DECAY,
            transform=PriorAuthoringTransform.POSITIVE_IDENTITY,
            prior_field="dynamics_decay",
            construct_names=(latent_name,),
            component_index=component_index,
        )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> StateDecaySpec:
        return replace(self, decay_prior=prior_fn(self.decay_site_name(prefix)))

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {
            "decay": numpyro.sample(
                self.decay_site_name(prefix),
                resolve_prior_distribution(self.decay_prior),
            )
        }


@dataclass(frozen=True, eq=False)
class StateInterceptSpec:
    """Single-latent constant intercept component."""

    target: int
    cint_prior: Any

    def build(self) -> StateIntercept:
        return StateIntercept(target=self.target)

    def cint_site_name(self, prefix: str) -> str:
        return f"{prefix}_cint"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int | None = None,  # noqa: ARG002 - scalar target parameter
    ) -> Iterator[SiteDescriptor]:
        yield make_site(
            self.cint_site_name(prefix),
            (),
            SupportClass.REAL,
            "dynamics",
            SiteKind.DYNAMICS_CINT,
            positions=(self.target,),
            priors_field="dynamics_cint",
        )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        latent_name = latent_names[self.target]
        yield SemanticBinding(
            parameter_name=f"cint_{latent_name}",
            site_name=self.cint_site_name(prefix),
            flat_index=0,
            site_kind=SiteKind.DYNAMICS_CINT,
            prior_field="dynamics_cint",
            construct_names=(latent_name,),
            component_index=component_index,
        )

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> StateInterceptSpec:
        return replace(self, cint_prior=prior_fn(self.cint_site_name(prefix)))

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {
            "cint": numpyro.sample(
                self.cint_site_name(prefix),
                resolve_prior_distribution(self.cint_prior),
            )
        }


# ---------------------------------------------------------------------------
# Full-vector component specs
# ---------------------------------------------------------------------------


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
            positions=((self.target, self.source),),
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
        positions = ((self.target, self.source),)
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
            positions=((self.target, self.source_a, self.source_b),),
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
    """Yield semantic prior bindings owned by vector-field component specs."""
    for component_index, component_spec in enumerate(spec.components):
        yield from component_spec.iter_semantic_bindings(
            prefix=f"{prefix}_{component_index}",
            latent_names=latent_names,
            component_index=component_index,
        )


def pack_component_params_from_samples(
    spec: CompositeSpec,
    samples: dict[str, Array],
    deterministics: dict[str, Array],  # noqa: ARG001 - retained for call-site uniformity
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
        if isinstance(component_spec, StateDecaySpec):
            packed.append({"decay": samples[component_spec.decay_site_name(site_prefix)]})
        elif isinstance(component_spec, StateInterceptSpec):
            packed.append({"cint": samples[component_spec.cint_site_name(site_prefix)]})
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
                f"Unsupported vector-field component spec for parameter packing: "
                f"{type(component_spec).__name__}"
            )
    return tuple(packed)
