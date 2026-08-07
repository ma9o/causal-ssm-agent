"""Spec → ``VectorField`` compiler.

Bridges between a *declarative* description of the SSM dynamics (the
kind of structure model-spec will eventually emit from the LLM) and the
runtime ``VectorField`` + ``args.params`` tuple that the
simulator, root-finder, and (eventually) Corenflos auxiliary samplers
consume.

Design:

- ``ComponentSpec`` is a Protocol. Each concrete spec carries
  *structure* (source / target indices, latent dimensionality
  expectation). Priors are resolved from canonical site-prior metadata
  at sampling time.
- ``compile_dynamics(spec)`` walks the components, builds the
  ``VectorField``, and returns a ``CompiledDynamics`` whose
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp
import numpyro

from nof1_causal_lab.models.ssm.structure.parameters import Fixed, Free, ParameterSlot
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
    NodePotential,
    StateDecay,
    StateIntercept,
)
from .vector_field import VectorField

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpyro.distributions as dist
    from jax import Array

    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

    from .edges import VectorFieldComponent

    PriorFn = Callable[[str], dist.Distribution]


@runtime_checkable
class ComponentSpec(Protocol):
    """Declarative description of one vector-field component."""

    def build(self) -> VectorFieldComponent: ...

    def iter_sites(self, prefix: str, *, n_latent: int) -> Iterator[SiteDescriptor]:
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

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        """Call inside a NumPyro model; returns this component's param slice."""
        ...


def _sample_parameter_slot(
    slot: ParameterSlot,
    site_name: str,
    prior_fn: PriorFn,
) -> Array:
    """Sample a free slot or return a fixed value without adding a trace site."""
    if isinstance(slot, Fixed):
        return jnp.asarray(slot.value)
    return numpyro.sample(site_name, prior_fn(site_name))


def _parameter_slot_from_samples(
    slot: ParameterSlot,
    site_name: str,
    samples: dict[str, Array],
) -> Array:
    """Resolve a parameter slot from constrained posterior samples."""
    if isinstance(slot, Fixed):
        return jnp.asarray(slot.value)
    return samples[site_name]


# ---------------------------------------------------------------------------
# Scalar target-owned component specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StateDecaySpec:
    """Single-latent relaxation component with one positive decay parameter."""

    target: int

    def build(self) -> StateDecay:
        return StateDecay(target=self.target)

    def decay_site_name(self, prefix: str) -> str:
        return f"{prefix}_decay"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int,  # noqa: ARG002 - scalar target parameter
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

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            "decay": numpyro.sample(
                self.decay_site_name(prefix),
                prior_fn(self.decay_site_name(prefix)),
            )
        }


@dataclass(frozen=True, eq=False)
class StateInterceptSpec:
    """Single-latent constant intercept component."""

    target: int

    def build(self) -> StateIntercept:
        return StateIntercept(target=self.target)

    def cint_site_name(self, prefix: str) -> str:
        return f"{prefix}_cint"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int,  # noqa: ARG002 - scalar target parameter
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

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            "cint": numpyro.sample(
                self.cint_site_name(prefix),
                prior_fn(self.cint_site_name(prefix)),
            )
        }


@dataclass(frozen=True, eq=False)
class NodePotentialSpec:
    """Node self-dynamics as a 1-D potential well at ``target``.

    Builds :class:`NodePotential`. Subsumes ``StateDecaySpec`` (``stiffness`` =
    relaxation rate / well curvature) and the set-point role of
    ``StateInterceptSpec`` (``center`` = well minimum), and adds opt-in
    nonlinear self-limitation (``quartic``). Each parameter is an explicit
    :class:`Fixed` or :class:`Free` slot. Free ``stiffness``/``quartic`` slots
    use positive supports. ``quartic`` defaults to ``Fixed(0.0)`` (pure
    quadratic = linear relaxation drift, exactly reproducing ``StateDecay`` +
    ``StateIntercept``).
    """

    target: int
    center: ParameterSlot = field(default_factory=Free)
    stiffness: ParameterSlot = field(default_factory=Free)
    quartic: ParameterSlot = field(default_factory=lambda: Fixed(0.0))

    def __post_init__(self) -> None:
        if isinstance(self.stiffness, Fixed) and self.stiffness.value <= 0.0:
            raise ValueError(f"stiffness must be positive; got {self.stiffness.value}.")
        if isinstance(self.quartic, Fixed) and self.quartic.value < 0.0:
            raise ValueError(f"quartic must be non-negative; got {self.quartic.value}.")

    def build(self) -> NodePotential:
        return NodePotential(target=self.target)

    def center_site_name(self, prefix: str) -> str:
        return f"{prefix}_center"

    def decay_site_name(self, prefix: str) -> str:
        # The relaxation rate (the well curvature / "stiffness") is sampled at the
        # shared decay site, so a quadratic NodePotential is site-identical to a
        # StateDecay. The runtime param key stays "stiffness".
        return f"{prefix}_decay"

    def quartic_site_name(self, prefix: str) -> str:
        return f"{prefix}_quartic"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int,  # noqa: ARG002 - scalar target parameters
    ) -> Iterator[SiteDescriptor]:
        positions = (self.target,)
        if isinstance(self.center, Free):
            yield make_site(
                self.center_site_name(prefix),
                (),
                SupportClass.REAL,
                "dynamics",
                SiteKind.DYNAMICS_POTENTIAL_CENTER,
                positions=positions,
                priors_field="dynamics_potential_center",
            )
        if isinstance(self.stiffness, Free):
            # Stiffness is the relaxation rate, i.e. the decay: reuse the decay
            # site-kind and prior so it shares StateDecay's authoring contract.
            yield make_site(
                self.decay_site_name(prefix),
                (),
                SupportClass.POSITIVE,
                "dynamics",
                SiteKind.DYNAMICS_DECAY,
                positions=positions,
                priors_field="dynamics_decay",
            )
        if isinstance(self.quartic, Free):
            yield make_site(
                self.quartic_site_name(prefix),
                (),
                SupportClass.POSITIVE,
                "dynamics",
                SiteKind.DYNAMICS_POTENTIAL_QUARTIC,
                positions=positions,
                priors_field="dynamics_potential_quartic",
            )

    def iter_semantic_bindings(
        self,
        prefix: str,
        *,
        latent_names: tuple[str, ...],
        component_index: int,
    ) -> Iterator[SemanticBinding]:
        latent_name = latent_names[self.target]
        if isinstance(self.center, Free):
            yield SemanticBinding(
                parameter_name=f"setpoint_{latent_name}",
                site_name=self.center_site_name(prefix),
                flat_index=0,
                site_kind=SiteKind.DYNAMICS_POTENTIAL_CENTER,
                transform=PriorAuthoringTransform.IDENTITY,
                prior_field="dynamics_potential_center",
                construct_names=(latent_name,),
                component_index=component_index,
            )
        if isinstance(self.stiffness, Free):
            # Stiffness is the relaxation rate (the decay): expose it through the
            # same persistence (``rho_``) / decay (``decay_``) authoring contract
            # as StateDecay, so causal-design priors resolve identically.
            stiffness_site = self.decay_site_name(prefix)
            yield SemanticBinding(
                parameter_name=f"rho_{latent_name}",
                site_name=stiffness_site,
                flat_index=0,
                site_kind=SiteKind.DYNAMICS_DECAY,
                transform=PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
                prior_field="dynamics_decay",
                construct_names=(latent_name,),
                component_index=component_index,
            )
            yield SemanticBinding(
                parameter_name=f"decay_{latent_name}",
                site_name=stiffness_site,
                flat_index=0,
                site_kind=SiteKind.DYNAMICS_DECAY,
                transform=PriorAuthoringTransform.POSITIVE_IDENTITY,
                prior_field="dynamics_decay",
                construct_names=(latent_name,),
                component_index=component_index,
            )
        if isinstance(self.quartic, Free):
            yield SemanticBinding(
                parameter_name=f"self_limit_{latent_name}",
                site_name=self.quartic_site_name(prefix),
                flat_index=0,
                site_kind=SiteKind.DYNAMICS_POTENTIAL_QUARTIC,
                transform=PriorAuthoringTransform.POSITIVE_IDENTITY,
                prior_field="dynamics_potential_quartic",
                construct_names=(latent_name,),
                component_index=component_index,
            )

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            "center": _sample_parameter_slot(
                self.center,
                self.center_site_name(prefix),
                prior_fn,
            ),
            "stiffness": _sample_parameter_slot(
                self.stiffness,
                self.decay_site_name(prefix),
                prior_fn,
            ),
            "quartic": _sample_parameter_slot(
                self.quartic,
                self.quartic_site_name(prefix),
                prior_fn,
            ),
        }


# ---------------------------------------------------------------------------
# Full-vector component specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DiagonalDecaySpec:
    """``DiagonalDecay`` component. Prior over ``(n_latent,)`` rate vector
    (must be positive)."""

    def build(self) -> DiagonalDecay:
        return DiagonalDecay()

    def decay_site_name(self, prefix: str) -> str:
        return f"{prefix}_decay"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int,
    ) -> Iterator[SiteDescriptor]:
        n = n_latent
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

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            "decay": numpyro.sample(
                self.decay_site_name(prefix),
                prior_fn(self.decay_site_name(prefix)),
            )
        }


@dataclass(frozen=True, eq=False)
class InterceptSpec:
    """``Intercept`` component. Prior over ``(n_latent,)`` intercept vector."""

    def build(self) -> Intercept:
        return Intercept()

    def cint_site_name(self, prefix: str) -> str:
        return f"{prefix}_cint"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int,
    ) -> Iterator[SiteDescriptor]:
        n = n_latent
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

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            "cint": numpyro.sample(
                self.cint_site_name(prefix),
                prior_fn(self.cint_site_name(prefix)),
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

    def build(self) -> LinearEdge:
        return LinearEdge(source=self.source, target=self.target)

    def weight_site_name(self, prefix: str) -> str:
        return f"{prefix}_weight"

    def iter_sites(
        self,
        prefix: str,
        *,
        n_latent: int,  # noqa: ARG002 - scalar edge parameter
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

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            "weight": numpyro.sample(
                self.weight_site_name(prefix),
                prior_fn(self.weight_site_name(prefix)),
            )
        }


@dataclass(frozen=True, eq=False)
class HillEdgeSpec:
    """``HillEdge`` component spec. Priors over ``Emax``, ``EC50``, ``n``.

    Each parameter is an explicit :class:`Fixed` or :class:`Free` slot. Free
    parameters use positive supports where required — typical choices are
    ``LogNormal`` for ``Emax`` and ``EC50``, ``TruncatedNormal`` for ``n``
    (Hill coefficient, biologically ≥ 1, rarely > 4).
    """

    source: int
    target: int
    emax: ParameterSlot = field(default_factory=Free)
    ec50: ParameterSlot = field(default_factory=Free)
    n: ParameterSlot = field(default_factory=Free)

    def __post_init__(self) -> None:
        for name, slot in (("emax", self.emax), ("ec50", self.ec50), ("n", self.n)):
            if isinstance(slot, Fixed) and slot.value <= 0.0:
                raise ValueError(f"{name} must be positive; got {slot.value}.")

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
        n_latent: int,  # noqa: ARG002 - scalar edge parameters
    ) -> Iterator[SiteDescriptor]:
        positions = ((self.target, self.source),)
        if isinstance(self.emax, Free):
            yield make_site(
                self.emax_site_name(prefix),
                (),
                SupportClass.POSITIVE,
                "dynamics",
                SiteKind.HILL_EMAX,
                positions=positions,
                priors_field="hill_emax",
            )
        if isinstance(self.ec50, Free):
            yield make_site(
                self.ec50_site_name(prefix),
                (),
                SupportClass.POSITIVE,
                "dynamics",
                SiteKind.HILL_EC50,
                positions=positions,
                priors_field="hill_ec50",
            )
        if isinstance(self.n, Free):
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
        for parameter_prefix, site_name, site_kind, transform, slot in (
            (
                "hill_emax",
                self.emax_site_name(prefix),
                SiteKind.HILL_EMAX,
                PriorAuthoringTransform.POSITIVE_IDENTITY,
                self.emax,
            ),
            (
                "hill_ec50",
                self.ec50_site_name(prefix),
                SiteKind.HILL_EC50,
                PriorAuthoringTransform.POSITIVE_IDENTITY,
                self.ec50,
            ),
            (
                "hill_n",
                self.n_site_name(prefix),
                SiteKind.HILL_N,
                PriorAuthoringTransform.IDENTITY,
                self.n,
            ),
        ):
            if isinstance(slot, Fixed):
                continue
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

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            "Emax": _sample_parameter_slot(
                self.emax,
                self.emax_site_name(prefix),
                prior_fn,
            ),
            "EC50": _sample_parameter_slot(
                self.ec50,
                self.ec50_site_name(prefix),
                prior_fn,
            ),
            "n": _sample_parameter_slot(
                self.n,
                self.n_site_name(prefix),
                prior_fn,
            ),
        }


@dataclass(frozen=True, eq=False)
class MultiplicativeEdgeSpec:
    """``MultiplicativeEdge`` component spec. Prior over scalar ``weight``."""

    source_a: int
    source_b: int
    target: int

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
        n_latent: int,  # noqa: ARG002 - scalar edge parameter
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

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            "weight": numpyro.sample(
                self.weight_site_name(prefix),
                prior_fn(self.weight_site_name(prefix)),
            )
        }


# ---------------------------------------------------------------------------
# Dynamics spec + compiler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DynamicsSpec:
    """Declarative SSM dynamics: latent dimension + tuple of component specs."""

    n_latent: int
    components: tuple[ComponentSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True, eq=False)
class CompiledDynamics:
    """Output of ``compile_dynamics``: ready-to-fit vector field plus a
    NumPyro-callable that produces the matching ``args.params`` tuple."""

    spec: DynamicsSpec
    vector_field: VectorField
    sample_params: Callable[[PriorFn], tuple[dict[str, Array], ...]]
    site_registry: tuple[SiteDescriptor, ...]
    site_prefix: str = "vf"


def compile_dynamics(spec: DynamicsSpec, *, prefix: str = "vf") -> CompiledDynamics:
    """Compile a ``DynamicsSpec`` into a ``VectorField`` and a
    NumPyro-callable parameter sampler.

    The ``prefix`` is prepended to every NumPyro sample site name so
    nested compositions or multiple SSMs in one model stay disambiguated.
    """
    components = tuple(component_spec.build() for component_spec in spec.components)
    vector_field = VectorField(n_latent=spec.n_latent, components=components)
    component_specs = spec.components
    site_registry = tuple(
        site
        for i, component_spec in enumerate(component_specs)
        for site in component_spec.iter_sites(
            prefix=f"{prefix}_{i}",
            n_latent=spec.n_latent,
        )
    )

    def _sample_all_params(prior_fn: PriorFn) -> tuple[dict[str, Array], ...]:
        return tuple(
            component_spec.sample_params(prefix=f"{prefix}_{i}", prior_fn=prior_fn)
            for i, component_spec in enumerate(component_specs)
        )

    return CompiledDynamics(
        spec=spec,
        vector_field=vector_field,
        sample_params=_sample_all_params,
        site_registry=site_registry,
        site_prefix=prefix,
    )


def iter_dynamics_semantic_bindings(
    spec: DynamicsSpec,
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
    spec: DynamicsSpec,
    samples: dict[str, Array],
    *,
    prefix: str = "vf",
) -> tuple[dict[str, Array], ...]:
    """Pack flat sample-site values into the vector-field param tuple.

    This mirrors ``CompiledDynamics.sample_params`` without entering a NumPyro
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
        elif isinstance(component_spec, NodePotentialSpec):
            packed.append(
                {
                    "center": _parameter_slot_from_samples(
                        component_spec.center,
                        component_spec.center_site_name(site_prefix),
                        samples,
                    ),
                    "stiffness": _parameter_slot_from_samples(
                        component_spec.stiffness,
                        component_spec.decay_site_name(site_prefix),
                        samples,
                    ),
                    "quartic": _parameter_slot_from_samples(
                        component_spec.quartic,
                        component_spec.quartic_site_name(site_prefix),
                        samples,
                    ),
                }
            )
        elif isinstance(component_spec, DiagonalDecaySpec):
            packed.append({"decay": samples[component_spec.decay_site_name(site_prefix)]})
        elif isinstance(component_spec, InterceptSpec):
            packed.append({"cint": samples[component_spec.cint_site_name(site_prefix)]})
        elif isinstance(component_spec, LinearEdgeSpec):
            packed.append({"weight": samples[component_spec.weight_site_name(site_prefix)]})
        elif isinstance(component_spec, HillEdgeSpec):
            packed.append(
                {
                    "Emax": _parameter_slot_from_samples(
                        component_spec.emax,
                        component_spec.emax_site_name(site_prefix),
                        samples,
                    ),
                    "EC50": _parameter_slot_from_samples(
                        component_spec.ec50,
                        component_spec.ec50_site_name(site_prefix),
                        samples,
                    ),
                    "n": _parameter_slot_from_samples(
                        component_spec.n,
                        component_spec.n_site_name(site_prefix),
                        samples,
                    ),
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
