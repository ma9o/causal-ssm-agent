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

    def sample_params(self, prefix: str) -> dict[str, Array]:
        """Call inside a NumPyro model; returns this component's param slice."""
        ...


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

    def sample_params(self, prefix: str) -> dict[str, Array]:
        drift_dist = resolve_prior_distribution(self.drift_prior)
        cint_dist = resolve_prior_distribution(self.cint_prior)
        out: dict[str, Array] = {
            "drift": numpyro.sample(f"{prefix}_drift", drift_dist),
        }
        if cint_dist is not None:
            out["cint"] = numpyro.sample(f"{prefix}_cint", cint_dist)
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

    def iter_sites(self, prefix: str) -> Iterator[SiteDescriptor]:
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

    def iter_sites(self, prefix: str) -> Iterator[SiteDescriptor]:
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

    def with_runtime_priors(self, prior_fn, *, prefix: str) -> StructuralInterceptSpec:
        return replace(
            self,
            cint_prior=prior_fn(self.cint_site_name(prefix)) if self.n_cint > 0 else self.cint_prior,
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
                    "StructuralInterceptSpec requires cint_prior when "
                    f"n_cint={self.n_cint} > 0."
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

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {"decay": numpyro.sample(f"{prefix}_decay", resolve_prior_distribution(self.decay_prior))}


@dataclass(frozen=True, eq=False)
class InterceptSpec:
    """``Intercept`` component. Prior over ``(n_latent,)`` intercept vector."""

    cint_prior: Any

    def build(self) -> Intercept:
        return Intercept()

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {"cint": numpyro.sample(f"{prefix}_cint", resolve_prior_distribution(self.cint_prior))}


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

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {"weight": numpyro.sample(f"{prefix}_weight", resolve_prior_distribution(self.weight_prior))}


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

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {
            "Emax": numpyro.sample(f"{prefix}_Emax", resolve_prior_distribution(self.emax_prior)),
            "EC50": numpyro.sample(f"{prefix}_EC50", resolve_prior_distribution(self.ec50_prior)),
            "n": numpyro.sample(f"{prefix}_n", resolve_prior_distribution(self.n_prior)),
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

    def sample_params(self, prefix: str) -> dict[str, Array]:
        return {"weight": numpyro.sample(f"{prefix}_weight", resolve_prior_distribution(self.weight_prior))}


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

    vector_field: CompositeVectorField
    sample_params: Callable[[], tuple[dict[str, Array], ...]]


def linear_drift_spec(
    *,
    n_latent: int,
    drift_diag_mask: np.ndarray,
    drift_offdiag_mask: np.ndarray,
    drift_template: jnp.ndarray,
    cint_mask: np.ndarray,
    cint_template: jnp.ndarray,
    time_invariant_mask: np.ndarray | None = None,
    stability_margin: float = 0.05,
    base_decay_prior: Any = None,
    offdiag_prior: Any = None,
    cint_prior: Any = None,
) -> CompositeSpec:
    """Build the canonical 2-component CompositeSpec for a linear drift.

    Wraps :class:`StructuralDenseLinearSpec` (drift) plus
    :class:`StructuralInterceptSpec` (intercept) into a single
    ``CompositeSpec``. This is the standard linear-path drift_spec used
    throughout the codebase; the factory absorbs the verbose two-component
    construction.
    """
    return CompositeSpec(
        n_latent=n_latent,
        components=(
            StructuralDenseLinearSpec(
                n_latent=n_latent,
                drift_diag_mask=drift_diag_mask,
                drift_offdiag_mask=drift_offdiag_mask,
                drift_template=drift_template,
                stability_margin=stability_margin,
                time_invariant_mask=time_invariant_mask,
                base_decay_prior=base_decay_prior,
                offdiag_prior=offdiag_prior,
            ),
            StructuralInterceptSpec(
                n_latent=n_latent,
                cint_mask=cint_mask,
                cint_template=cint_template,
                cint_prior=cint_prior,
            ),
        ),
    )


def default_linear_drift_spec(n_latent: int) -> CompositeSpec:
    """Full-free linear drift around zero with no intercept.

    The standard test default: every drift entry sampleable, no continuous
    intercept, zeros template. Tests that need a fixed drift, a sparse
    drift mask, or a free intercept construct :func:`linear_drift_spec`
    directly.
    """
    import jax.numpy as _jnp
    import numpy as _np

    diag_mask = _np.ones(n_latent, dtype=bool)
    offdiag_mask = _np.ones((n_latent, n_latent), dtype=bool) & ~_np.eye(
        n_latent, dtype=bool
    )
    return linear_drift_spec(
        n_latent=n_latent,
        drift_diag_mask=diag_mask,
        drift_offdiag_mask=offdiag_mask,
        drift_template=_jnp.zeros((n_latent, n_latent)),
        cint_mask=_np.zeros(n_latent, dtype=bool),
        cint_template=_jnp.zeros(n_latent),
    )


def compile_composite(
    spec: CompositeSpec, *, prefix: str = "vf"
) -> CompiledComposite:
    """Compile a ``CompositeSpec`` into a ``CompositeVectorField`` and a
    NumPyro-callable parameter sampler.

    The ``prefix`` is prepended to every NumPyro sample site name so
    nested compositions or multiple SSMs in one model stay disambiguated.
    """
    components = tuple(component_spec.build() for component_spec in spec.components)
    vector_field = CompositeVectorField(n_latent=spec.n_latent, components=components)
    component_specs = spec.components

    def _sample_all_params() -> tuple[dict[str, Array], ...]:
        return tuple(
            component_spec.sample_params(prefix=f"{prefix}_{i}")
            for i, component_spec in enumerate(component_specs)
        )

    return CompiledComposite(
        vector_field=vector_field, sample_params=_sample_all_params
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
            packed.append({"drift": deterministics[component_spec.drift_deterministic_name(site_prefix)]})
        elif isinstance(component_spec, StructuralInterceptSpec):
            packed.append({"cint": deterministics[component_spec.cint_deterministic_name(site_prefix)]})
        elif isinstance(component_spec, DenseLinearSpec):
            params = {"drift": samples[f"{site_prefix}_drift"]}
            cint_name = f"{site_prefix}_cint"
            if cint_name in samples:
                params["cint"] = samples[cint_name]
            packed.append(params)
        elif isinstance(component_spec, DiagonalDecaySpec):
            packed.append({"decay": samples[f"{site_prefix}_decay"]})
        elif isinstance(component_spec, InterceptSpec):
            packed.append({"cint": samples[f"{site_prefix}_cint"]})
        elif isinstance(component_spec, LinearEdgeSpec):
            packed.append({"weight": samples[f"{site_prefix}_weight"]})
        elif isinstance(component_spec, HillEdgeSpec):
            packed.append(
                {
                    "Emax": samples[f"{site_prefix}_Emax"],
                    "EC50": samples[f"{site_prefix}_EC50"],
                    "n": samples[f"{site_prefix}_n"],
                }
            )
        elif isinstance(component_spec, MultiplicativeEdgeSpec):
            packed.append({"weight": samples[f"{site_prefix}_weight"]})
        else:
            raise TypeError(
                f"Unsupported drift component spec for parameter packing: "
                f"{type(component_spec).__name__}"
            )
    return tuple(packed)
