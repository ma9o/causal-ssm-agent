"""Canonical array-shape vocabulary for the SSM numerical core.

Single home for the jaxtyping primitives and the axis-name glossary used across
``models.ssm``. Import array types from here (rather than from ``jaxtyping``
directly) so jaxtyping's per-call named-axis checks stay consistent across
modules and the vocabulary has one documented source of truth.

Axis glossary
-------------
Use these short names in shape strings; the right column is the concept.

==========  =====================================================================
Axis        Meaning
==========  =====================================================================
``D``       latent state dimension (η; elsewhere ``n_latent``)
``M``       manifest / observation dimension (``n_manifest``)
``T``       number of timesteps on the observation grid
``P``       number of latent particles (``num_particles``)
``K``       number of parameter particles (``num_parameter_particles``)
``C``       number of categorical / ordinal response levels
``U``       length of an unconstrained parameter vector
``N``       neutral dimension for generic helpers agnostic to whether the axis
            is latent (D), manifest (M), or other — a square covariance side or
            a reduced vector axis
``*batch``  leading batch axes a rank-polymorphic helper passes through
==========  =====================================================================

Conventions
-----------
- Dtypes stay generic: ``Float`` / ``Int`` / ``Bool``, never bit-width pinned.
  The runtime is float32 but some parameter paths are float64, and ``Float``
  accepts both. Masks / indicators that may arrive as bool *or* float use
  ``Shaped`` (shape-checked, dtype-agnostic).
- A floating-point scalar is ``Float[Array, ""]`` — exported as :data:`FloatScalar`.
- PRNG keys are :data:`PRNGKeyArray` (accepts old ``uint32[2]`` and typed keys).
- Rank-polymorphic helpers use ``*batch`` for leading axes they pass through.
- The vmap rule: annotate the shape a function sees *at its own boundary*. Under
  ``jax.vmap`` that is the reduced rank — a helper mapped over particles sees
  ``Float[Array, " D"]``, not ``Float[Array, "P D"]``.
- Pytree boundaries (the per-parameter ``context`` / batched ``contexts``,
  smoother closures, and ``dict[str, Array]`` sample bundles) are documented in
  prose, not shape-checked: annotate them ``Any``.

Runtime checking is opt-in per module via the ``--jaxtyping-packages`` pytest
flag in ``pyproject.toml``. Instrumented modules must NOT use
``from __future__ import annotations``: eager annotations keep the jaxtyping
imports runtime-resolvable for beartype and stop ruff's ``TCH`` rule from
relocating them into ``TYPE_CHECKING``. For jitted code the checks fire at trace
time, so they are effectively free shape assertions on the numerical core.
"""

from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Shaped

FloatScalar = Float[Array, ""]
"""A floating-point scalar (rank-0 array)."""

__all__ = ["Array", "Bool", "Float", "FloatScalar", "Int", "PRNGKeyArray", "Shaped"]
