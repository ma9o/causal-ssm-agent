"""Composite SSM runtime envelope.

:class:`RuntimeSSM` is the closure-bearing runtime artifact that
composite-style consumers (Gibbs MCMC, MAP, prior predictive) work
against. It bundles vector field, parameter sampler, initial-state
moments, diffusion covariance, observation operator, observation kernel,
and a constant-vs-trajectory linearisation hint.

Constructed via :func:`runtime_from_ssm_model` (the canonical bridge
from an :class:`SSMModel` with a populated ``drift_spec``) or
:func:`runtime_from_composite` when the caller already has a compiled
``CompiledComposite``. :func:`runtime_from_dense_linear` remains the
small adapter for code that already has concrete ``(A, c)`` draws.

The ``linearisation`` field is a fast-path hint: ``"constant"`` when
every drift component has a state-independent Jacobian (``DenseLinear``,
``DiagonalDecay``, ``Intercept``, ``LinearEdge``), so the inference
driver can skip per-step relinearisation; ``"trajectory"`` for
non-linear primitives (``HillEdge``, ``MultiplicativeEdge``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

from .edges import DenseLinear, DiagonalDecay, Intercept, LinearEdge

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from nof1_causal_lab.models.ssm import SSMModel
    from nof1_causal_lab.models.ssm.inference.targets.kernels import ObservationKernel
    from nof1_causal_lab.models.ssm.inference.targets.observation_dispatch import (
        PredictiveObservationSampler,
    )

    from .composite import CompiledComposite
    from .vector_field import CompositeVectorField


# Components whose Jacobian is independent of the state. A composite
# vector field built entirely from these is bit-for-bit equivalent to a
# single dense-linear vector field — so the "constant" fast-path applies.
_CONSTANT_JACOBIAN_COMPONENTS = (DenseLinear, DiagonalDecay, Intercept, LinearEdge)


Linearisation = Literal["constant", "trajectory"]


def infer_linearisation(vector_field: CompositeVectorField) -> Linearisation:
    """Classify whether the composite drift has a state-independent Jacobian.

    Returns ``"constant"`` when every component is one of
    ``DenseLinear``, ``DiagonalDecay``, ``Intercept``, ``LinearEdge`` —
    those have linear-in-state drift contributions, so ``∂f/∂x`` is
    state-independent and one linearisation suffices for the whole
    trajectory. Returns ``"trajectory"`` otherwise (Hill, multiplicative,
    or any future non-linear primitive).
    """
    return (
        "constant"
        if all(isinstance(c, _CONSTANT_JACOBIAN_COMPONENTS) for c in vector_field.components)
        else "trajectory"
    )


def _build_predictive_sampler(
    manifest_dists: tuple,
    R: Array,
    *,
    manifest_links: tuple = (),
    obs_extra_params: dict | None = None,
) -> PredictiveObservationSampler | None:
    if not manifest_dists:
        return None

    from nof1_causal_lab.models.ssm.inference.targets.observation_dispatch import (
        build_predictive_observation_sampler,
    )

    return build_predictive_observation_sampler(
        manifest_dists,
        R,
        manifest_links=manifest_links or None,
        extra_params=obs_extra_params,
    )


@dataclass(frozen=True)
class RuntimeSSM:
    """Internal canonical SSM representation. See module docstring.

    The optional ``predictive_sampler`` is materialised when callers
    pass ``manifest_dists`` / ``manifest_links`` / ``obs_extra_params``
    through the adapter. It powers non-Gaussian observation sampling
    via the existing ``build_predictive_observation_sampler`` factory,
    so the composite prior-predictive can emit Beta / Binomial /
    Poisson / Student-t observations without re-implementing per-family
    samplers.
    """

    vector_field: CompositeVectorField
    sample_params: Callable[[], tuple[dict[str, Array], ...]]
    init_mean: Array
    init_cov: Array
    diffusion_cov: Array
    H: Array
    d_meas: Array
    R: Array
    obs_kernel: ObservationKernel
    linearisation: Linearisation
    site_prefix: str = "vf"
    predictive_sampler: PredictiveObservationSampler | None = None


def runtime_from_composite(
    compiled: CompiledComposite,
    *,
    init_mean: Array,
    init_cov: Array,
    diffusion_cov: Array,
    H: Array,
    d_meas: Array,
    R: Array,
    obs_kernel: ObservationKernel,
    site_prefix: str = "vf",
    manifest_dists: tuple = (),
    manifest_links: tuple = (),
    obs_extra_params: dict | None = None,
) -> RuntimeSSM:
    """Translate a ``CompiledComposite`` plus external SSM parts into a
    :class:`RuntimeSSM`.

    The ``linearisation`` hint is inferred from the vector field's
    components — callers do not need to supply it. A spec with only
    linear-in-state components gets ``"constant"`` automatically.

    Passing ``manifest_dists``, ``manifest_links`` and ``obs_extra_params``
    additionally builds a :class:`PredictiveObservationSampler` for the
    canonical, which lets downstream callers (prior-predictive,
    posterior-predictive) sample non-Gaussian observations via the
    existing family registry rather than re-implementing per-family
    samplers.
    """
    return RuntimeSSM(
        vector_field=compiled.vector_field,
        sample_params=compiled.sample_params,
        init_mean=init_mean,
        init_cov=init_cov,
        diffusion_cov=diffusion_cov,
        H=H,
        d_meas=d_meas,
        R=R,
        obs_kernel=obs_kernel,
        linearisation=infer_linearisation(compiled.vector_field),
        site_prefix=site_prefix,
        predictive_sampler=_build_predictive_sampler(
            manifest_dists,
            R,
            manifest_links=manifest_links,
            obs_extra_params=obs_extra_params,
        ),
    )


def runtime_from_ssm_model(
    model: SSMModel,
    *,
    obs_kernel: ObservationKernel,
    obs_extra_params: dict | None = None,
) -> RuntimeSSM:
    """Translate an ``SSMModel`` (carrying an ``SSMSpec``) into a :class:`RuntimeSSM`.

    Pulls vector field, initial-state moments, diffusion covariance, and
    observation operator from the spec's template values. For composite
    specs the sample callable comes from the compiled composite drift
    spec; linear SSMs are represented as structural composite specs too.

    The spec's template fields are treated as fixed runtime values — this
    factory is the right shape for composite-style Gibbs MCMC where the
    SSM hyperparams are conditioned on rather than sampled. The mask
    fields are not inspected; callers that want to sample the
    hyperparams should keep using the linear ``SSMModel`` numpyro
    pipeline directly.
    """
    from nof1_causal_lab.models.ssm.dynamics.composite import compile_composite

    spec = model.spec
    compiled = compile_composite(spec.drift_spec)

    diffusion_chol = jnp.asarray(spec.diffusion_block.diffusion_chol_template)
    diffusion_cov = diffusion_chol @ diffusion_chol.T

    t0_chol = jnp.asarray(spec.t0_chol_block.template)
    init_cov = t0_chol @ t0_chol.T

    manifest_chol = jnp.asarray(spec.manifest_chol_block.template)
    R = manifest_chol @ manifest_chol.T

    return runtime_from_composite(
        compiled,
        init_mean=jnp.asarray(spec.t0_means_block.template),
        init_cov=init_cov,
        diffusion_cov=diffusion_cov,
        H=jnp.asarray(spec.lambda_block.template),
        d_meas=jnp.asarray(spec.manifest_means_block.template),
        R=R,
        obs_kernel=obs_kernel,
        manifest_dists=tuple(spec.manifest_dists or ()),
        manifest_links=tuple(spec.manifest_links or ()),
        obs_extra_params=obs_extra_params,
    )


def runtime_from_dense_linear(
    drift: Array,
    cint: Array,
    *,
    init_mean: Array,
    init_cov: Array,
    diffusion_cov: Array,
    H: Array,
    d_meas: Array,
    R: Array,
    obs_kernel: ObservationKernel,
) -> RuntimeSSM:
    """Build a :class:`RuntimeSSM` from a single ``(drift, cint)`` pair.

    For callers that already have linear-path posterior samples and want
    to consume them through the canonical surface (Stage 6 counterfactual
    orchestrator, Stage 4 prior-predictive validator). ``sample_params``
    returns the Delta-distributed parameters so the canonical-shaped
    consumer can treat the linear pair as a single-component tuple.

    This is the minimum-viable translator for callers that already have
    concrete posterior draws and do not need the compiled artifact
    machinery.
    """
    from .vector_field import CompositeVectorField

    n_latent = int(drift.shape[-1])
    vf = CompositeVectorField(n_latent=n_latent, components=(DenseLinear(),))
    params_tuple: tuple[dict[str, Array], ...] = ({"drift": drift, "cint": cint},)

    def _sample_delta() -> tuple[dict[str, Array], ...]:
        return params_tuple

    return RuntimeSSM(
        vector_field=vf,
        sample_params=_sample_delta,
        init_mean=init_mean,
        init_cov=init_cov,
        diffusion_cov=diffusion_cov,
        H=H,
        d_meas=d_meas,
        R=R,
        obs_kernel=obs_kernel,
        linearisation="constant",
    )
