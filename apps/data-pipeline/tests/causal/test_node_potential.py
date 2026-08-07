"""Tests for the ``NodePotential`` self-dynamics primitive.

``NodePotential`` is node-intrinsic self-regulation — the gradient (curl-free)
part of the drift, distinct from a directed edge. A quadratic well
(``quartic == 0``) reproduces ``StateDecay`` + ``StateIntercept`` exactly; a
positive ``quartic`` adds stiffening self-limitation.

Coverage: the runtime contribution, the decay+intercept equivalence (the
"recovery" property), spec compilation / sampling / packing, dict serialization
round-trip, and the warmup-safety property (a ``NodePotential`` field is
classified trajectory-dependent, so it never enters the affine fast path that
``derive_affine_dynamics`` would reject).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

from nof1_causal_lab.models.ssm.dynamics import (
    Fixed,
    Free,
    NodePotential,
    StateDecay,
    StateIntercept,
    VectorField,
    VectorFieldArgs,
    infer_linearisation,
)
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.serialization import (
    dynamics_spec_from_dict,
    dynamics_spec_to_dict,
)
from nof1_causal_lab.models.ssm.dynamics.spec import (
    DynamicsSpec,
    NodePotentialSpec,
    compile_dynamics,
    iter_dynamics_semantic_bindings,
    pack_component_params_from_samples,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind


def _args(params: tuple[dict[str, jnp.ndarray], ...]) -> VectorFieldArgs:
    return VectorFieldArgs(params=params, intervention=Intervention.none())


def _params(
    center: float,
    stiffness: float,
    quartic: float = 0.0,
) -> dict[str, jnp.ndarray]:
    return {
        "center": jnp.asarray(center),
        "stiffness": jnp.asarray(stiffness),
        "quartic": jnp.asarray(quartic),
    }


class TestNodePotentialContribute:
    def test_quadratic_is_linear_restoring_force(self):
        out = NodePotential(target=0).contribute(
            jnp.zeros(1), jnp.array([3.0]), jnp.zeros((1, 1)), jnp.asarray(0.0), _params(1.0, 2.0)
        )
        # -stiffness * (eta - center) = -2 * (3 - 1) = -4
        assert float(out[0]) == pytest.approx(-4.0)

    def test_quartic_adds_cubic_term(self):
        out = NodePotential(target=0).contribute(
            jnp.zeros(1),
            jnp.array([2.0]),
            jnp.zeros((1, 1)),
            jnp.asarray(0.0),
            _params(0.0, 1.0, 0.5),
        )
        # -(stiffness*d + quartic*d^3) = -(1*2 + 0.5*8) = -6
        assert float(out[0]) == pytest.approx(-6.0)

    def test_only_touches_target(self):
        out = NodePotential(target=1).contribute(
            jnp.zeros(3),
            jnp.array([5.0, 5.0, 5.0]),
            jnp.zeros((3, 3)),
            jnp.asarray(0.0),
            _params(0.0, 1.0),
        )
        assert float(out[0]) == 0.0
        assert float(out[2]) == 0.0
        assert float(out[1]) == pytest.approx(-5.0)


class TestDecayInterceptEquivalence:
    @pytest.mark.parametrize("eta_val", [-2.0, 0.0, 1.5, 4.0])
    def test_quadratic_potential_matches_decay_plus_intercept(self, eta_val):
        k, mu = 0.7, 2.0  # stiffness (relaxation rate), center (set-point)
        eta, t = jnp.array([eta_val]), jnp.asarray(0.0)

        pot_field = VectorField(n_latent=1, components=(NodePotential(target=0),))
        pot_args = _args((_params(mu, k),))

        # StateDecay(-k*eta) + StateIntercept(+k*mu) == -k*eta + k*mu == -k(eta - mu)
        lin_field = VectorField(
            n_latent=1, components=(StateDecay(target=0), StateIntercept(target=0))
        )
        lin_args = _args(({"decay": jnp.asarray(k)}, {"cint": jnp.asarray(k * mu)}))

        assert float(pot_field(t, eta, pot_args)[0]) == pytest.approx(
            float(lin_field(t, eta, lin_args)[0]), abs=1e-6
        )

    def test_center_is_the_steady_state(self):
        k, mu = 1.3, -0.5
        field = VectorField(n_latent=1, components=(NodePotential(target=0),))
        args = _args((_params(mu, k),))
        # drift vanishes exactly at eta = center (the well minimum)
        assert float(field(jnp.asarray(0.0), jnp.array([mu]), args)[0]) == pytest.approx(
            0.0, abs=1e-6
        )


class TestNodePotentialSpec:
    def test_default_emits_center_and_decay_but_not_quartic_site(self):
        compiled = compile_dynamics(
            DynamicsSpec(n_latent=1, components=(NodePotentialSpec(target=0),))
        )
        names = {s.name for s in compiled.site_registry}
        assert any(n.endswith("_center") for n in names)
        # the relaxation rate is sampled at the shared decay site
        assert any(n.endswith("_decay") for n in names)
        assert not any(n.endswith("_quartic") for n in names)  # quartic fixed at 0 by default

    def test_free_quartic_emits_site(self):
        compiled = compile_dynamics(
            DynamicsSpec(n_latent=1, components=(NodePotentialSpec(target=0, quartic=Free()),))
        )
        assert any(s.name.endswith("_quartic") for s in compiled.site_registry)

    def test_sample_and_pack_roundtrip(self):
        spec = DynamicsSpec(n_latent=1, components=(NodePotentialSpec(target=0),))
        compiled = compile_dynamics(spec)

        def prior_fn(name: str) -> dist.Distribution:
            return dist.Normal(0.0, 1.0) if name.endswith("_center") else dist.HalfNormal(1.0)

        tr = trace(seed(lambda: compiled.sample_params(prior_fn), 0)).get_trace()
        samples = {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}

        packed = pack_component_params_from_samples(spec, samples)
        assert len(packed) == 1
        assert set(packed[0]) == {"center", "stiffness", "quartic"}
        assert float(packed[0]["quartic"]) == 0.0  # fixed default flows through
        center_site = next(n for n in samples if n.endswith("_center"))
        assert float(packed[0]["center"]) == pytest.approx(float(samples[center_site]))

    def test_fixed_stiffness_must_be_positive(self):
        with pytest.raises(ValueError, match="stiffness"):
            NodePotentialSpec(target=0, stiffness=Fixed(0.0))

    def test_fixed_quartic_must_be_nonnegative(self):
        with pytest.raises(ValueError, match="quartic"):
            NodePotentialSpec(target=0, quartic=Fixed(-1.0))


class TestSerializationRoundTrip:
    def test_default_roundtrip(self):
        spec = DynamicsSpec(n_latent=2, components=(NodePotentialSpec(target=1),))
        payload = dynamics_spec_to_dict(spec)
        assert payload["components"][0] == {
            "kind": "NodePotential",
            "target": 1,
            "parameters": {
                "center": {"kind": "free"},
                "stiffness": {"kind": "free"},
                "quartic": {"kind": "fixed", "value": 0.0},
            },
        }
        comp = dynamics_spec_from_dict(payload).components[0]
        assert isinstance(comp, NodePotentialSpec)
        assert comp.target == 1
        assert isinstance(comp.center, Free)
        assert isinstance(comp.stiffness, Free)
        assert comp.quartic == Fixed(0.0)

    def test_fixed_and_free_quartic_roundtrip(self):
        spec = DynamicsSpec(
            n_latent=1,
            components=(
                NodePotentialSpec(
                    target=0,
                    center=Fixed(0.5),
                    stiffness=Fixed(1.2),
                    quartic=Free(),
                ),
            ),
        )
        comp = dynamics_spec_from_dict(dynamics_spec_to_dict(spec)).components[0]
        assert isinstance(comp, NodePotentialSpec)
        assert comp.center == Fixed(0.5)
        assert comp.stiffness == Fixed(1.2)
        assert isinstance(comp.quartic, Free)


class TestWarmupSafety:
    def test_node_potential_is_trajectory_dependent(self):
        # Even a quadratic (affine) NodePotential must classify as "trajectory":
        # this routes warmup through local linearization and keeps it out of
        # derive_affine_dynamics, which does not handle NodePotential.
        field = VectorField(n_latent=1, components=(NodePotential(target=0),))
        assert infer_linearisation(field) == "trajectory"


class TestSemanticBindings:
    def test_default_binds_setpoint_and_decay_by_construct(self):
        spec = DynamicsSpec(n_latent=2, components=(NodePotentialSpec(target=1),))
        bindings = list(iter_dynamics_semantic_bindings(spec, latent_names=("mood", "energy")))
        by_name = {b.parameter_name: b for b in bindings}
        # target=1 -> "energy"; quartic fixed at 0 -> no self_limit binding.
        # Stiffness reuses StateDecay's persistence/decay authoring contract.
        assert set(by_name) == {"setpoint_energy", "rho_energy", "decay_energy"}
        assert by_name["setpoint_energy"].prior_field == "dynamics_potential_center"
        assert by_name["setpoint_energy"].construct_names == ("energy",)
        assert by_name["rho_energy"].prior_field == "dynamics_decay"
        assert by_name["rho_energy"].site_kind == SiteKind.DYNAMICS_DECAY
        assert by_name["decay_energy"].prior_field == "dynamics_decay"

    def test_free_quartic_adds_self_limit_binding(self):
        spec = DynamicsSpec(n_latent=2, components=(NodePotentialSpec(target=0, quartic=Free()),))
        names = {
            b.parameter_name
            for b in iter_dynamics_semantic_bindings(spec, latent_names=("mood", "energy"))
        }
        assert names == {"setpoint_mood", "rho_mood", "decay_mood", "self_limit_mood"}
