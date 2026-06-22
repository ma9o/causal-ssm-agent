"""Architectural guard: linearization is confined to particle-sampler init.

Project invariant (see AGENTS.md / the SSM design): the continuous-time model is
nonlinear and the framework must never *assume* linearizability on any path that
produces a reported result. The one sanctioned use of the linearised IEKS/Laplace
marginal-likelihood backend is to *initialise* the particle samplers — initial
parameter positions, the proposal preconditioner, and the cSMC reference path.

This test fails if the linearised Laplace backend leaks out of that init/warmup
allowlist (e.g. into a counterfactual, predictive, estimate, or diagnostic path).
It is a static import scan — no JAX, no model build.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src" / "nof1_causal_lab"

# Symbols that construct/return the linearised (EKF/IEKS + Laplace) marginal-
# likelihood backend.
_LAPLACE_BACKEND_SYMBOLS = (
    "make_laplace_backend",
    "build_laplace_backend",
    "LaplaceLikelihood",
)

# The ONLY modules permitted to reference the Laplace backend: its own
# implementation package, the factories on the model, and the warmup/init path
# (Pathfinder/MAP positions + preconditioner, cSMC reference trajectory, and the
# m-PGibbs diagnostic backend stored alongside — never the sampler target).
_ALLOWED_EXACT = {
    "models/ssm/model.py",
    "models/ssm/inference/backend_factory.py",
    "models/ssm/inference/methods/marginal_particle_gibbs/fit.py",
    "models/ssm/inference/warmup/latent_init.py",
    "models/ssm/inference/warmup/map.py",
    "models/ssm/inference/warmup/scipy_pathfinder.py",
}
_ALLOWED_PREFIXES = (
    # The backend's own implementation.
    "models/ssm/inference/targets/laplace/",
)


def _is_allowed(rel_path: str) -> bool:
    return rel_path in _ALLOWED_EXACT or rel_path.startswith(_ALLOWED_PREFIXES)


def test_laplace_backend_is_confined_to_sampler_init() -> None:
    offenders: dict[str, list[str]] = {}
    for py_path in _SRC.rglob("*.py"):
        rel = py_path.relative_to(_SRC).as_posix()
        if _is_allowed(rel):
            continue
        text = py_path.read_text(encoding="utf-8")
        hits = [symbol for symbol in _LAPLACE_BACKEND_SYMBOLS if symbol in text]
        if hits:
            offenders[rel] = hits

    assert not offenders, (
        "Linearised IEKS/Laplace backend referenced outside the sampler-init "
        "allowlist. Linearization may only initialise the particle samplers, never "
        "feed an estimate/counterfactual/predictive/diagnostic. Offending modules: "
        f"{offenders}"
    )
