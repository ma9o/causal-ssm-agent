from types import SimpleNamespace

import numpy as np

import causal_ssm_agent.models.ssm.inference.methods.scipy_pathfinder as scipy_pathfinder_module
from causal_ssm_agent.models.ssm.inference.methods.scipy_pathfinder import scipy_pathfinder


def test_scipy_pathfinder_uses_accepted_iterates_for_custom_history(monkeypatch):
    def log_post_and_grad_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_np = np.asarray(x, dtype=np.float64)
        return float(-0.5 * np.dot(x_np, x_np)), -x_np

    def fake_minimize(fun, x0, jac, method, callback=None, options=None):
        del jac, options
        assert method == "L-BFGS-B"
        assert callback is not None
        x0_np = np.asarray(x0, dtype=np.float64)
        accepted_1 = np.array([0.5], dtype=np.float64)
        accepted_2 = np.array([0.1], dtype=np.float64)

        fun(x0_np)
        fun(np.array([0.75], dtype=np.float64))
        fun(accepted_1)
        callback(accepted_1)
        fun(np.array([0.2], dtype=np.float64))
        final_fun, _final_grad = fun(accepted_2)
        callback(accepted_2)
        return SimpleNamespace(
            x=accepted_2,
            fun=float(final_fun),
            nit=2,
            status=0,
            success=True,
            hess_inv=None,
        )

    monkeypatch.setattr(scipy_pathfinder_module.scipy.optimize, "minimize", fake_minimize)

    result = scipy_pathfinder(
        log_post_and_grad_fn,
        [np.array([1.0], dtype=np.float64)],
        maxiter=5,
        elbo_samples=8,
        seed=0,
    )

    per_start = result.diagnostics["per_start"][0]
    assert per_start["n_trajectory_points"] == 3
    assert per_start["n_lbfgs_iterations"] == 2
    assert per_start["n_valid_iterates"] == 2
    assert np.isfinite(result.best_elbo)


def test_scipy_pathfinder_finds_valid_custom_iterates_on_gaussian_target():
    precision = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    def log_post_and_grad_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_np = np.asarray(x, dtype=np.float64)
        return float(-0.5 * x_np @ precision @ x_np), -(precision @ x_np)

    result = scipy_pathfinder(
        log_post_and_grad_fn,
        [np.array([3.0, -2.0], dtype=np.float64)],
        maxiter=10,
        elbo_samples=20,
        seed=0,
    )

    per_start = result.diagnostics["per_start"][0]
    assert per_start["n_valid_iterates"] > 0
    np.testing.assert_allclose(result.mean, np.zeros((2,), dtype=np.float64), atol=1e-3)
