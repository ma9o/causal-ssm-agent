from types import SimpleNamespace
from typing import ClassVar

import numpy as np

import nof1_causal_lab.models.ssm.inference.methods.scipy_pathfinder as scipy_pathfinder_module
from nof1_causal_lab.models.ssm.inference.methods.scipy_pathfinder import scipy_pathfinder


def test_scipy_pathfinder_uses_accepted_iterates_for_custom_history(monkeypatch):
    call_counts = {"value_batch": 0, "value_and_grad": 0}
    batch_sizes: list[int] = []

    def log_post_batch_fn(x: np.ndarray) -> np.ndarray:
        call_counts["value_batch"] += 1
        x_np = np.asarray(x, dtype=np.float64)
        batch_sizes.append(int(x_np.shape[0]))
        return -0.5 * np.sum(x_np * x_np, axis=1)

    def log_post_and_grad_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        call_counts["value_and_grad"] += 1
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
        log_post_batch_fn,
        log_post_and_grad_fn,
        [np.array([1.0], dtype=np.float64)],
        maxiter=5,
        elbo_samples=8,
        elbo_candidate_batch_size=4,
        seed=0,
    )

    per_start = result.diagnostics["per_start"][0]
    assert per_start["n_trajectory_points"] == 3
    assert per_start["n_lbfgs_iterations"] == 2
    assert per_start["n_valid_iterates"] == 2
    assert per_start["n_elbo_candidates"] == 2
    assert per_start["n_elbo_batch_evaluations"] == 1
    assert call_counts["value_batch"] == 1
    assert batch_sizes == [32]
    assert call_counts["value_and_grad"] > call_counts["value_batch"]
    assert np.isfinite(result.best_elbo)


def test_scipy_pathfinder_submits_multistarts_to_thread_pool(monkeypatch):
    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _RecordingExecutor:
        instances: ClassVar[list] = []

        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.submissions = []
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            self.submissions.append((fn, args, kwargs))
            return _ImmediateFuture(fn(*args, **kwargs))

    def log_post_and_grad_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_np = np.asarray(x, dtype=np.float64)
        return float(-0.5 * np.dot(x_np, x_np)), -x_np

    def log_post_batch_fn(x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float64)
        return -0.5 * np.sum(x_np * x_np, axis=1)

    def fake_minimize(fun, x0, jac, method, callback=None, options=None):
        del jac, options
        assert method == "L-BFGS-B"
        assert callback is not None
        accepted = np.asarray(x0, dtype=np.float64) * 0.5
        final_fun, _final_grad = fun(accepted)
        callback(accepted)
        return SimpleNamespace(
            x=accepted,
            fun=float(final_fun),
            nit=1,
            status=0,
            success=True,
            hess_inv=None,
        )

    monkeypatch.setattr(scipy_pathfinder_module, "ThreadPoolExecutor", _RecordingExecutor)
    monkeypatch.setattr(scipy_pathfinder_module.scipy.optimize, "minimize", fake_minimize)

    result = scipy_pathfinder(
        log_post_batch_fn,
        log_post_and_grad_fn,
        [
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        ],
        maxiter=2,
        elbo_samples=4,
        elbo_candidate_batch_size=2,
        parallel_workers=2,
        seed=0,
    )

    executor = _RecordingExecutor.instances[0]
    assert executor.max_workers == 2
    assert len(executor.submissions) == 3
    assert result.diagnostics["parallel_workers"] == 2
    assert [item["start_idx"] for item in result.diagnostics["per_start"]] == [0, 1, 2]


def test_scipy_pathfinder_finds_valid_custom_iterates_on_gaussian_target():
    precision = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    def log_post_batch_fn(x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float64)
        return -0.5 * np.einsum("bi,ij,bj->b", x_np, precision, x_np)

    def log_post_and_grad_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_np = np.asarray(x, dtype=np.float64)
        return float(-0.5 * x_np @ precision @ x_np), -(precision @ x_np)

    result = scipy_pathfinder(
        log_post_batch_fn,
        log_post_and_grad_fn,
        [np.array([3.0, -2.0], dtype=np.float64)],
        maxiter=10,
        elbo_samples=20,
        elbo_candidate_batch_size=4,
        seed=0,
    )

    per_start = result.diagnostics["per_start"][0]
    assert per_start["n_valid_iterates"] > 0
    np.testing.assert_allclose(result.mean, np.zeros((2,), dtype=np.float64), atol=1e-3)
