"""Focused behavioral matrix for shared inference evaluators."""

from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import nof1_causal_lab.models.ssm.inference.utils as inference_utils
from nof1_causal_lab.models.ssm.constants import MIN_DT


class _RecordingBackend:
    checkpoint_loglik = False

    def __init__(self, lnc: jnp.ndarray) -> None:
        self.lnc = lnc
        self.calls: list[dict[str, Any]] = []

    def _evaluate(self, *args: Any, with_aux: bool, **kwargs: Any) -> Any:
        self.calls.append({"args": args, "kwargs": kwargs, "with_aux": with_aux})
        if with_aux:
            return self.lnc, {"latent_state": jnp.asarray([7.0])}
        return self.lnc

    def compute_log_likelihood(self, *args: Any, **kwargs: Any) -> jnp.ndarray:
        return self._evaluate(*args, with_aux=False, **kwargs)

    def compute_log_likelihood_with_aux(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return self._evaluate(*args, with_aux=True, **kwargs)


def _build_test_evaluators(monkeypatch, *, runtime: bool, backend: _RecordingBackend):
    registry = object()
    assembled_samples: list[dict[str, jnp.ndarray]] = []
    bound_observations = jnp.asarray([[1.0], [2.0], [3.0]])
    bound_times = jnp.asarray([0.0, 0.5, 1.5])
    transition_inputs = jnp.arange(10, dtype=jnp.float32).reshape(5, 2)
    model = SimpleNamespace(spec=object(), transition_inputs=transition_inputs)

    monkeypatch.setattr(
        inference_utils,
        "_build_original_sample_resolver",
        lambda *_args, **_kwargs: lambda samples: {"theta": samples["theta"] + 10.0},
    )
    monkeypatch.setattr(inference_utils, "build_site_registry", lambda _spec: registry)

    def assemble(samples, spec, *, registry: object):
        assert spec is model.spec
        assert registry is not None
        assembled_samples.append(samples)
        return "dynamics", "measurement", "initial", {"obs_df": 5.0}

    monkeypatch.setattr(inference_utils, "_assemble_likelihood_inputs", assemble)
    site_info = {
        "theta": {
            "shape": (),
            "distribution": dist.Normal(0.0, 1.0),
            "transform": dist.transforms.AffineTransform(1.0, 2.0),
            "value": jnp.asarray(0.0),
        }
    }
    functions = inference_utils._build_eval_fns(
        model,
        bound_observations,
        bound_times,
        site_info,
        lambda z: {"theta": z},
        backend,
        include_likelihood_aux=True,
        runtime_observations_times=runtime,
    )
    return functions, assembled_samples, bound_observations, bound_times, transition_inputs


@pytest.mark.parametrize("runtime", [False, True])
def test_eval_fns_share_preparation_and_backend_semantics(monkeypatch, runtime: bool) -> None:
    backend = _RecordingBackend(jnp.asarray([1.0, 2.0, 4.0]))
    (log_lik, _log_prior, log_lik_with_aux), assembled, bound_obs, bound_times, inputs = (
        _build_test_evaluators(monkeypatch, runtime=runtime, backend=backend)
    )
    z = jnp.asarray(2.0)
    if runtime:
        observations = jnp.asarray([[8.0], [9.0]])
        times = jnp.asarray([2.0, 2.25])
        value = log_lik(z, observations, times)
        value_with_aux, aux = log_lik_with_aux(
            z,
            observations,
            times,
            latent_mode_init=jnp.asarray([3.0]),
        )
        expected_inputs = inputs[:2]
    else:
        observations = bound_obs
        times = bound_times
        value = log_lik(z)
        value_with_aux, aux = log_lik_with_aux(z, latent_mode_init=jnp.asarray([3.0]))
        expected_inputs = inputs

    assert float(value) == 4.0
    assert float(value_with_aux) == 4.0
    np.testing.assert_array_equal(aux["latent_state"], [7.0])
    assert len(assembled) == 2
    np.testing.assert_allclose(assembled[0]["theta"], 15.0)

    first_call, aux_call = backend.calls
    assert first_call["with_aux"] is False
    assert aux_call["with_aux"] is True
    assert "latent_mode_init" not in first_call["kwargs"]
    np.testing.assert_array_equal(aux_call["kwargs"]["latent_mode_init"], [3.0])
    np.testing.assert_array_equal(first_call["args"][3], observations)
    np.testing.assert_allclose(
        first_call["args"][4],
        [MIN_DT, *np.diff(np.asarray(times))],
    )
    np.testing.assert_array_equal(first_call["kwargs"]["transition_inputs"], expected_inputs)
    assert first_call["kwargs"]["extra_params"] == {"obs_df": 5.0}


@pytest.mark.parametrize(
    ("lnc", "expected"),
    [(jnp.asarray(2.5), 2.5), (jnp.asarray([0.0, jnp.nan]), -jnp.inf)],
)
def test_eval_fns_normalize_scalar_and_nonfinite_results(monkeypatch, lnc, expected) -> None:
    backend = _RecordingBackend(lnc)
    (log_lik, _log_prior, _log_lik_with_aux), *_ = _build_test_evaluators(
        monkeypatch,
        runtime=False,
        backend=backend,
    )

    result = log_lik(jnp.asarray(0.0))

    if jnp.isneginf(expected):
        assert jnp.isneginf(result)
    else:
        assert float(result) == expected


def test_aux_evaluator_is_not_checkpointed(monkeypatch) -> None:
    checkpointed: list[Any] = []
    monkeypatch.setattr(
        inference_utils.jax,
        "checkpoint",
        lambda fn: checkpointed.append(fn) or fn,
    )
    backend = _RecordingBackend(jnp.asarray(1.0))
    backend.checkpoint_loglik = True

    _build_test_evaluators(monkeypatch, runtime=False, backend=backend)

    assert len(checkpointed) == 1
