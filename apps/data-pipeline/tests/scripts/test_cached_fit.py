"""Tests for the development-only cached production fit runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _load_cached_fit() -> Any:
    module_name = "cached_fit_under_test"
    path = Path(__file__).resolve().parents[2] / "scripts" / "cached_fit.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cached_fit = _load_cached_fit()
CACHE_SCHEMA_VERSION = cached_fit.CACHE_SCHEMA_VERSION
CachedWarmup = cached_fit.CachedWarmup
WarmupCacheIdentity = cached_fit.WarmupCacheIdentity
fit_input_fingerprint = cached_fit.fit_input_fingerprint
load_cached_warmup = cached_fit.load_cached_warmup
save_cached_warmup = cached_fit.save_cached_warmup
warmup_fingerprint = cached_fit.warmup_fingerprint


def _sampler_config() -> dict[str, object]:
    return {
        "method": "marginal_particle_gibbs",
        "num_warmup": 4000,
        "num_samples": 1000,
        "num_chains": 4,
        "seed": 0,
        "n_particles": 64,
        "n_ieks_iters": 6,
        "init_method": "pathfinder",
        "latent_init_method": "predictive",
        "init_scale": 0.05,
        "pathfinder_num_elbo_samples": 20,
        "pathfinder_maxiter": 20,
        "n_pathfinder_starts": 8,
        "pathfinder_parallel_workers": None,
        "pathfinder_init_scale": 0.1,
        "auto_preconditioner_method": "pathfinder",
        "auto_preconditioner_maxiter": 200,
        "dsmc_leaf_proposal": "paid_mix",
    }


def _identity(
    *,
    input_fingerprint: str = "input",
    source_fingerprint: str = "source",
) -> WarmupCacheIdentity:
    sampler = _sampler_config()
    return WarmupCacheIdentity(
        fingerprint=warmup_fingerprint(
            input_fingerprint=input_fingerprint,
            source_fingerprint=source_fingerprint,
            sampler_config=sampler,
        ),
        input_fingerprint=input_fingerprint,
        source_fingerprint=source_fingerprint,
        sampler={
            name: sampler[name]
            for name in (
                "method",
                "num_chains",
                "seed",
                "n_ieks_iters",
                "init_method",
                "latent_init_method",
                "init_scale",
                "pathfinder_num_elbo_samples",
                "pathfinder_maxiter",
                "n_pathfinder_starts",
                "pathfinder_parallel_workers",
                "pathfinder_init_scale",
                "auto_preconditioner_method",
                "auto_preconditioner_maxiter",
                "dsmc_leaf_proposal",
            )
        },
    )


def _warmup() -> CachedWarmup:
    return CachedWarmup(
        pathfinder_mean=np.array([1.0, 2.0, 3.0]),
        pathfinder_chol=np.diag([1.0, 2.0, 3.0]),
        initial_positions=np.arange(12, dtype=np.float32).reshape(4, 3),
        parameter_preconditioner_chol=np.diag(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        initial_latent_trajectories=np.arange(32, dtype=np.float32).reshape(4, 4, 2),
        diagnostics={"best_pathfinder_elbo": -12.5},
    )


def test_fit_input_fingerprint_covers_model_panel_and_format():
    baseline = fit_input_fingerprint(
        {"spec": {"n_latent": 2}},
        b"panel",
        panel_format="binary",
    )

    assert baseline == fit_input_fingerprint(
        {"spec": {"n_latent": 2}},
        b"panel",
        panel_format="binary",
    )
    assert baseline != fit_input_fingerprint(
        {"spec": {"n_latent": 3}},
        b"panel",
        panel_format="binary",
    )
    assert baseline != fit_input_fingerprint(
        {"spec": {"n_latent": 2}},
        b"other",
        panel_format="binary",
    )
    assert baseline != fit_input_fingerprint(
        {"spec": {"n_latent": 2}},
        b"panel",
        panel_format="parquet",
    )


def test_warmup_fingerprint_ignores_sampling_budget_but_covers_warmup_policy():
    sampler = _sampler_config()
    baseline = warmup_fingerprint(
        input_fingerprint="input",
        source_fingerprint="source",
        sampler_config=sampler,
    )

    different_budget = {**sampler, "num_warmup": 10, "num_samples": 20, "n_particles": 8}
    assert baseline == warmup_fingerprint(
        input_fingerprint="input",
        source_fingerprint="source",
        sampler_config=different_budget,
    )
    assert baseline != warmup_fingerprint(
        input_fingerprint="other",
        source_fingerprint="source",
        sampler_config=sampler,
    )
    assert baseline != warmup_fingerprint(
        input_fingerprint="input",
        source_fingerprint="other",
        sampler_config=sampler,
    )
    assert baseline != warmup_fingerprint(
        input_fingerprint="input",
        source_fingerprint="source",
        sampler_config={**sampler, "pathfinder_maxiter": 21},
    )


def test_cached_warmup_round_trip(tmp_path):
    identity = _identity()
    original = _warmup()

    target = save_cached_warmup(tmp_path, identity, original)
    restored = load_cached_warmup(tmp_path, identity)

    assert target.name == identity.fingerprint
    assert restored is not None
    assert restored.diagnostics == original.diagnostics
    np.testing.assert_array_equal(restored.pathfinder_mean, original.pathfinder_mean)
    np.testing.assert_array_equal(
        restored.pathfinder_chol,
        original.pathfinder_chol,
    )
    np.testing.assert_array_equal(
        restored.initial_positions,
        original.initial_positions,
    )
    np.testing.assert_array_equal(
        restored.parameter_preconditioner_chol,
        original.parameter_preconditioner_chol,
    )
    np.testing.assert_array_equal(
        restored.initial_latent_trajectories,
        original.initial_latent_trajectories,
    )
    with pytest.raises(FileExistsError):
        save_cached_warmup(tmp_path, identity, original)


def test_cached_warmup_rejects_corrupted_arrays(tmp_path):
    identity = _identity()
    target = save_cached_warmup(tmp_path, identity, _warmup())
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema_version"] == CACHE_SCHEMA_VERSION
    (target / "arrays.npz").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="checksum"):
        load_cached_warmup(tmp_path, identity)
