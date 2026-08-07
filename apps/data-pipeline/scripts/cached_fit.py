"""Run repeatable production fits on Modal with persistent development caches.

This is a development runner, not a production pipeline transition. It calls the
same runtime preparation, Pathfinder warmup, and exact particle inference code as
production while persisting:

* the JAX compilation cache on a Modal Volume;
* a content-addressed Pathfinder/IEKS warmup artifact;
* immutable fit inputs, posterior samples, diagnostics, and PPC output.

Example:
    uv run modal run scripts/cached_fit.py \
        --compiled-ssm ../../scratchpad/fit-input/compiled_ssm.json \
        --panel ../../scratchpad/fit-input/panel.bin \
        --label nine-construct

Sampler overrides are a JSON object:
    uv run modal run scripts/cached_fit.py \
        --compiled-ssm ../../scratchpad/fit-input/compiled_ssm.json \
        --panel ../../scratchpad/fit-input/panel.bin \
        --sampler-overrides ../../scratchpad/fit-input/sampler_overrides.json
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import modal
import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from nof1_causal_lab.flows.modal_runners import (  # noqa: E402
    GPU_A100_80GB,
    gpu_image,
    secrets,
)

type JsonDict = dict[str, Any]
type PanelFormat = Literal["binary", "parquet"]

CACHE_SCHEMA_VERSION = 1
CACHE_VOLUME_NAME = "nof1-cached-fit-cache"
RESULTS_VOLUME_NAME = "nof1-cached-fits"
_CACHE_ROOT = Path("/cache")
_RESULTS_ROOT = Path("/results")
_WARMUP_ARRAYS_FILENAME = "arrays.npz"
_WARMUP_METADATA_FILENAME = "metadata.json"
_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_WARMUP_CONFIG_FIELDS = (
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
_WARMUP_OVERRIDE_FIELDS = (
    "initial_positions_override",
    "parameter_preconditioner_chol",
    "initial_latent_trajectories",
)

app = modal.App("nof1-cached-development-fit")
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
_JAX_CACHE_ENV: dict[str, str | None] = {
    "JAX_COMPILATION_CACHE_DIR": "/cache/jax",
    "JAX_ENABLE_COMPILATION_CACHE": "true",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _update_digest(digest: Any, name: str, payload: bytes) -> None:
    encoded_name = name.encode("utf-8")
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def source_environment_fingerprint(pipeline_root: Path) -> str:
    """Hash inference source, the runner, dependency lock, and configuration."""
    root = pipeline_root.resolve()
    source_root = root / "src" / "nof1_causal_lab"
    if not source_root.is_dir():
        raise ValueError(f"Missing inference source tree: {source_root}")

    digest = hashlib.sha256()
    for relative in ("config.yaml", "pyproject.toml", "uv.lock", "scripts/cached_fit.py"):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Missing cached-fit environment input: {path}")
        _update_digest(digest, relative, path.read_bytes())
    for path in sorted(source_root.rglob("*.py")):
        _update_digest(digest, path.relative_to(root).as_posix(), path.read_bytes())
    return digest.hexdigest()


def fit_input_fingerprint(
    compiled_payload: JsonDict,
    panel_payload: bytes,
    *,
    panel_format: PanelFormat,
) -> str:
    """Hash the complete compiled artifact and exact observation payload."""
    digest = hashlib.sha256()
    _update_digest(digest, "schema_version", str(CACHE_SCHEMA_VERSION).encode())
    _update_digest(digest, "compiled_ssm", _canonical_json_bytes(compiled_payload))
    _update_digest(digest, f"panel:{panel_format}", panel_payload)
    return digest.hexdigest()


def _warmup_sampler_config(sampler_config: JsonDict) -> JsonDict:
    missing = [name for name in _WARMUP_CONFIG_FIELDS if name not in sampler_config]
    if missing:
        raise ValueError(
            "Resolved sampler configuration is missing warmup fields: " + ", ".join(missing)
        )
    return {name: sampler_config[name] for name in _WARMUP_CONFIG_FIELDS}


def warmup_fingerprint(
    *,
    input_fingerprint: str,
    source_fingerprint: str,
    sampler_config: JsonDict,
) -> str:
    """Fingerprint the target, implementation, and warmup policy."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "input_fingerprint": input_fingerprint,
        "source_fingerprint": source_fingerprint,
        "sampler": _warmup_sampler_config(sampler_config),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class WarmupCacheIdentity:
    """Expected identity for one immutable warmup cache entry."""

    fingerprint: str
    input_fingerprint: str
    source_fingerprint: str
    sampler: JsonDict


def _warmup_cache_identity(
    *,
    input_fingerprint: str,
    source_fingerprint: str,
    sampler_config: JsonDict,
) -> WarmupCacheIdentity:
    return WarmupCacheIdentity(
        fingerprint=warmup_fingerprint(
            input_fingerprint=input_fingerprint,
            source_fingerprint=source_fingerprint,
            sampler_config=sampler_config,
        ),
        input_fingerprint=input_fingerprint,
        source_fingerprint=source_fingerprint,
        sampler=_warmup_sampler_config(sampler_config),
    )


def _finite_array(name: str, value: Any) -> np.ndarray:
    import jax

    array = np.asarray(jax.device_get(value))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Warmup array {name!r} contains non-finite values.")
    return array


def _savez_compressed(target: Any, arrays: dict[str, np.ndarray]) -> None:
    """Call NumPy's named-array NPZ API across its incomplete type stub."""
    savez_compressed = cast("Any", np.savez_compressed)
    savez_compressed(target, **arrays)


@dataclass(frozen=True, slots=True)
class CachedWarmup:
    """Pathfinder and IEKS outputs sufficient to skip production warmup."""

    pathfinder_mean: np.ndarray
    pathfinder_chol: np.ndarray
    initial_positions: np.ndarray
    parameter_preconditioner_chol: np.ndarray
    initial_latent_trajectories: np.ndarray | None
    diagnostics: JsonDict

    def validate(self) -> None:
        mean = _finite_array("pathfinder_mean", self.pathfinder_mean)
        chol = _finite_array("pathfinder_chol", self.pathfinder_chol)
        positions = _finite_array("initial_positions", self.initial_positions)
        preconditioner = _finite_array(
            "parameter_preconditioner_chol",
            self.parameter_preconditioner_chol,
        )
        if mean.ndim != 1:
            raise ValueError(f"pathfinder_mean must be rank 1; got {mean.shape}.")
        dim = int(mean.shape[0])
        if chol.shape != (dim, dim):
            raise ValueError(f"pathfinder_chol must have shape ({dim}, {dim}).")
        if positions.ndim != 2 or positions.shape[1] != dim:
            raise ValueError(
                f"initial_positions must have shape (num_chains, dim); got {positions.shape}."
            )
        if preconditioner.shape != (dim, dim):
            raise ValueError(
                "parameter_preconditioner_chol must have shape "
                f"({dim}, {dim}); got {preconditioner.shape}."
            )
        if not np.allclose(chol, np.tril(chol)):
            raise ValueError("pathfinder_chol must be lower triangular.")
        if not np.allclose(preconditioner, np.tril(preconditioner)):
            raise ValueError("parameter_preconditioner_chol must be lower triangular.")
        if self.initial_latent_trajectories is not None:
            trajectories = _finite_array(
                "initial_latent_trajectories",
                self.initial_latent_trajectories,
            )
            if trajectories.ndim != 3:
                raise ValueError(
                    "initial_latent_trajectories must have shape "
                    f"(num_chains, time, latent); got {trajectories.shape}."
                )
            if trajectories.shape[0] != positions.shape[0]:
                raise ValueError(
                    "initial_latent_trajectories and initial_positions must have "
                    "the same num_chains."
                )

    @property
    def parameter_dim(self) -> int:
        return int(self.pathfinder_mean.shape[0])

    @property
    def num_chains(self) -> int:
        return int(self.initial_positions.shape[0])

    def sampler_overrides(self) -> JsonDict:
        self.validate()
        return {
            "initial_positions_override": self.initial_positions,
            "parameter_preconditioner_chol": self.parameter_preconditioner_chol,
            "initial_latent_trajectories": self.initial_latent_trajectories,
        }

    def to_npz_bytes(self) -> bytes:
        self.validate()
        arrays = {
            "pathfinder_mean": _finite_array("pathfinder_mean", self.pathfinder_mean),
            "pathfinder_chol": _finite_array("pathfinder_chol", self.pathfinder_chol),
            "initial_positions": _finite_array(
                "initial_positions",
                self.initial_positions,
            ),
            "parameter_preconditioner_chol": _finite_array(
                "parameter_preconditioner_chol",
                self.parameter_preconditioner_chol,
            ),
        }
        if self.initial_latent_trajectories is not None:
            arrays["initial_latent_trajectories"] = _finite_array(
                "initial_latent_trajectories",
                self.initial_latent_trajectories,
            )
        buffer = io.BytesIO()
        _savez_compressed(buffer, arrays)
        return buffer.getvalue()

    @classmethod
    def from_npz_bytes(
        cls,
        payload: bytes,
        *,
        diagnostics: JsonDict,
    ) -> CachedWarmup:
        required = {
            "pathfinder_mean",
            "pathfinder_chol",
            "initial_positions",
            "parameter_preconditioner_chol",
        }
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("Warmup payload is missing arrays: " + ", ".join(missing))
            artifact = cls(
                pathfinder_mean=np.asarray(archive["pathfinder_mean"]),
                pathfinder_chol=np.asarray(archive["pathfinder_chol"]),
                initial_positions=np.asarray(archive["initial_positions"]),
                parameter_preconditioner_chol=np.asarray(archive["parameter_preconditioner_chol"]),
                initial_latent_trajectories=(
                    np.asarray(archive["initial_latent_trajectories"])
                    if "initial_latent_trajectories" in archive.files
                    else None
                ),
                diagnostics=diagnostics,
            )
        artifact.validate()
        return artifact


def _warmup_cache_path(cache_root: Path, fingerprint: str) -> Path:
    return cache_root / "pathfinder" / fingerprint


def save_cached_warmup(
    cache_root: Path,
    identity: WarmupCacheIdentity,
    artifact: CachedWarmup,
) -> Path:
    """Write one immutable content-addressed warmup entry."""
    import jax

    artifact.validate()
    target = _warmup_cache_path(cache_root, identity.fingerprint)
    target.mkdir(parents=True, exist_ok=False)
    arrays_payload = artifact.to_npz_bytes()
    arrays_sha256 = hashlib.sha256(arrays_payload).hexdigest()
    (target / _WARMUP_ARRAYS_FILENAME).write_bytes(arrays_payload)
    trajectories = artifact.initial_latent_trajectories
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": identity.fingerprint,
        "input_fingerprint": identity.input_fingerprint,
        "source_fingerprint": identity.source_fingerprint,
        "sampler": identity.sampler,
        "arrays_sha256": arrays_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "accelerator": GPU_A100_80GB,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "parameter_dim": artifact.parameter_dim,
        "num_chains": artifact.num_chains,
        "initial_positions_shape": list(artifact.initial_positions.shape),
        "parameter_preconditioner_shape": list(artifact.parameter_preconditioner_chol.shape),
        "initial_latent_trajectories_shape": (
            list(trajectories.shape) if trajectories is not None else None
        ),
        "pathfinder_diagnostics": artifact.diagnostics,
    }
    (target / _WARMUP_METADATA_FILENAME).write_bytes(_canonical_json_bytes(metadata))
    return target


def load_cached_warmup(
    cache_root: Path,
    identity: WarmupCacheIdentity,
) -> CachedWarmup | None:
    """Load an exact hit; reject incomplete or mismatched entries."""
    target = _warmup_cache_path(cache_root, identity.fingerprint)
    if not target.exists():
        return None
    if not target.is_dir():
        raise ValueError(f"Warmup cache entry is not a directory: {target}")

    metadata_path = target / _WARMUP_METADATA_FILENAME
    arrays_path = target / _WARMUP_ARRAYS_FILENAME
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise ValueError(f"Warmup cache entry is incomplete: {target}")

    metadata = json.loads(metadata_path.read_text())
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": identity.fingerprint,
        "input_fingerprint": identity.input_fingerprint,
        "source_fingerprint": identity.source_fingerprint,
        "sampler": identity.sampler,
    }
    mismatched = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatched:
        raise ValueError("Warmup cache metadata mismatch for: " + ", ".join(mismatched))

    arrays_payload = arrays_path.read_bytes()
    if metadata.get("arrays_sha256") != hashlib.sha256(arrays_payload).hexdigest():
        raise ValueError("Warmup cache array checksum mismatch.")
    diagnostics = metadata.get("pathfinder_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("Warmup cache metadata lacks pathfinder_diagnostics.")
    artifact = CachedWarmup.from_npz_bytes(
        arrays_payload,
        diagnostics=diagnostics,
    )
    if artifact.parameter_dim != metadata.get("parameter_dim"):
        raise ValueError("Warmup cache parameter dimension mismatch.")
    if artifact.num_chains != metadata.get("num_chains"):
        raise ValueError("Warmup cache chain count mismatch.")
    return artifact


def _prepare_pathfinder_warmup(runtime: Any, sampler_config: JsonDict) -> CachedWarmup:
    """Run production warmup primitives and capture their reusable outputs."""
    import jax.random as random

    from nof1_causal_lab.models.ssm.autoreparam import AutoReparam
    from nof1_causal_lab.models.ssm.inference.bundle import (
        build_particle_runtime_bundle,
    )
    from nof1_causal_lab.models.ssm.inference.methods._pmcmc_shared import (
        prepare_pmcmc_parameter_warmup,
    )
    from nof1_causal_lab.models.ssm.inference.warmup.latent_init import (
        compute_ieks_latent_paths,
    )
    from nof1_causal_lab.models.ssm.transition_kinds import (
        LATENT_TRANSITION_EULER_MARUYAMA,
    )

    if sampler_config["init_method"] != "pathfinder":
        raise ValueError("Cached warmup requires init_method='pathfinder'.")
    if sampler_config["auto_preconditioner_method"] != "pathfinder":
        raise ValueError("Cached warmup requires auto_preconditioner_method='pathfinder'.")

    seed = int(sampler_config["seed"])
    num_chains = int(sampler_config["num_chains"])
    n_ieks_iters = int(sampler_config["n_ieks_iters"])
    reparam = AutoReparam(centered=0.0)
    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, sample_key = random.split(base_key, 3)
    bundle = build_particle_runtime_bundle(
        runtime.model,
        runtime.observations,
        runtime.times,
        scheme=LATENT_TRANSITION_EULER_MARUYAMA,
        trace_key=trace_key,
        reparam=reparam,
    )
    warmup = prepare_pmcmc_parameter_warmup(
        runtime.model,
        runtime.observations,
        runtime.times,
        bundle=bundle,
        method_label="marginal_particle_gibbs",
        phase_label="cached warmup",
        trace_key=trace_key,
        pathfinder_key=pathfinder_key,
        sample_key=sample_key,
        reparam=reparam,
        seed=seed,
        n_ieks_iters=n_ieks_iters,
        num_chains=num_chains,
        init_method=cast("str", sampler_config["init_method"]),
        initial_positions_override=None,
        init_scale=float(sampler_config["init_scale"]),
        parameter_preconditioner_chol=None,
        auto_preconditioner_method=cast(
            "str",
            sampler_config["auto_preconditioner_method"],
        ),
        auto_preconditioner_maxiter=int(sampler_config["auto_preconditioner_maxiter"]),
        pathfinder_num_elbo_samples=int(sampler_config["pathfinder_num_elbo_samples"]),
        pathfinder_maxiter=int(sampler_config["pathfinder_maxiter"]),
        n_pathfinder_starts=int(sampler_config["n_pathfinder_starts"]),
        pathfinder_parallel_workers=cast(
            "int | None",
            sampler_config["pathfinder_parallel_workers"],
        ),
        pathfinder_init_scale=cast(
            "float | None",
            sampler_config["pathfinder_init_scale"],
        ),
    )
    if warmup.pathfinder_state is None or warmup.pathfinder_diagnostics is None:
        raise RuntimeError("Production Pathfinder did not return reusable state.")
    if warmup.init_positions is None or warmup.preconditioner_chol is None:
        raise RuntimeError("Production Pathfinder did not return sampler initialization.")

    initial_latent_trajectories = None
    if sampler_config["dsmc_leaf_proposal"] == "paid_mix":
        initial_latent_trajectories = compute_ieks_latent_paths(
            runtime.model,
            runtime.observations,
            runtime.times,
            positions=warmup.init_positions,
            trace_key=trace_key,
            reparam=reparam,
            n_ieks_iters=n_ieks_iters,
        )

    return CachedWarmup(
        pathfinder_mean=_finite_array("pathfinder_mean", warmup.pathfinder_state.mean),
        pathfinder_chol=_finite_array("pathfinder_chol", warmup.pathfinder_state.chol),
        initial_positions=_finite_array("initial_positions", warmup.init_positions),
        parameter_preconditioner_chol=_finite_array(
            "parameter_preconditioner_chol",
            warmup.preconditioner_chol,
        ),
        initial_latent_trajectories=(
            _finite_array(
                "initial_latent_trajectories",
                initial_latent_trajectories,
            )
            if initial_latent_trajectories is not None
            else None
        ),
        diagnostics=warmup.pathfinder_diagnostics,
    )


def _deserialize_panel(payload: bytes, panel_format: PanelFormat) -> Any:
    import polars as pl

    buffer = io.BytesIO(payload)
    if panel_format == "binary":
        return pl.DataFrame.deserialize(buffer, format="binary")
    if panel_format == "parquet":
        return pl.read_parquet(buffer)
    raise ValueError(f"Unsupported panel format: {panel_format!r}")


def _resolved_sampler_config(overrides: JsonDict) -> JsonDict:
    from nof1_causal_lab.flows.transitions.inference.flow import build_sampler_config

    resolved = dict(build_sampler_config(None))
    unknown = sorted(set(overrides) - set(resolved))
    if unknown:
        raise ValueError("Unknown sampler override fields: " + ", ".join(unknown))
    forbidden = sorted(set(overrides) & set(_WARMUP_OVERRIDE_FIELDS))
    if forbidden:
        raise ValueError("Warmup array overrides are owned by the cache: " + ", ".join(forbidden))
    resolved.update(overrides)
    return resolved


def _jsonable(value: Any) -> Any:
    import math

    from pydantic_core import to_jsonable_python

    converted = to_jsonable_python(value, fallback=lambda item: repr(item))

    def _clean(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return repr(item)
        if isinstance(item, dict):
            return {str(key): _clean(child) for key, child in item.items()}
        if isinstance(item, list):
            return [_clean(child) for child in item]
        if isinstance(item, tuple):
            return [_clean(child) for child in item]
        return item

    return _clean(converted)


def _diagnostic_extrema(mcmc_diagnostics: JsonDict | None) -> JsonDict:
    if not mcmc_diagnostics:
        return {}
    groups: dict[str, list[float]] = {
        "r_hat": [],
        "ess_bulk": [],
        "ess_tail": [],
    }
    for parameter in mcmc_diagnostics.get("per_parameter", []):
        for name, values in groups.items():
            value = parameter.get(name)
            if value is not None:
                values.extend(np.asarray(value, dtype=float).reshape(-1).tolist())

    extrema: JsonDict = {}
    for name, values in groups.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        extrema[name] = (
            {"min": float(finite.min()), "max": float(finite.max())} if finite.size else None
        )
    extrema.update(
        {
            "parameter_accept_prob_mean": mcmc_diagnostics.get("parameter_accept_prob_mean"),
            "latent_accept_prob_mean": mcmc_diagnostics.get("latent_accept_prob_mean"),
            "num_chains": mcmc_diagnostics.get("num_chains"),
            "num_samples": mcmc_diagnostics.get("num_samples"),
        }
    )
    return extrema


def _validate_label(label: str) -> str:
    if not _LABEL_PATTERN.fullmatch(label):
        raise ValueError(f"label must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}; got {label!r}.")
    return label


def _persist_inference(
    *,
    result_dir: Path,
    fitted: JsonDict,
    compiled_payload: JsonDict,
    panel_payload: bytes,
    panel_format: PanelFormat,
    sampler_config: JsonDict,
    cache_summary: JsonDict,
) -> None:
    import pickle

    from nof1_causal_lab.models.ssm.inference.types import (
        _serialize_fitted_result,
    )

    result = fitted["result"]
    samples = result.get_samples()
    latent_paths = result.get_latent_paths()
    _savez_compressed(
        result_dir / "posterior_samples.npz",
        {name: np.asarray(value) for name, value in samples.items()},
    )
    if latent_paths is not None:
        _savez_compressed(
            result_dir / "latent_paths.npz",
            {"latent_paths": np.asarray(latent_paths)},
        )
    diagnostic_arrays = {
        key: np.asarray(value)
        for key in (
            "chain_complete_log_posterior_history",
            "warmup_complete_log_posterior_history",
            "all_complete_log_posterior_history",
        )
        if (value := result.diagnostics.get(key)) is not None
    }
    latent_summary = result.diagnostics.get("latent_posterior_summary")
    if latent_summary:
        diagnostic_arrays.update(
            {f"latent_summary_{key}": np.asarray(value) for key, value in latent_summary.items()}
        )
    if diagnostic_arrays:
        _savez_compressed(
            result_dir / "diagnostic_arrays.npz",
            diagnostic_arrays,
        )
    with (result_dir / "posterior.pkl").open("wb") as file:
        pickle.dump(_serialize_fitted_result(result), file)
    (result_dir / "compiled_ssm.json").write_bytes(_canonical_json_bytes(compiled_payload))
    panel_suffix = "bin" if panel_format == "binary" else "parquet"
    (result_dir / f"panel.{panel_suffix}").write_bytes(panel_payload)
    (result_dir / "sampler_config.json").write_text(
        json.dumps(_jsonable(sampler_config), indent=2, allow_nan=False)
    )
    (result_dir / "cache.json").write_text(
        json.dumps(_jsonable(cache_summary), indent=2, allow_nan=False)
    )


@app.function(
    timeout=10800,
    cpu=8,
    memory=32768,
    image=gpu_image,
    env=_JAX_CACHE_ENV,
    gpu=GPU_A100_80GB,
    secrets=[secrets],
    volumes={
        "/cache": cache_volume,
        "/results": results_volume,
    },
)
def run_cached_fit(
    compiled_payload: JsonDict,
    panel_payload: bytes,
    panel_format: PanelFormat,
    sampler_overrides: JsonDict,
    source_fingerprint: str,
    label: str,
    reuse_warmup: bool,
    run_ppc_checks: bool,
) -> JsonDict:
    """Run one production fit and persist all reusable development artifacts."""
    import logging

    from nof1_causal_lab.flows.transitions.inference.fit import fit_model, run_ppc
    from nof1_causal_lab.models.ssm.compile.contracts import CompiledSSMArtifact
    from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.monotonic()
    label = _validate_label(label)
    compiled = CompiledSSMArtifact.model_validate(compiled_payload)
    panel = _deserialize_panel(panel_payload, panel_format)
    sampler_config = _resolved_sampler_config(sampler_overrides)
    input_fingerprint = fit_input_fingerprint(
        compiled_payload,
        panel_payload,
        panel_format=panel_format,
    )
    identity = _warmup_cache_identity(
        input_fingerprint=input_fingerprint,
        source_fingerprint=source_fingerprint,
        sampler_config=sampler_config,
    )

    warmup = None
    cache_hit = False
    warmup_seconds = 0.0
    if reuse_warmup:
        warmup = load_cached_warmup(_CACHE_ROOT, identity)
        cache_hit = warmup is not None
        if warmup is None:
            warmup_t0 = time.monotonic()
            runtime = prepare_model_runtime(
                data_for_model=panel,
                compiled_ssm=compiled,
                sampler_config=cast("Any", sampler_config),
            )
            warmup = _prepare_pathfinder_warmup(runtime, sampler_config)
            save_cached_warmup(_CACHE_ROOT, identity, warmup)
            warmup_seconds = time.monotonic() - warmup_t0
            cache_volume.commit()

    fit_sampler_config = dict(sampler_config)
    if warmup is not None:
        fit_sampler_config.update(warmup.sampler_overrides())

    cache_summary = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "input_fingerprint": input_fingerprint,
        "source_fingerprint": source_fingerprint,
        "warmup_fingerprint": identity.fingerprint,
        "warmup_cache_enabled": reuse_warmup,
        "warmup_cache_hit": cache_hit,
        "warmup_seconds": warmup_seconds,
        "jax_cache_volume": CACHE_VOLUME_NAME,
        "jax_cache_path": "/jax",
    }

    try:
        fitted = fit_model(
            compiled,
            panel,
            sampler_config=cast("Any", fit_sampler_config),
            workspace_id=None,
            wait_for_compile_cache=False,
            compute_loo_diagnostics=False,
        )
        if not fitted.get("fitted", False):
            raise RuntimeError(f"Production fit failed: {fitted.get('error')}")

        result = fitted["result"]
        run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        result_name = f"{run_tag}--{label}--{input_fingerprint[:12]}"
        result_dir = _RESULTS_ROOT / "runs" / result_name
        result_dir.mkdir(parents=True, exist_ok=False)
        _persist_inference(
            result_dir=result_dir,
            fitted=fitted,
            compiled_payload=compiled_payload,
            panel_payload=panel_payload,
            panel_format=panel_format,
            sampler_config=sampler_config,
            cache_summary=cache_summary,
        )

        mcmc_diagnostics = fitted.get("mcmc_diagnostics")
        pre_ppc_summary = {
            "completed": False,
            "status": "inference_complete",
            "method": result.method,
            "duration_seconds": float(fitted["duration_seconds"]),
            "wall_seconds": time.monotonic() - started,
            "result_volume": RESULTS_VOLUME_NAME,
            "result_path": f"/runs/{result_name}",
            "cache": cache_summary,
            "diagnostic_extrema": _diagnostic_extrema(mcmc_diagnostics),
        }
        (result_dir / "summary.json").write_text(
            json.dumps(_jsonable(pre_ppc_summary), indent=2, allow_nan=False)
        )
        (result_dir / "mcmc_diagnostics.json").write_text(
            json.dumps(_jsonable(mcmc_diagnostics), indent=2, allow_nan=False)
        )
        results_volume.commit()

        ppc = (
            run_ppc(fitted)
            if run_ppc_checks
            else {
                "checked": False,
                "per_variable_warnings": [],
            }
        )
        (result_dir / "ppc.json").write_text(json.dumps(_jsonable(ppc), indent=2, allow_nan=False))
        samples = result.get_samples()
        latent_paths = result.get_latent_paths()
        summary = {
            **pre_ppc_summary,
            "completed": True,
            "status": "completed",
            "wall_seconds": time.monotonic() - started,
            "sample_sites": len(samples),
            "sample_shapes": {name: list(value.shape) for name, value in samples.items()},
            "latent_path_shape": (list(latent_paths.shape) if latent_paths is not None else None),
            "ppc_checked": bool(ppc.get("checked", False)),
            "ppc_warning_count": len(ppc.get("per_variable_warnings", [])),
        }
        (result_dir / "summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2, allow_nan=False)
        )
        results_volume.commit()
        return cast("JsonDict", _jsonable(summary))
    finally:
        cache_volume.commit()


def _infer_panel_format(path: Path, requested: str) -> PanelFormat:
    if requested:
        if requested not in {"binary", "parquet"}:
            raise ValueError("panel_format must be 'binary' or 'parquet'.")
        return requested
    if path.suffix == ".parquet":
        return "parquet"
    if path.suffix == ".bin":
        return "binary"
    raise ValueError("Cannot infer panel format; use a .bin/.parquet suffix or --panel-format.")


def _load_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


@app.local_entrypoint()
def main(
    compiled_ssm: str,
    panel: str,
    label: str = "fit",
    sampler_overrides: str = "",
    panel_format: str = "",
    reuse_warmup: bool = True,
    run_ppc_checks: bool = True,
) -> None:
    """Load local fit inputs and dispatch the canonical cached Modal runner."""
    compiled_path = Path(compiled_ssm).expanduser().resolve()
    panel_path = Path(panel).expanduser().resolve()
    if not compiled_path.is_file():
        raise ValueError(f"Compiled SSM file does not exist: {compiled_path}")
    if not panel_path.is_file():
        raise ValueError(f"Panel file does not exist: {panel_path}")
    resolved_panel_format = _infer_panel_format(panel_path, panel_format)
    overrides = (
        _load_json_object(Path(sampler_overrides).expanduser().resolve())
        if sampler_overrides
        else {}
    )
    summary = run_cached_fit.remote(
        _load_json_object(compiled_path),
        panel_path.read_bytes(),
        resolved_panel_format,
        overrides,
        source_environment_fingerprint(PIPELINE_ROOT),
        _validate_label(label),
        reuse_warmup,
        run_ppc_checks,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
