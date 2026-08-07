"""Modal-backed execution for expensive transitions.

When ``DEPLOYMENT_ENV=production``, ``machine.runners.execute_transition``
routes statistical model specification and posterior inference here. The remote
function runs the same ``execute_transition_locally`` against the same R2-backed artifact store —
Modal is compute placement, not a different execution path. Version
stamps come back as plain dicts (Modal pickles across an image boundary).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import modal
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.machine.artifacts import ArtifactId, EpisodeState
    from nof1_causal_lab.machine.moves import ExecOptions, TransitionEffects

# ═══════════════════════════════════════════════════════════════════════════════
# Modal images
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[3]  # -> apps/data-pipeline/
GPU_A100_80GB = "A100-80GB"

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev", "cloud"], frozen=True)
    .env({"PYTHONPATH": "/root/src", "DEPLOYMENT_ENV": "production"})
    .add_local_file(ROOT / "config.yaml", remote_path="/root/config.yaml")
    .add_local_file(ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
    .add_local_dir(ROOT / "src" / "nof1_causal_lab", remote_path="/root/src/nof1_causal_lab")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev", "cloud"], frozen=True)
    .uv_pip_install("jax[cuda12]", gpu=GPU_A100_80GB)
    .env({"PYTHONPATH": "/root/src", "DEPLOYMENT_ENV": "production"})
    .add_local_file(ROOT / "config.yaml", remote_path="/root/config.yaml")
    .add_local_file(ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
    .add_local_dir(ROOT / "src" / "nof1_causal_lab", remote_path="/root/src/nof1_causal_lab")
)

app = modal.App("nof1-causal-lab-pipeline", image=cpu_image)
secrets = modal.Secret.from_name("nof1-causal-lab-pipeline-secrets")


# ═══════════════════════════════════════════════════════════════════════════════
# Remote functions
# ═══════════════════════════════════════════════════════════════════════════════


async def _run_transition_remote(
    workspace_id: str,
    artifact_id: str,
    pins: dict[str, int],
    state: JsonObject,
    options: JsonObject,
) -> JsonObject:
    from nof1_causal_lab.machine.artifacts import ArtifactId, EpisodeState
    from nof1_causal_lab.machine.moves import ExecOptions
    from nof1_causal_lab.machine.runners import execute_transition_locally

    validated_artifact_id = TypeAdapter(ArtifactId).validate_python(artifact_id)
    validated_pins = TypeAdapter(dict[ArtifactId, int]).validate_python(pins)
    result = await execute_transition_locally(
        workspace_id,
        validated_artifact_id,
        validated_pins,
        EpisodeState.model_validate(state),
        ExecOptions.model_validate(options),
    )
    return cast("JsonObject", result.model_dump(mode="json"))


@app.function(
    timeout=10800,
    cpu=8,
    memory=32768,
    image=gpu_image,
    gpu=GPU_A100_80GB,
    secrets=[secrets],
)
async def _run_transition_gpu(
    workspace_id: str,
    artifact_id: str,
    pins: dict[str, int],
    state: JsonObject,
    options: JsonObject,
) -> JsonObject:
    """Run a transition on Modal GPU compute against the R2 artifact store."""
    return await _run_transition_remote(
        workspace_id,
        artifact_id,
        pins,
        state,
        options,
    )


@app.function(
    image=cpu_image.env({"EPISODE_FACADE_READ_ONLY": "1"}),
    secrets=[secrets],
)
@modal.asgi_app()
def read_facade():
    """The hosted viewer's backend: journal reads over the R2 store, no moves."""
    from nof1_causal_lab.read_facade import create_read_facade_app

    return create_read_facade_app()


@app.function(timeout=3600, cpu=4, memory=8192, secrets=[secrets])
async def _run_transition_cpu(
    workspace_id: str,
    artifact_id: str,
    pins: dict[str, int],
    state: JsonObject,
    options: JsonObject,
) -> JsonObject:
    """Run a transition on Modal CPU compute against the R2 artifact store."""
    return await _run_transition_remote(
        workspace_id,
        artifact_id,
        pins,
        state,
        options,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Runner callables (bound by machine.runners)
# ═══════════════════════════════════════════════════════════════════════════════

_GPU_TRANSITIONS = frozenset({"posterior"})


async def run_transition_on_modal(
    workspace_id: str,
    artifact_id: ArtifactId,
    pins: dict[ArtifactId, int],
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Invoke a transition remotely; credentials come from the Modal secret block."""
    from nof1_causal_lab.machine.moves import TransitionEffects

    remote_fn = _run_transition_gpu if artifact_id in _GPU_TRANSITIONS else _run_transition_cpu
    raw = await remote_fn.remote.aio(
        workspace_id,
        artifact_id,
        dict(pins),
        state.model_dump(mode="json"),
        options.model_dump(mode="json"),
    )
    return TransitionEffects.model_validate(raw)
