"""Modal-backed execution for expensive stages.

When ``DEPLOYMENT_ENV=production``, ``machine.runners.execute_stage``
routes stage-4 and stage-5b here. The remote function runs the same
``execute_stage_locally`` against the same R2-backed artifact store —
Modal is compute placement, not a different execution path. Version
stamps come back as plain dicts (Modal pickles across an image boundary).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId
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


@app.function(
    timeout=10800,
    cpu=8,
    memory=32768,
    image=gpu_image,
    gpu=GPU_A100_80GB,
    secrets=[secrets],
)
async def _run_stage_gpu(
    workspace_id: str,
    stage_id: str,
    pins: dict[str, int],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Run a stage on Modal GPU compute against the R2 artifact store."""
    from nof1_causal_lab.machine.moves import ExecOptions
    from nof1_causal_lab.machine.runners import execute_stage_locally

    result = await execute_stage_locally(
        workspace_id,
        stage_id,
        pins,  # type: ignore[arg-type]
        ExecOptions.model_validate(options),
    )
    return result.model_dump(mode="json")


@app.function(timeout=3600, cpu=4, memory=8192, secrets=[secrets])
async def _run_stage_cpu(
    workspace_id: str,
    stage_id: str,
    pins: dict[str, int],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Run a stage on Modal CPU compute against the R2 artifact store."""
    from nof1_causal_lab.machine.moves import ExecOptions
    from nof1_causal_lab.machine.runners import execute_stage_locally

    result = await execute_stage_locally(
        workspace_id,
        stage_id,
        pins,  # type: ignore[arg-type]
        ExecOptions.model_validate(options),
    )
    return result.model_dump(mode="json")


@app.function(
    timeout=10800,
    cpu=8,
    memory=32768,
    image=gpu_image,
    gpu=GPU_A100_80GB,
    secrets=[secrets],
)
def _warm_stage4_compile_cache(
    workspace_id: str,
    model_spec: dict,
    causal_spec: dict | None,
    topology_fingerprint: str,
) -> dict:
    """Warm the topology-matched JAX compile cache on Modal A100 and persist it."""
    from nof1_causal_lab.flows.stage4_compile_cache import warm_stage4_compile_cache_artifact

    return warm_stage4_compile_cache_artifact(
        workspace_id=workspace_id,
        model_spec=model_spec,
        causal_spec=causal_spec,
        topology_fingerprint=topology_fingerprint,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Runner callables (bound by machine.runners)
# ═══════════════════════════════════════════════════════════════════════════════

_GPU_STAGES = frozenset({"stage-5b"})


async def run_stage_on_modal(
    workspace_id: str,
    stage_id: str,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> TransitionEffects:
    """Invoke a stage remotely; credentials come from the Modal secret block."""
    from nof1_causal_lab.machine.moves import TransitionEffects

    remote_fn = _run_stage_gpu if stage_id in _GPU_STAGES else _run_stage_cpu
    raw = await remote_fn.remote.aio(
        workspace_id,
        stage_id,
        dict(pins),
        options.model_dump(mode="json"),
    )
    return TransitionEffects.model_validate(raw)


def spawn_stage4_model_compile_warmup(
    *,
    workspace_id: str,
    model_spec: dict,
    causal_spec: dict | None,
    topology_fingerprint: str,
) -> str:
    """Spawn the Stage 4 compile-cache warmup job and return its FunctionCall id."""
    function_call = _warm_stage4_compile_cache.spawn(
        workspace_id,
        model_spec,
        causal_spec,
        topology_fingerprint,
    )
    return str(function_call.object_id)
