"""Modal-backed runners for expensive pipeline stages.

When DEPLOYMENT_ENV=production, the stage registry swaps local runners for
these Modal-backed versions. Stages 4 and 5b run on Modal's compute; the
remaining stages run locally on Prefect.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from .stage_contracts import BaseStageContract

# ═══════════════════════════════════════════════════════════════════════════════
# Modal images
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[3]  # -> apps/data-pipeline/
GPU_A100_80GB = modal.gpu.A100(size="80GB")

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev", "cloud"], frozen=True)
    .env({"PYTHONPATH": "/root/src"})
    .add_local_file(ROOT / "config.yaml", remote_path="/root/config.yaml")
    .add_local_file(ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
    .add_local_dir(ROOT / "src" / "causal_ssm_agent", remote_path="/root/src/causal_ssm_agent")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev", "cloud"], frozen=True)
    .uv_pip_install("jax[cuda12]", gpu=GPU_A100_80GB)
    .env({"PYTHONPATH": "/root/src"})
    .add_local_file(ROOT / "config.yaml", remote_path="/root/config.yaml")
    .add_local_file(ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
    .add_local_dir(ROOT / "src" / "causal_ssm_agent", remote_path="/root/src/causal_ssm_agent")
)

app = modal.App("causal-ssm-pipeline", image=cpu_image)
secrets = modal.Secret.from_name("causal-ssm-pipeline-secrets")


# ═══════════════════════════════════════════════════════════════════════════════
# Remote functions
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(
    timeout=3600,
    cpu=8,
    memory=32768,
    image=gpu_image,
    gpu=GPU_A100_80GB,
    secrets=[secrets],
)
def _run_stage5b(
    stage4: BaseStageContract,
    stage2: BaseStageContract,
    inference_method: str | None,
    workspace_id: str,
) -> BaseStageContract:
    """Run stage 5b on Modal and persist artifacts to R2."""
    from causal_ssm_agent.flows.dag import stage5b

    return stage5b(stage4, stage2, workspace_id, inference_method)


@app.function(timeout=3600, cpu=4, memory=8192, secrets=[secrets])
async def _run_stage4(
    question: str,
    stage1b: BaseStageContract,
    stage2: BaseStageContract,
    stage3: BaseStageContract,
    enable_literature: bool,
    workspace_id: str,
    openrouter_api_key: str | None,
    root_run_id: str | None,
) -> BaseStageContract:
    """Run stage 4 on Modal."""
    from causal_ssm_agent.flows.dag import stage4

    return await stage4(
        question,
        stage1b,
        stage2,
        stage3,
        enable_literature,
        workspace_id=workspace_id,
        openrouter_api_key=openrouter_api_key,
        root_run_id=root_run_id,
    )


@app.function(
    timeout=3600,
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
    from causal_ssm_agent.flows.stage4_compile_cache import warm_stage4_compile_cache_artifact

    return warm_stage4_compile_cache_artifact(
        workspace_id=workspace_id,
        model_spec=model_spec,
        causal_spec=causal_spec,
        topology_fingerprint=topology_fingerprint,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Runner callables (bound by the registry)
# ═══════════════════════════════════════════════════════════════════════════════


def modal_stage5b_runner(
    stage4: BaseStageContract,
    stage2: BaseStageContract,
    workspace_id: str = "",
    inference_method: str | None = None,
) -> BaseStageContract:
    """Invoke stage 5b on Modal."""
    return _run_stage5b.remote(stage4, stage2, inference_method, workspace_id)


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


async def modal_stage4_runner(
    question: str,
    stage1b: BaseStageContract,
    stage2: BaseStageContract,
    stage3: BaseStageContract,
    enable_literature: bool,
    workspace_id: str,
    root_run_id: str | None = None,
) -> BaseStageContract:
    """Invoke stage 4 on Modal."""
    from causal_ssm_agent.utils.openrouter_client import get_openrouter_api_key

    return await _run_stage4.remote.aio(
        question,
        stage1b,
        stage2,
        stage3,
        enable_literature,
        workspace_id,
        get_openrouter_api_key(),
        root_run_id,
    )
