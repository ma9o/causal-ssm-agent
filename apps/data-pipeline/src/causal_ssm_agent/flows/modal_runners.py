# ruff: noqa: ARG001
"""Modal-backed runners for expensive pipeline stages.

When DEPLOYMENT_ENV=production, the stage registry swaps local runners for
these Modal-backed versions. Stages 4 and 5b run on Modal's compute; the
remaining stages run locally on Prefect.
"""

from __future__ import annotations

from pathlib import Path

import modal

# ═══════════════════════════════════════════════════════════════════════════════
# Modal image (CPU-only)
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[3]  # -> apps/data-pipeline/

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev"], frozen=True)
    .env({"PYTHONPATH": "/root/src"})
    .add_local_file(ROOT / "config.yaml", remote_path="/root/config.yaml")
    .add_local_file(ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
    .add_local_dir(ROOT / "src" / "causal_ssm_agent", remote_path="/root/src/causal_ssm_agent")
)

app = modal.App("causal-ssm-pipeline", image=image)
secrets = modal.Secret.from_name("causal-ssm-pipeline-secrets")


# ═══════════════════════════════════════════════════════════════════════════════
# Remote functions
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(timeout=3600, cpu=8, memory=16384, secrets=[secrets])
def _run_stage5b(
    stage4: dict,
    stage2: dict,
    inference_method: str | None,
    user_id: str,
) -> dict:
    """Run stage 5b on Modal and persist artifacts to R2."""
    from causal_ssm_agent.flows.dag import stage5b
    from causal_ssm_agent.flows.stage_registry import _persist_stage5b

    result = stage5b(stage4, stage2, inference_method)
    return _persist_stage5b(result, user_id)


@app.function(timeout=3600, cpu=4, memory=8192, secrets=[secrets])
async def _run_stage4(
    question: str,
    stage1b: dict,
    stage2: dict,
    stage3: dict,
    enable_literature: bool,
) -> dict:
    """Run stage 4 on Modal."""
    from causal_ssm_agent.flows.dag import stage4

    return await stage4(question, stage1b, stage2, stage3, enable_literature)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner callables (bound by the registry)
# ═══════════════════════════════════════════════════════════════════════════════


def modal_stage5b_runner(
    stage4: dict,
    stage2: dict,
    inference_method: str | None,
    user_id: str = "",
) -> dict:
    """Invoke stage 5b on Modal."""
    return _run_stage5b.remote(stage4, stage2, inference_method, user_id)


async def modal_stage4_runner(
    question: str,
    stage1b: dict,
    stage2: dict,
    stage3: dict,
    enable_literature: bool,
) -> dict:
    """Invoke stage 4 on Modal."""
    return await _run_stage4.remote.aio(question, stage1b, stage2, stage3, enable_literature)


# ═══════════════════════════════════════════════════════════════════════════════
# No-op persist (Modal already persisted artifacts before returning)
# ═══════════════════════════════════════════════════════════════════════════════


def persist_noop(result: dict, user_id: str) -> dict:
    """Pass through — artifacts were already persisted inside Modal."""
    return result
