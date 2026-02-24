"""Shared Modal GPU infrastructure for recovery benchmarks."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root


def make_modal_image(gpu: str):
    """Build a Modal image for JAX + CUDA with the project installed."""
    import modal

    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("uv")
        .uv_sync(uv_project_dir=str(ROOT), groups=["dev"], frozen=True)
        .uv_pip_install("jax[cuda12]", gpu=gpu)
        .env({"PYTHONPATH": "/root"})
        .add_local_dir(ROOT / "src" / "causal_ssm_agent", remote_path="/root/causal_ssm_agent")
        .add_local_dir(ROOT / "benchmarks", remote_path="/root/benchmarks")
    )


def make_modal_app(name: str, gpu: str):
    """Create a Modal app with a GPU image.

    Returns (app, gpu_type) tuple.
    """
    import modal

    image = make_modal_image(gpu)
    app = modal.App(name, image=image)
    return app, gpu
