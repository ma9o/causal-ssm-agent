"""Run slow (GPU) tests on Modal with a T4.

Usage:
    modal run ci.py            # run slow tests
    modal run ci.py --all      # run full test suite
    modal shell ci.py          # interactive debug shell
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    # Sync all project + dev deps from lockfile
    .uv_sync(
        uv_project_dir=str(ROOT),
        groups=["dev"],
        frozen=True,
    )
    # Install CUDA plugin on top of the locked jax version
    .uv_pip_install("jax[cuda12]", gpu="A100")
    # Config + pytest settings needed at collection time
    .env({"PYTHONPATH": "/root/src"})
    .add_local_file(ROOT / "config.yaml", remote_path="/root/config.yaml")
    .add_local_file(ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
    # Mount source code and tests last (fast re-mount, no rebuild)
    .add_local_dir(ROOT / "src" / "causal_ssm_agent", remote_path="/root/src/causal_ssm_agent")
    .add_local_dir(ROOT / "tests", remote_path="/root/tests")
)

app = modal.App("causal-ssm-gpu-tests", image=image)


@app.function(gpu="A100", timeout=1800)
def run_tests(all_tests: bool = False):
    """Run pytest on a remote GPU."""
    import subprocess

    cmd = ["pytest", "tests/", "-v", "--tb=short", "-x"]
    if not all_tests:
        cmd.extend(["-m", "slow"])

    result = subprocess.run(cmd, cwd="/root", check=False)
    return result.returncode


@app.local_entrypoint()
def main(all: bool = False):
    exit_code = run_tests.remote(all_tests=all)
    if exit_code != 0:
        raise SystemExit(exit_code)
