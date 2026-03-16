"""Package initialization for the causal SSM agent."""

from __future__ import annotations

import os
from pathlib import Path


def _truthy_env(name: str) -> bool:
    """Return True when an environment flag is set to a truthy value."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _configure_jax_persistent_cache() -> None:
    """Enable JAX's persistent compilation cache unless explicitly disabled."""
    if _truthy_env("CAUSAL_SSM_DISABLE_JAX_PERSISTENT_CACHE"):
        return

    try:
        import jax
    except Exception:
        return

    cache_dir = os.getenv("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        cache_dir = str(Path.home() / ".cache" / "causal-ssm-agent" / "jax")

    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        if not jax.config.values.get("jax_compilation_cache_dir"):
            jax.config.update("jax_compilation_cache_dir", cache_dir)
    except Exception:
        # Cache configuration is an optimization only; it must never block imports.
        return


_configure_jax_persistent_cache()
