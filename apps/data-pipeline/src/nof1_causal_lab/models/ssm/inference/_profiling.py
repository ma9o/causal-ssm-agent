"""Optional, flag-gated profiling for inference run loops.

Off by default. Enable by passing ``profile_dir=...`` to ``fit`` (or by setting
the ``NOF1_PROFILE_DIR`` environment variable). When enabled, the run loop is
wrapped in a ``jax.profiler`` trace — a TensorBoard/Perfetto timeline, legible by
the ``jax.named_scope`` spans the smoothers already carry — and the compiled
step's HLO and cost analysis are dumped.

The HLO + cost dump is the *access-pattern* surface that a FLOP count cannot
show: aggregate bytes-vs-FLOPs (bandwidth- vs compute-bound), and the
``transpose`` / ``gather`` / ``dynamic-slice`` / ``copy`` data-movement ops plus
the ``triangular-solve`` / ``dot`` tiny-linear-algebra and ``while`` (scan)
serialization that dominate at these shapes. ``ncu`` / ``nsys`` on device read
the same op metadata for coalescing / DRAM detail once the HLO points at the
fusion cluster worth that scrutiny.

Placed at the inference-package level (not inside a single method) so the toggle
sits above the method/smoother dispatch and generalizes for free.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import jax

_PROFILE_DIR_ENV = "NOF1_PROFILE_DIR"

# HLO ops whose counts characterize the device access pattern:
#   data movement      — transpose / copy / bitcast / reshape / broadcast /
#                         concatenate / slice
#   indexed access     — gather / scatter / dynamic-slice / dynamic-update-slice
#   tiny linear algebra — dot / triangular-solve / cholesky (the D=2-3 offenders)
#   fused compute      — fusion
#   serialization      — while (the lax.scan forward/backward filters)
_ACCESS_PATTERN_OPS = (
    "fusion",
    "dot",
    "triangular-solve",
    "cholesky",
    "transpose",
    "copy",
    "bitcast",
    "reshape",
    "broadcast",
    "concatenate",
    "slice",
    "gather",
    "scatter",
    "dynamic-slice",
    "dynamic-update-slice",
    "reduce",
    "while",
    "conditional",
    "sort",
    "custom-call",
)


def resolve_profile_dir(profile_dir: str | os.PathLike[str] | None) -> Path | None:
    """Resolve the effective profile directory, or ``None`` when profiling is off.

    ``profile_dir`` takes precedence; otherwise ``NOF1_PROFILE_DIR`` is consulted.
    The directory is created if it does not exist.
    """
    raw = profile_dir if profile_dir is not None else os.environ.get(_PROFILE_DIR_ENV)
    if raw is None or str(raw) == "":
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def start_trace(profile_dir: Path | None, *, label: str) -> None:
    """Begin a ``jax.profiler`` trace into ``profile_dir / label`` (no-op if None)."""
    if profile_dir is None:
        return
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    options.include_dataset_ops = False
    options.enable_hlo_proto = False
    jax.profiler.start_trace(str(profile_dir / label), profiler_options=options)


def stop_trace(profile_dir: Path | None) -> None:
    """Stop the active ``jax.profiler`` trace (no-op if None)."""
    if profile_dir is None:
        return
    jax.profiler.stop_trace()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return repr(value)


def _summarize_access_patterns(hlo_text: str) -> dict[str, int]:
    """Count producer instructions per op in the optimized HLO.

    Matches ``<op>(`` at a call site (operand references like ``%transpose.5``
    carry no paren, so they are not counted). A heuristic histogram, not a parse.
    """
    return {op: len(re.findall(rf"\b{re.escape(op)}\(", hlo_text)) for op in _ACCESS_PATTERN_OPS}


def dump_compiled_analysis(
    jitted_fn: Any,
    *call_args: Any,
    profile_dir: Path | None,
    label: str,
    **call_kwargs: Any,
) -> None:
    """Lower + compile ``jitted_fn`` for one call and dump its HLO + cost analysis.

    Writes, under ``profile_dir``:
      * ``<label>.hlo.txt`` — optimized HLO (fusion clusters and data-movement ops),
      * ``<label>.cost.json`` — aggregate FLOPs / bytes-accessed,
      * ``<label>.access_patterns.json`` — op histogram from ``_ACCESS_PATTERN_OPS``.

    No-op when ``profile_dir`` is None. This compiles once more than the run loop
    would; on an opt-in profiling path (and with the persistent compilation cache)
    that is a one-time cost.
    """
    if profile_dir is None:
        return
    compiled = jitted_fn.lower(*call_args, **call_kwargs).compile()
    hlo_text = compiled.as_text()
    (profile_dir / f"{label}.hlo.txt").write_text(hlo_text)
    (profile_dir / f"{label}.cost.json").write_text(
        json.dumps(_json_ready(compiled.cost_analysis()), indent=2, sort_keys=True)
    )
    (profile_dir / f"{label}.access_patterns.json").write_text(
        json.dumps(_summarize_access_patterns(hlo_text), indent=2, sort_keys=True)
    )
