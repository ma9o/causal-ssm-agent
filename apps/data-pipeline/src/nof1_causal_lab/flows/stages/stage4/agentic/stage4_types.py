"""Shared Stage 4 runtime types used across the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from nof1_causal_lab.flows.stages.stage4.assembly import AssemblyValidation

    from .stage4_feedback import Stage4GroundingResult
    from .stage4_skeleton import Stage4Skeleton


@dataclass
class Stage4Result:
    """Result of the agentic Stage 4 flow."""

    model_spec: dict[str, Any]
    authored_priors: dict[str, dict]
    search_queries: dict[str, str] = field(default_factory=dict)
    validation: AssemblyValidation | None = None


@dataclass(frozen=True)
class Stage4Deps:
    """Static Stage 4 runtime dependencies shared across reducer steps."""

    skeleton: Stage4Skeleton
    causal_spec: dict[str, Any]
    data_for_model: pl.DataFrame
    indicator_audits: dict[str, dict[str, Any]]
    grounding_fn: Callable[..., Stage4GroundingResult]
