"""Stage 4b contracts."""

from __future__ import annotations

from nof1_causal_lab.flows.contracts_base import BaseStageContract
from nof1_causal_lab.models.ssm.inference.schemas import (  # noqa: TC001
    InferenceStructureResult,
    ParametricIdResult,
)

STAGE_ID = "stage-4b"
IS_INTERACTIVE_STAGE = False


class Stage4bContract(BaseStageContract):
    parametric_id: ParametricIdResult
    inference_structure: InferenceStructureResult | None = None

    def summary_message(self) -> str:
        pid = self.parametric_id
        summary = pid.summary
        return (
            f"Stage 4b summary: checked={pid.checked} "
            f"structural_issues={len(summary.structural_issues if summary else [])} "
            f"boundary_issues={len(summary.boundary_issues if summary else [])} "
            f"weak_params={len(summary.weak_params if summary else [])} outcome={self.outcome}"
        )
