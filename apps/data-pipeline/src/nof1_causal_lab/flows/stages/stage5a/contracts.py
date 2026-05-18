"""Stage 5a contracts."""

from __future__ import annotations

from nof1_causal_lab.flows.contracts_base import BaseStageContract
from nof1_causal_lab.flows.stages.inference_contracts import (  # noqa: TC001
    InferenceMetadataContract,
)
from nof1_causal_lab.models.ssm.inference.schemas import (  # noqa: TC001
    PosteriorMarginal,
    PosteriorPair,
    SVIDiagnostics,
)

STAGE_ID = "stage-5a"
IS_INTERACTIVE_STAGE = False


class Stage5aContract(BaseStageContract):
    """SVI preflight: fast approximate fit before expensive inference."""

    inference_metadata: InferenceMetadataContract
    svi_diagnostics: SVIDiagnostics | None = None
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None

    def summary_message(self) -> str:
        converged = self.svi_diagnostics is not None
        return f"Stage 5a summary: method=svi converged={converged} outcome={self.outcome}"
