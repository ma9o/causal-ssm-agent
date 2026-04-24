"""Stage 5b contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from causal_ssm_agent.flows.contracts_base import BaseStageContract
from causal_ssm_agent.flows.stages.inference_contracts import (  # noqa: TC001
    InferenceMetadataContract,
)
from causal_ssm_agent.models.posterior_predictive import (  # noqa: TC001
    PPCOverlay,
    PPCTestStat,
    PPCWarning,
)
from causal_ssm_agent.models.ssm.inference.schemas import (  # noqa: TC001
    LOODiagnostics,
    MCMCDiagnostics,
    PosteriorMarginal,
    PosteriorPair,
    SMCDiagnostics,
    SVIDiagnostics,
)

STAGE_ID = "stage-5b"
IS_INTERACTIVE_STAGE = False


class PowerScalingResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: str
    diagnosis: Literal["prior_dominated", "well_identified", "prior_data_conflict"]
    prior_sensitivity: float
    likelihood_sensitivity: float
    psis_k_hat: float | None = None


class PPCResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_variable_warnings: list[PPCWarning]
    checked: bool | None = None
    n_subsample: int | None = None
    overlays: list[PPCOverlay] = Field(default_factory=list)
    test_stats: list[PPCTestStat] = Field(default_factory=list)


class Stage5bContract(BaseStageContract):
    power_scaling: list[PowerScalingResultContract]
    ppc: PPCResultContract
    inference_metadata: InferenceMetadataContract
    mcmc_diagnostics: MCMCDiagnostics | None = None
    svi_diagnostics: SVIDiagnostics | None = None
    smc_diagnostics: SMCDiagnostics | None = None
    loo_diagnostics: LOODiagnostics | None = None
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None

    def summary_message(self) -> str:
        ps_issues = sum(
            1
            for item in self.power_scaling
            if item.diagnosis in {"prior_dominated", "prior_data_conflict"}
        )
        ppc_warnings = len(self.ppc.per_variable_warnings)
        return (
            f"Stage 5b summary: method={self.inference_metadata.method} "
            f"samples={self.inference_metadata.n_samples} "
            f"power_scaling_issues={ps_issues} ppc_warnings={ppc_warnings} outcome={self.outcome}"
        )
