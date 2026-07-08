"""posterior contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.contracts_base import BaseArtifactContract
from nof1_causal_lab.flows.transitions.inference_contracts import (  # noqa: TC001
    InferenceMetadataContract,
)
from nof1_causal_lab.models.posterior_predictive import (  # noqa: TC001
    PPCOverlay,
    PPCTestStat,
    PPCWarning,
)
from nof1_causal_lab.models.ssm.inference.schemas import (  # noqa: TC001
    LOODiagnostics,
    MCMCDiagnostics,
    PosteriorMarginal,
    PosteriorPair,
    SMCDiagnostics,
)

IS_INTERACTIVE_CONTEXT = False


class PPCResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_variable_warnings: list[PPCWarning]
    checked: bool | None = None
    n_subsample: int | None = None
    overlays: list[PPCOverlay] = Field(default_factory=list)
    test_stats: list[PPCTestStat] = Field(default_factory=list)


class PosteriorContract(BaseArtifactContract):
    ppc: PPCResultContract
    inference_metadata: InferenceMetadataContract
    mcmc_diagnostics: MCMCDiagnostics | None = None
    smc_diagnostics: SMCDiagnostics | None = None
    loo_diagnostics: LOODiagnostics | None = None
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None
