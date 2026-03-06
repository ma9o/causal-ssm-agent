"""Executable contracts for stage payloads persisted to the web layer.

These schemas are the single runtime source of truth for stage JSON written by
``persist_web_result``. Any contract drift fails immediately at persistence time.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from causal_ssm_agent.models.posterior_predictive import (  # noqa: TC001
    PPCOverlay,
    PPCTestStat,
    PPCWarning,
)
from causal_ssm_agent.models.ssm.schemas_inference import (  # noqa: TC001
    LOODiagnostics,
    MCMCDiagnostics,
    ParametricIdResult,
    PosteriorMarginal,
    PosteriorPair,
    RBPartitionResult,
    SVIDiagnostics,
    TemporalEffect,
)
from causal_ssm_agent.orchestrator.schemas import (  # noqa: TC001
    CausalSpec,
    LatentModel,
)
from causal_ssm_agent.orchestrator.schemas_model import ModelSpec  # noqa: TC001
from causal_ssm_agent.utils.llm import LLMTrace  # noqa: TC001
from causal_ssm_agent.workers.schemas_prior import PriorProposal  # noqa: TC001

StageId = Literal[
    "stage-0",
    "stage-1a",
    "stage-1b",
    "stage-2",
    "stage-3",
    "stage-4",
    "stage-4b",
    "stage-5",
    "stage-6",
]


class GateOverrideContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


class DateRangeContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    end: str


class ColumnDescriptionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    description: str


class Stage0Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    source_label: str
    n_records: int
    n_columns: int
    date_range: DateRangeContract
    sample: list[dict[str, str | None]]
    column_descriptions: list[ColumnDescriptionContract]
    llm_trace: LLMTrace | None = None


class Stage1aContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    latent_model: LatentModel
    outcome_name: str
    treatments: list[str]
    llm_trace: LLMTrace | None = None


class Stage1bContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    causal_spec: CausalSpec
    llm_trace: LLMTrace | None = None
    gate_overridden: GateOverrideContract | None = None


class WorkerStatusContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: int
    status: Literal["pending", "running", "completed", "failed"]
    n_extractions: int
    chunk_size: int
    error: str | None = None


class ExtractionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator: str
    value: int | float | bool | str | None
    timestamp: str | None


class Stage2Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    workers: list[WorkerStatusContract]
    combined_extractions_sample: list[ExtractionContract]
    per_indicator_counts: dict[str, int]


class ValidationIssueContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator: str
    issue_type: str
    severity: Literal["error", "warning", "info"]
    message: str


class IndicatorHealthContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator: str
    n_obs: int
    variance: float | None
    time_coverage_ratio: float | None
    max_gap_ratio: float | None
    dtype_violations: int
    duplicate_pct: float
    arithmetic_sequence_detected: bool
    cell_statuses: dict[str, Literal["ok", "warning", "error"]]


class ValidationReportContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    issues: list[ValidationIssueContract]
    per_indicator_health: list[IndicatorHealthContract]


class Stage3Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    validation_report: ValidationReportContract


class ValidationRetryContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int
    failed_params: list[str]
    feedback: str


class Stage4Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    model_spec: ModelSpec
    priors: list[PriorProposal]
    validation_retries: list[ValidationRetryContract] | None = None
    llm_trace: LLMTrace | None = None
    prior_predictive_samples: dict[str, list[float]] | None = None


class Stage4bContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    parametric_id: ParametricIdResult
    rb_partition: RBPartitionResult | None = None
    gate_overridden: GateOverrideContract | None = None


class TreatmentEffectContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treatment: str
    effect_size: float | None
    posterior_draws: list[float] | None = None
    prob_positive: float | None = None
    identifiable: bool
    ppc_warnings: list[str] | None = None
    prior_sensitivity_warning: str | None = None
    temporal: TemporalEffect | None = None
    manifest_effects: dict[str, float] | None = None


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


class InferenceMetadataContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    n_samples: int
    duration_seconds: float


class Stage5Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    power_scaling: list[PowerScalingResultContract]
    ppc: PPCResultContract
    inference_metadata: InferenceMetadataContract
    mcmc_diagnostics: MCMCDiagnostics | None = None
    svi_diagnostics: SVIDiagnostics | None = None
    loo_diagnostics: LOODiagnostics | None = None
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None


class Stage6Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    intervention_results: list[TreatmentEffectContract]


class LiveMetadata(BaseModel):
    """Metadata attached to partial stage results while an LLM stage is running."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["running"]
    label: str
    turn: int
    elapsed_seconds: float


class PartialStageResult(BaseModel):
    """Partial stage result written to disk during LLM generation.

    A subset of the full stage contract: only the ``llm_trace`` field (the part
    available mid-run) plus ``_live`` metadata so the frontend can distinguish
    in-progress from completed results.  Overwritten by ``persist_web_result``
    when the stage completes.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    llm_trace: LLMTrace
    live: LiveMetadata = Field(alias="_live")


STAGE_CONTRACTS: dict[StageId, type[BaseModel]] = {
    "stage-0": Stage0Contract,
    "stage-1a": Stage1aContract,
    "stage-1b": Stage1bContract,
    "stage-2": Stage2Contract,
    "stage-3": Stage3Contract,
    "stage-4": Stage4Contract,
    "stage-4b": Stage4bContract,
    "stage-5": Stage5Contract,
    "stage-6": Stage6Contract,
}


def validate_stage_payload(stage_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Validate stage payload by stage id and return a JSON-serializable dict."""
    if stage_id not in STAGE_CONTRACTS:
        known = ", ".join(sorted(STAGE_CONTRACTS.keys()))
        raise ValueError(f"Unknown stage_id '{stage_id}'. Expected one of: {known}")
    # After the membership check, stage_id is guaranteed to be a valid StageId
    sid = cast("StageId", stage_id)
    model = STAGE_CONTRACTS[sid].model_validate(data)
    return model.model_dump(mode="json")
