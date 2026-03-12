"""Executable contracts for stage payloads persisted to the web layer.

These schemas are the single runtime source of truth for stage JSON written by
``persist_web_result``. Any contract drift fails immediately at persistence time.

Also defines tool contracts — declarative metadata for every tool available to
pipeline stages. These feed into TypeScript codegen (Zod schemas + tool defs)
and the refinement proxy (same tool schemas the pipeline used).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    SMCDiagnostics,
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
    "stage-5a",
    "stage-5b",
    "stage-6",
]


# ---------------------------------------------------------------------------
# Tool contracts — declarative metadata for pipeline tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolContract:
    """Declarative tool definition shared between pipeline, codegen, and refinement proxy.

    The ``input_schema`` Pydantic model mirrors the tool's execute function
    signature. Codegen uses ``parameters_json_schema()`` to produce matching
    Zod schemas on the TypeScript side.
    """

    name: str
    description: str
    input_schema: type[BaseModel]

    def parameters_json_schema(self) -> dict[str, Any]:
        """Generate JSON Schema for the tool's input parameters."""
        schema = self.input_schema.model_json_schema()
        schema["additionalProperties"] = False
        return schema


# --- Stage 0 tool inputs ---


class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", description="Relative path within the input directory.")


class ReadFileSampleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Relative path to the file within the input directory.")
    n_lines: int = Field(default=50, description="Number of lines to read.")


class ExecutePythonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Python code to execute.")


class SubmitTableInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_label: str = Field(description="A short human-readable label for the data source.")
    column_descriptions_json: str = Field(
        description="JSON object mapping column names to descriptions."
    )


# --- Stage 1a tool inputs ---


class ValidateLatentModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure_json: str = Field(
        description="The JSON string containing the latent model to validate."
    )


# --- Stage 1b tool inputs ---


class ValidateMeasurementModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_json: str = Field(
        description="The JSON string containing the measurement model to validate."
    )


# --- Stage 2 tool inputs ---


class ValidateExtractionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_json: str = Field(
        description="The JSON string containing the worker output to validate."
    )


# --- Stage 4 tool inputs ---


class SearchLiteratureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query for empirical literature about effect sizes.")


class ValidateModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_json: str = Field(
        description=(
            "JSON object with proposed changes. Include 'model_spec' (complete ModelSpec) "
            "and/or 'priors' (dict mapping parameter names to prior proposals). "
            "Only include fields you are changing."
        ),
    )


# --- Stage tools registry ---

STAGE_TOOLS: dict[str, list[ToolContract]] = {
    "stage-0": [
        ToolContract(
            name="list_files",
            description="List files in the prepared input directory.",
            input_schema=ListFilesInput,
        ),
        ToolContract(
            name="read_file_sample",
            description="Read a sample of lines from a file to understand its format.",
            input_schema=ReadFileSampleInput,
        ),
        ToolContract(
            name="execute_python",
            description="Execute Python code in a Modal sandbox to parse files into a Polars DataFrame.",
            input_schema=ExecutePythonInput,
        ),
        ToolContract(
            name="submit_table",
            description="Validate and finalize the ingested DataFrame with column descriptions.",
            input_schema=SubmitTableInput,
        ),
    ],
    "stage-1a": [
        ToolContract(
            name="validate_latent_model_tool",
            description="Tool for validating latent model JSON (Stage 1a).",
            input_schema=ValidateLatentModelInput,
        ),
    ],
    "stage-1b": [
        ToolContract(
            name="validate_measurement_model_tool",
            description="Validate measurement model JSON, check compiler constraints, and verify causal identifiability.",
            input_schema=ValidateMeasurementModelInput,
        ),
    ],
    "stage-2": [
        ToolContract(
            name="validate_extractions",
            description="Tool for validating worker extraction output JSON.",
            input_schema=ValidateExtractionsInput,
        ),
    ],
    "stage-4": [
        ToolContract(
            name="search_literature",
            description="Search for empirical literature about effect sizes for model parameters.",
            input_schema=SearchLiteratureInput,
        ),
        ToolContract(
            name="validate_model",
            description="Validate model specification and/or prior proposals: schema check, compile, prior predictive simulation.",
            input_schema=ValidateModelInput,
        ),
    ],
}

# Stages with an interactive LLM trace panel (refinement chat + replay).
# Stage 0 is excluded (tools depend on sandbox/filesystem state).
# Stage 2 is excluded (parallel worker extraction, not a single LLM conversation).
INTERACTIVE_STAGES: frozenset[str] = frozenset({"stage-1a", "stage-1b", "stage-4"})


# ---------------------------------------------------------------------------
# Stage output contracts
# ---------------------------------------------------------------------------


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

    def summarize(self) -> tuple[int, str]:
        return (
            logging.INFO,
            f"Stage 0 summary: source={self.source_label} "
            f"records={self.n_records} columns={self.n_columns} "
            f"date_range={self.date_range.start}..{self.date_range.end}",
        )


class Stage1aContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    latent_model: LatentModel
    outcome_name: str
    treatments: list[str]
    llm_trace: LLMTrace | None = None

    def summarize(self) -> tuple[int, str]:
        return (
            logging.INFO,
            f"Stage 1a summary: constructs={len(self.latent_model.constructs)} "
            f"edges={len(self.latent_model.edges)} "
            f"treatments={len(self.treatments)} "
            f"outcome={self.outcome_name or 'unknown'}",
        )


class Stage1bContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    causal_spec: CausalSpec
    llm_trace: LLMTrace | None = None
    gate_overridden: GateOverrideContract | None = None

    def summarize(self) -> tuple[int, str]:
        non_id = (
            self.causal_spec.identifiability.non_identifiable_treatments
            if self.causal_spec.identifiability
            else {}
        ) or {}
        return (
            logging.WARNING if self.outcome in {"warn", "fail"} else logging.INFO,
            f"Stage 1b summary: constructs={len(self.causal_spec.latent.constructs)} "
            f"indicators={len(self.causal_spec.measurement.indicators)} "
            f"filtered_treatments={len(non_id)} outcome={self.outcome}",
        )


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
    llm_trace: LLMTrace | None = None

    def summarize(self) -> tuple[int, str]:
        completed = sum(1 for w in self.workers if w.status == "completed")
        failed = sum(1 for w in self.workers if w.status == "failed")
        return (
            logging.WARNING if self.outcome in {"warn", "fail"} else logging.INFO,
            f"Stage 2 summary: workers={len(self.workers)} completed={completed} "
            f"failed={failed} sample_rows={len(self.combined_extractions_sample)} "
            f"indicators={len(self.per_indicator_counts)} outcome={self.outcome}",
        )


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

    def summarize(self) -> tuple[int, str]:
        rpt = self.validation_report
        errors = sum(1 for i in rpt.issues if i.severity == "error")
        warnings = sum(1 for i in rpt.issues if i.severity == "warning")
        return (
            logging.WARNING if self.outcome in {"warn", "fail"} else logging.INFO,
            f"Stage 3 summary: is_valid={rpt.is_valid} "
            f"issues={len(rpt.issues)} "
            f"errors={errors} warnings={warnings} outcome={self.outcome}",
        )


class ValidationRetryContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int
    failed_params: list[str]
    feedback: str


class Stage4Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    model_spec: ModelSpec
    priors: dict[str, PriorProposal]
    validation_retries: list[ValidationRetryContract] | None = None
    llm_trace: LLMTrace | None = None
    prior_predictive_samples: dict[str, list[float]] | None = None

    def summarize(self) -> tuple[int, str]:
        return (
            logging.INFO,
            f"Stage 4 summary: parameters={len(self.model_spec.parameters)} "
            f"likelihoods={len(self.model_spec.likelihoods)} "
            f"priors={len(self.priors)} "
            f"prior_predictive_channels={len(self.prior_predictive_samples or {})}",
        )


class Stage4bContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    parametric_id: ParametricIdResult
    rb_partition: RBPartitionResult | None = None
    gate_overridden: GateOverrideContract | None = None

    def summarize(self) -> tuple[int, str]:
        pid = self.parametric_id
        t_pass = "pass" if (pid.t_rule is None or pid.t_rule.satisfies) else "fail"
        s = pid.summary
        return (
            logging.WARNING if self.outcome in {"warn", "fail"} else logging.INFO,
            f"Stage 4b summary: checked={pid.checked} "
            f"t_rule={t_pass} "
            f"structural_issues={len(s.structural_issues if s else [])} "
            f"boundary_issues={len(s.boundary_issues if s else [])} "
            f"weak_params={len(s.weak_params if s else [])} outcome={self.outcome}",
        )


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


class Stage5aContract(BaseModel):
    """SVI preflight: fast approximate fit before expensive inference."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    inference_metadata: InferenceMetadataContract
    svi_diagnostics: SVIDiagnostics | None = None
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None

    def summarize(self) -> tuple[int, str]:
        converged = self.svi_diagnostics is not None
        return (
            logging.INFO,
            f"Stage 5a summary: method=svi converged={converged} outcome={self.outcome}",
        )


class Stage5bContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    power_scaling: list[PowerScalingResultContract]
    ppc: PPCResultContract
    inference_metadata: InferenceMetadataContract
    mcmc_diagnostics: MCMCDiagnostics | None = None
    svi_diagnostics: SVIDiagnostics | None = None
    smc_diagnostics: SMCDiagnostics | None = None
    loo_diagnostics: LOODiagnostics | None = None
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None

    def summarize(self) -> tuple[int, str]:
        ps_issues = sum(
            1
            for item in self.power_scaling
            if item.diagnosis in {"prior_dominated", "prior_data_conflict"}
        )
        ppc_warnings = len(self.ppc.per_variable_warnings)
        return (
            logging.WARNING if self.outcome in {"warn", "fail"} else logging.INFO,
            f"Stage 5b summary: method={self.inference_metadata.method} "
            f"samples={self.inference_metadata.n_samples} "
            f"power_scaling_issues={ps_issues} ppc_warnings={ppc_warnings} outcome={self.outcome}",
        )


class Stage6Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"
    intervention_results: list[TreatmentEffectContract]

    def summarize(self) -> tuple[int, str]:
        warnings = sum(
            1 for r in self.intervention_results if r.ppc_warnings or r.prior_sensitivity_warning
        )
        return (
            logging.WARNING if self.outcome in {"warn", "fail"} else logging.INFO,
            f"Stage 6 summary: treatments_ranked={len(self.intervention_results)} "
            f"warnings={warnings} outcome={self.outcome}",
        )


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
    "stage-5a": Stage5aContract,
    "stage-5b": Stage5bContract,
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
