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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal_ssm_agent.models.posterior_predictive import (  # noqa: TC001
    PPCOverlay,
    PPCTestStat,
    PPCWarning,
)
from causal_ssm_agent.models.ssm.schemas_inference import (  # noqa: TC001
    InferenceStructureResult,
    LOODiagnostics,
    MCMCDiagnostics,
    ParametricIdResult,
    PosteriorMarginal,
    PosteriorPair,
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
    parameter_name: str = Field(
        description="Name of the parameter this search is for (e.g. 'beta_stress_sleep')."
    )


class ValidateModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_json: str = Field(
        description=(
            "JSON object with proposed changes. Include 'model_spec' (complete ModelSpec) "
            "and/or 'priors' (dict mapping parameter names to prior proposals). "
            "Only include fields you are changing."
        ),
    )


# --- Stage 6 tool inputs ---


class GetModelInfoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[
        Literal[
            "overview",
            "variables",
            "measurement",
            "identifiability",
            "diagnostics",
            "baseline_effects",
            "capabilities",
        ]
    ] = Field(
        default_factory=lambda: ["overview", "variables", "capabilities"],
        description="Named sections to include in the read-only model summary.",
    )
    names: list[str] = Field(
        default_factory=list,
        description="Optional construct or indicator names to focus the summary on.",
    )


class InterventionActionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable: str = Field(description="Latent construct to intervene on.")
    mode: Literal["set", "shift"] = Field(
        description="'set' clamps the construct to a value; 'shift' adds an amount to baseline."
    )
    value: float | None = Field(
        default=None,
        description="Required when mode='set'. Absolute latent-space value to clamp to.",
    )
    amount: float | None = Field(
        default=None,
        description="Required when mode='shift'. Additive latent-space delta from baseline.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> InterventionActionInput:
        if self.mode == "set" and self.value is None:
            raise ValueError("mode='set' requires value")
        if self.mode == "shift" and self.amount is None:
            raise ValueError("mode='shift' requires amount")
        return self


class InterventionQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimand: Literal["steady_state", "trajectory"] = Field(
        default="steady_state",
        description="Steady-state effect or forward trajectory effect.",
    )
    horizon_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Forward horizon in days when estimand='trajectory'.",
    )
    projection: Literal["latent", "manifest", "both"] = Field(
        default="latent",
        description="Whether to report latent outcome effects, manifest projections, or both.",
    )


class SimulateInterventionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: InterventionActionInput
    outcome: str | None = Field(
        default=None,
        description="Outcome construct. Defaults to the stage-1a outcome.",
    )
    query: InterventionQueryInput = Field(default_factory=InterventionQueryInput)


class CounterfactualEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["observed_window"] = Field(
        default="observed_window",
        description="Condition on an actual observed history window from the run.",
    )
    start_time: str | None = Field(
        default=None,
        description="Inclusive ISO-8601 lower bound for the evidence window.",
    )
    end_time: str | None = Field(
        default=None,
        description="Inclusive ISO-8601 upper bound for the evidence window. Defaults to the final observed time.",
    )
    variables: list[str] = Field(
        default_factory=list,
        description="Optional constructs or indicators to highlight when describing evidence coverage.",
    )


class CounterfactualQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimand: Literal["end_state", "trajectory"] = Field(
        default="end_state",
        description="Compare the final forecasted outcome or the full effect trajectory.",
    )
    horizon_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Forward horizon in days for the counterfactual forecast.",
    )
    projection: Literal["latent", "manifest", "both"] = Field(
        default="latent",
        description="Whether to report latent outcome effects, manifest projections, or both.",
    )


class SimulateCounterfactualInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: CounterfactualEvidenceInput = Field(default_factory=CounterfactualEvidenceInput)
    action: InterventionActionInput
    outcome: str | None = Field(
        default=None,
        description="Outcome construct. Defaults to the stage-1a outcome.",
    )
    query: CounterfactualQueryInput = Field(default_factory=CounterfactualQueryInput)


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
            name="validate_latent_model",
            description="Tool for validating latent model JSON (Stage 1a).",
            input_schema=ValidateLatentModelInput,
        ),
    ],
    "stage-1b": [
        ToolContract(
            name="validate_measurement_model",
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
    "stage-6": [
        ToolContract(
            name="get_model_info",
            description="Return a read-only summary of the fitted model, variables, identifiability status, diagnostics, and stage-6 baseline effects.",
            input_schema=GetModelInfoInput,
        ),
        ToolContract(
            name="simulate_intervention",
            description="Run a Pearl rung-2 interventional simulation on the fitted generative model.",
            input_schema=SimulateInterventionInput,
        ),
        ToolContract(
            name="simulate_counterfactual",
            description="Run a Pearl rung-3 counterfactual forecast by conditioning on an observed history window, then applying an action.",
            input_schema=SimulateCounterfactualInput,
        ),
    ],
}

# Stages with an interactive LLM trace panel in the refinement UI.
# Stage 0 is excluded (tools depend on sandbox/filesystem state).
# Stage 2 is excluded (parallel worker extraction, not a single LLM conversation).
INTERACTIVE_STAGES: frozenset[str] = frozenset({"stage-1a", "stage-1b", "stage-4", "stage-6"})


# ---------------------------------------------------------------------------
# Stage output contracts
# ---------------------------------------------------------------------------


class BaseStageContract(BaseModel):
    """Shared base for persisted stage payloads."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "warn", "fail"] = "success"

    def summary_level(self) -> int:
        return logging.WARNING if self.outcome in {"warn", "fail"} else logging.INFO

    def summary_message(self) -> str:
        raise NotImplementedError

    def summarize(self) -> tuple[int, str]:
        return self.summary_level(), self.summary_message()


class LLMStageContract(BaseStageContract):
    """Base contract for stages that surface an LLM trace."""

    llm_trace: LLMTrace | None = None


class GateOverrideContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


class DateRangeContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    end: str


class Stage0ColumnDescriptionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class Stage0Contract(LLMStageContract):
    column_descriptions: list[Stage0ColumnDescriptionContract]

    def summary_message(self) -> str:
        return f"Stage 0 summary: described_columns={len(self.column_descriptions)}"


class Stage1aContract(LLMStageContract):
    latent_model: LatentModel

    def summary_message(self) -> str:
        return (
            f"Stage 1a summary: constructs={len(self.latent_model.constructs)} "
            f"edges={len(self.latent_model.edges)}"
        )


class Stage1bContract(LLMStageContract):
    causal_spec: CausalSpec
    gate_overridden: GateOverrideContract | None = None

    def summary_message(self) -> str:
        non_id = (
            self.causal_spec.identifiability.non_identifiable_treatments
            if self.causal_spec.identifiability
            else {}
        ) or {}
        return (
            f"Stage 1b summary: constructs={len(self.causal_spec.latent.constructs)} "
            f"indicators={len(self.causal_spec.measurement.indicators)} "
            f"filtered_treatments={len(non_id)} outcome={self.outcome}"
        )


class WorkerStatusContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: int
    status: Literal["pending", "running", "completed", "failed"]
    n_extractions: int
    n_windows: int
    error: str | None = None


class ExtractionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator: str
    value: int | float | bool | str | None
    anchor_time: str | None
    support_kind: str | None = None
    summary_operator: str | None = None
    anchor_policy: str | None = None
    observation_window: str | None = None
    support_start: str | None = None
    support_end: str | None = None


class Stage2Contract(LLMStageContract):
    workers: list[WorkerStatusContract]
    combined_extractions_sample: list[ExtractionContract]
    per_indicator_counts: dict[str, int]

    def summary_message(self) -> str:
        completed = sum(1 for w in self.workers if w.status == "completed")
        failed = sum(1 for w in self.workers if w.status == "failed")
        return (
            f"Stage 2 summary: workers={len(self.workers)} completed={completed} "
            f"failed={failed} sample_rows={len(self.combined_extractions_sample)} "
            f"indicators={len(self.per_indicator_counts)} outcome={self.outcome}"
        )


class ValidationIssueContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator: str | None = None
    issue_type: str
    severity: Literal["error", "warning", "info"]
    message: str


class IndicatorEmpiricalProfileContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_dtype: str | None = None
    n_obs: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    variance: float | None
    time_coverage_ratio: float | None
    max_gap_ratio: float | None
    dtype_violations: int | None = None
    duplicate_pct: float | None = None
    arithmetic_sequence_detected: bool
    n_unparseable_timestamps: int | None = None
    zero_fraction: float | None = None
    is_nonnegative: bool | None = None
    is_unit_interval: bool | None = None
    looks_integer_valued: bool | None = None
    variance_to_mean_ratio: float | None = None


class IndicatorValidationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssueContract]
    checks: dict[str, Literal["ok", "warning", "error"]]


class IndicatorAuditContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: IndicatorEmpiricalProfileContract | None = None
    validation: IndicatorValidationContract


class Stage3Contract(BaseStageContract):
    is_valid: bool
    indicators: dict[str, IndicatorAuditContract]
    dataset_issues: list[ValidationIssueContract]

    def summary_message(self) -> str:
        indicator_issues = [
            issue for audit in self.indicators.values() for issue in audit.validation.issues
        ]
        all_issues = [*indicator_issues, *self.dataset_issues]
        errors = sum(1 for i in all_issues if i.severity == "error")
        warnings = sum(1 for i in all_issues if i.severity == "warning")
        return (
            f"Stage 3 summary: is_valid={self.is_valid} "
            f"issues={len(all_issues)} "
            f"errors={errors} warnings={warnings} outcome={self.outcome}"
        )


class Stage4Contract(LLMStageContract):
    model_spec: ModelSpec
    priors: dict[str, PriorProposal]
    search_queries: dict[str, str] | None = None
    prior_predictive_samples: dict[str, list[float]] | None = None

    def summary_message(self) -> str:
        return (
            f"Stage 4 summary: parameters={len(self.model_spec.parameters)} "
            f"likelihoods={len(self.model_spec.likelihoods)} "
            f"priors={len(self.priors)} "
            f"prior_predictive_channels={len(self.prior_predictive_samples or {})}"
        )


class Stage4bContract(BaseStageContract):
    parametric_id: ParametricIdResult
    inference_structure: InferenceStructureResult | None = None
    gate_overridden: GateOverrideContract | None = None

    def summary_message(self) -> str:
        pid = self.parametric_id
        t_pass = "pass" if (pid.t_rule is None or pid.t_rule.satisfies) else "warn"
        s = pid.summary
        return (
            f"Stage 4b summary: checked={pid.checked} "
            f"t_rule={t_pass} "
            f"structural_issues={len(s.structural_issues if s else [])} "
            f"boundary_issues={len(s.boundary_issues if s else [])} "
            f"weak_params={len(s.weak_params if s else [])} outcome={self.outcome}"
        )


class TreatmentEffectContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treatment: str
    effect_size: float | None
    posterior_draws: list[float] | None = None
    prob_positive: float | None = None
    identifiable: bool
    ppc_warnings: list[PPCWarning] | None = None
    prior_sensitivity_warning: str | None = None
    temporal: TemporalEffect | None = None
    manifest_effects: dict[str, float] | None = None


class SavedScenarioContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    query: str
    summary: str | None = None


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


class Stage5aContract(BaseStageContract):
    """SVI preflight: fast approximate fit before expensive inference."""

    inference_metadata: InferenceMetadataContract
    svi_diagnostics: SVIDiagnostics | None = None
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None

    def summary_message(self) -> str:
        converged = self.svi_diagnostics is not None
        return f"Stage 5a summary: method=svi converged={converged} outcome={self.outcome}"


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


class Stage6Contract(LLMStageContract):
    intervention_results: list[TreatmentEffectContract]
    saved_scenarios: list[SavedScenarioContract] | None = None
    final_summary: str | None = None

    def summary_message(self) -> str:
        warnings = sum(
            1 for r in self.intervention_results if r.ppc_warnings or r.prior_sensitivity_warning
        )
        return (
            f"Stage 6 summary: treatments_ranked={len(self.intervention_results)} "
            f"warnings={warnings} outcome={self.outcome}"
        )


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


def _validate_stage_model(stage_id: str, data: dict[str, Any]) -> BaseModel:
    """Validate stage payload and return the Pydantic model instance."""
    if stage_id not in STAGE_CONTRACTS:
        known = ", ".join(sorted(STAGE_CONTRACTS.keys()))
        raise ValueError(f"Unknown stage_id '{stage_id}'. Expected one of: {known}")
    sid = cast("StageId", stage_id)
    return STAGE_CONTRACTS[sid].model_validate(data)


def validate_stage_payload(stage_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Validate stage payload by stage id and return a JSON-serializable dict."""
    return _validate_stage_model(stage_id, data).model_dump(mode="json")
