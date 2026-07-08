"""Message types crossing the workflow/activity/facade boundaries.

Kept free of heavy imports (only pydantic + the pure machine modules) so
the workflow sandbox can import this module without dragging in storage,
polars, or jax.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.machine.artifacts import (  # noqa: TC001 (pydantic field annotations)
    ArtifactId,
    ArtifactVersionInfo,
    EpisodeState,
    Provenance,
)
from nof1_causal_lab.machine.moves import (
    ArtifactStatus,
    ExecOptions,
    Move,
    RetractedArtifact,
)

LLMSubroutineContextKind = Literal[
    "measurement_extraction",
    "latent_structure",
    "measurement_structure",
    "analysis_commentary",
    "raw_data_ingestion",
    "model_spec_construct",
]

SingleLLMTransitionId = Literal[
    "raw_data",
    "latent_structure",
    "measurement_structure",
    "baseline_report",
]


class EpisodeInit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    # Rehydration seed: reconstructed from the on-disk episode journal so a
    # workflow (re)started after Temporal lost its in-memory history resumes
    # with the artifacts already produced, instead of re-running from raw_data.
    # Empty/0 for a genuinely new episode; ignored when attaching to a live
    # workflow (USE_EXISTING).
    initial_state: EpisodeState | None = None
    initial_seq: int = 0


class MoveRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    move: Move
    payload: dict[str, Any] | None = None  # write moves
    options: ExecOptions = Field(default_factory=ExecOptions)  # run moves


class MoveOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    status: str  # applied | rejected | raised
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    state: EpisodeState


class EpisodeStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    state: EpisodeState
    artifacts: list[ArtifactStatus]
    legal: list[Move]


class RunArtifactInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    artifact_id: ArtifactId
    state: EpisodeState
    options: ExecOptions = Field(default_factory=ExecOptions)


class MeasurementsWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    state: EpisodeState
    options: ExecOptions = Field(default_factory=ExecOptions)


class OpenRouterLLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    max_tokens: int | None = None
    timeout: int | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None


class LLMBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    harness: Literal["none", "claude-code", "codex"]
    model: str
    max_tokens: int | None = None
    timeout: int | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    bin: str | None = None
    effort: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    fallback_model: str | None = None
    service_tier: str | None = None


class LLMToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    param_name: str = ""
    param_description: str = ""
    parameters_schema: dict[str, Any] | None = None
    kind: Literal["read_only", "checkpoint", "terminal"] = "terminal"
    executor: Literal[
        "context_json_validation",
        "raw_data_list_files",
        "raw_data_read_file_sample",
        "raw_data_execute_python",
        "raw_data_submit_table",
        "model_spec_submit_construct",
        "model_spec_search_literature",
    ] = "context_json_validation"
    success_output: str | None = "VALID"

    @property
    def parameters(self) -> dict[str, Any]:
        if self.parameters_schema is not None:
            return self.parameters_schema
        return {
            "type": "object",
            "properties": {
                self.param_name: {"type": "string", "description": self.param_description}
            },
            "required": [self.param_name],
            "additionalProperties": False,
        }


class LLMSubroutineInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    llm: LLMBackendConfig
    max_tool_turns: int
    require_result: bool = True


class LLMSubroutineStartInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str


class LLMSubroutineStart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    conversation_ref_base: str
    user_message_count: int
    tools: list[LLMToolSpec] = Field(default_factory=list)
    call_ref_base: str
    assistant_ref_base: str
    tool_execution_ref_base: str
    harness_state_ref: str
    harness_tool_ref_base: str
    result_ref_base: str


class AppendLLMUserMessageInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    conversation_ref: str
    user_message_index: int


class AppendLLMUserMessageResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class AppendLLMRepairMessageInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    conversation_ref: str
    next_conversation_ref: str
    error_text: str
    tools: list[LLMToolSpec] = Field(default_factory=list)


class AppendLLMRepairMessageResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class LLMToolExecutionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    conversation_ref: str
    assistant_ref: str
    execution_ref: str
    result_ref: str
    tools: list[LLMToolSpec] = Field(default_factory=list)
    max_tool_output: int | None = None


class LLMToolExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    terminal_success: bool
    result_ref: str | None = None
    feedback_preview: str
    tool_calls_fired: list[str] = Field(default_factory=list)


class HarnessTurnInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    workflow_run_id: str
    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    harness_state_ref: str
    harness_tool_ref_base: str
    result_ref: str
    llm: LLMBackendConfig
    tools: list[LLMToolSpec] = Field(default_factory=list)
    user_message_index: int
    log_label: str


class HarnessToolRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    result_ref: str
    tool: LLMToolSpec
    tool_name: str
    arguments: dict[str, Any]
    request_ref: str
    response_ref: str


class HarnessToolExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    tool_name: str
    output: str
    result_ref: str | None = None
    success: bool = False


class HarnessTurnResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    harness_state_ref: str
    trace_ref: str
    completion_preview: str
    result_ref: str | None = None
    terminal_tool_name: str | None = None
    tool_calls_fired: list[str] = Field(default_factory=list)


class LLMSubroutineResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result_ref: str | None = None
    conversation_ref: str | None = None
    trace_ref: str | None = None
    n_llm_calls: int = 0
    n_harness_turns: int = 0


class LLMSubroutineTraceInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    conversation_ref: str
    call_ref_base: str
    harness_trace_refs: list[str] = Field(default_factory=list)


class LLMSubroutineTraceResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_ref: str


class SingleLLMTransitionWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    transition_id: SingleLLMTransitionId
    state: EpisodeState
    options: ExecOptions = Field(default_factory=ExecOptions)


class SingleLLMTransitionPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    context_ref: str
    pins: dict[ArtifactId, int]
    llm: LLMBackendConfig
    max_tool_turns: int


class SingleLLMTransitionFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    transition_id: SingleLLMTransitionId
    state: EpisodeState
    pins: dict[ArtifactId, int]
    context_ref: str
    result_ref: str | None = None
    trace_ref: str


class StatisticalModelSpecWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    state: EpisodeState
    options: ExecOptions = Field(default_factory=ExecOptions)


class StatisticalModelSpecPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    state_ref: str
    context_ref: str
    pins: dict[ArtifactId, int]
    order: list[str]
    llm: LLMBackendConfig
    max_tool_turns: int
    max_attempts_per_construct: int


class StatisticalModelSpecAttemptPlanInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    state_ref: str
    context_ref: str
    attempt: int


class StatisticalModelSpecAttemptPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context_ref: str
    construct_name: str
    attempt: int
    subroutine_id: str


class StatisticalModelSpecAttemptFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state_ref: str
    construct_name: str
    attempt: int


class StatisticalModelSpecAttemptResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    construct_name: str
    attempt: int
    admitted: bool
    outcome: str


class StatisticalModelSpecFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    state: EpisodeState
    pins: dict[ArtifactId, int]
    state_ref: str
    context_ref: str
    trace_refs: list[str] = Field(default_factory=list)


class StatisticalModelSpecFailedEventInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    construct_name: str | None = None
    message: str


class MeasurementChunkRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: int
    n_windows: int
    spec_ref: str


class MeasurementsPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    plan_ref: str
    pins: dict[ArtifactId, int]
    chunks: list[MeasurementChunkRef] = Field(default_factory=list)
    max_concurrent_workers: int
    max_rpm: int
    max_tool_turns: int
    llm: LLMBackendConfig


class ExtractionProgressSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_workers: int
    pending_workers: int
    running_workers: int
    completed_workers: int
    failed_workers: int
    llm_requests_last_60s: int = 0


class ExtractionProgressEventInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    kind: str  # plan | worker | snapshot
    total_workers: int | None = None
    max_concurrent_workers: int | None = None
    max_rpm: int | None = None
    worker_id: int | None = None
    state: str | None = None
    n_windows: int | None = None
    n_extractions: int | None = None
    n_llm_calls: int | None = None
    error: str | None = None
    snapshot: ExtractionProgressSnapshot | None = None


class TransitionRuntimeEventInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    transition_id: str
    status: str
    error: dict[str, Any] | None = None


class ExtractionChunkWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    worker_id: int
    n_windows: int
    spec_ref: str
    attempt: int
    llm: LLMBackendConfig
    max_tool_turns: int


class OpenRouterCallInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    next_conversation_ref: str
    call_ref: str
    assistant_ref: str
    llm: OpenRouterLLMConfig
    tools: list[LLMToolSpec] = Field(default_factory=list)
    log_label: str


class ToolCallSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    id: str
    name: str


class OpenRouterCallResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    assistant_ref: str
    model: str
    stop_reason: str | None = None
    time: float
    usage: dict[str, int | None] | None = None
    completion_preview: str
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)


class ExtractionChunkFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    worker_id: int
    attempt: int
    n_windows: int
    result_ref: str
    conversation_ref: str
    trace_ref: str
    n_llm_calls: int


class ExtractionChunkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: int
    status: str
    n_extractions: int
    n_windows: int
    n_llm_calls: int = 0
    result_ref: str | None = None
    trace_ref: str | None = None
    error: str | None = None


class MeasurementsFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    state: EpisodeState
    run_id: str
    plan_ref: str
    pins: dict[ArtifactId, int]
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)


class WriteArtifactInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    artifact_id: ArtifactId
    payload: dict[str, Any]
    provenance: Provenance
    state: EpisodeState


class JournalInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    move: Move
    status: str
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    state_after: EpisodeState
