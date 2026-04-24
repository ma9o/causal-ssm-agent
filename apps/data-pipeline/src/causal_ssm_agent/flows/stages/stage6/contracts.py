"""Stage 6 contracts and tool metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from causal_ssm_agent.flows.contracts_base import LLMStageContract, ToolContract
from causal_ssm_agent.models.ssm.inference.schemas import TemporalEffect  # noqa: TC001

STAGE_ID = "stage-6"
IS_INTERACTIVE_STAGE = True


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

    start_time: str | None = Field(
        default=None,
        description="Inclusive ISO-8601 lower bound for the evidence window.",
    )
    end_time: str | None = Field(
        default=None,
        description=(
            "Inclusive ISO-8601 upper bound for the evidence window. "
            "Defaults to the final observed time."
        ),
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


class ToolErrorContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    identifiable_treatments: list[str] | None = None


class EffectSummaryContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float
    median: float
    lower_95: float
    upper_95: float
    prob_positive: float


class EffectTrajectoryPointContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: float
    effect: float


class Stage6VisualizationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_node_trajectories: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent trajectories for the reference path aligned to "
            "effect_trajectory days. This is the no-action baseline forecast for rung-2 "
            "queries and the factual forecast from the abducted state for rung-3 queries."
        ),
    )
    action_node_trajectories: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent trajectories under the queried action aligned to "
            "effect_trajectory days."
        ),
    )
    node_effect_trajectories: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent effect trajectories aligned to effect_trajectory days. "
            "Values are causal deltas relative to the relevant reference path."
        ),
    )
    abducted_state: dict[str, float] | None = Field(
        default=None,
        description="Recovered latent state at the evidence boundary for rung-3 queries.",
    )


class CounterfactualEvidenceResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_time: str
    end_time: str
    n_timepoints: int
    variables: list[str] = Field(default_factory=list)
    conditioning_method: str


class BaseStage6SimulationResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: InterventionActionInput
    outcome: str
    summary: EffectSummaryContract
    effect_trajectory: list[EffectTrajectoryPointContract] | None = None
    visualization: Stage6VisualizationContract | None = None
    manifest_effects: dict[str, float] | None = None
    warnings: list[str] = Field(default_factory=list)


class SimulateInterventionResultContract(BaseStage6SimulationResultContract):
    rung: Literal[2]
    estimand: Literal["steady_state", "trajectory"]
    baseline_treatment_mean: float


class SimulateCounterfactualResultContract(BaseStage6SimulationResultContract):
    rung: Literal[3]
    evidence: CounterfactualEvidenceResultContract
    estimand: Literal["end_state", "trajectory"]
    baseline_forecast_mean: float


class SimulateInterventionToolResultContract(
    RootModel[SimulateInterventionResultContract | ToolErrorContract]
):
    pass


class SimulateCounterfactualToolResultContract(
    RootModel[SimulateCounterfactualResultContract | ToolErrorContract]
):
    pass


STAGE6_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="get_model_info",
        description=(
            "Return a read-only summary of the fitted model, variables, identifiability status, "
            "diagnostics, and stage-6 baseline effects."
        ),
        input_schema=GetModelInfoInput,
    ),
    ToolContract(
        name="simulate_intervention",
        description="Run a Pearl rung-2 interventional simulation on the fitted generative model.",
        input_schema=SimulateInterventionInput,
        output_schema=SimulateInterventionToolResultContract,
    ),
    ToolContract(
        name="simulate_counterfactual",
        description=(
            "Run a Pearl rung-3 counterfactual forecast by conditioning on an observed "
            "history window, then applying an action."
        ),
        input_schema=SimulateCounterfactualInput,
        output_schema=SimulateCounterfactualToolResultContract,
    ),
]


EXPORTED_TOOL_RESULT_MODELS: tuple[type[BaseModel], ...] = (
    ToolErrorContract,
    EffectSummaryContract,
    EffectTrajectoryPointContract,
    Stage6VisualizationContract,
    CounterfactualEvidenceResultContract,
    SimulateInterventionResultContract,
    SimulateCounterfactualResultContract,
    SimulateInterventionToolResultContract,
    SimulateCounterfactualToolResultContract,
)


class TreatmentEffectContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treatment: str
    posterior_draws: list[float] | None = None
    temporal: TemporalEffect | None = None
    manifest_effects: dict[str, float] | None = None


class SavedScenarioContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    query: str
    summary: str | None = None


class Stage6Contract(LLMStageContract):
    intervention_results: list[TreatmentEffectContract]
    saved_scenarios: list[SavedScenarioContract] | None = None
    final_summary: str | None = None

    def summary_message(self) -> str:
        return (
            f"Stage 6 summary: treatments_ranked={len(self.intervention_results)} "
            f"outcome={self.outcome}"
        )
