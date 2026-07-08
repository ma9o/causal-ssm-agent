"""analysis contracts and tool metadata.

analysis exposes a single composable simulation tool. A *scenario* is one
operation on the fitted latent SSM:

    start state  →  apply timed latent clamp(s)  →  roll forward  →  contrast vs reference

The start is either the population baseline steady state or an abducted individual
state (conditioning on observed evidence up to a boundary). A clamp is a do-operator
on one latent variable over a time window; a clamp whose window opens at the start is a
forward "intervention", and the same machinery expresses counterfactual "what-if" edits.
The Pearl rung is therefore emergent from the start (baseline → rung 2, abducted →
rung 3), not a separate query type.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from nof1_causal_lab.flows.contracts_base import LLMArtifactContract, ToolContract
from nof1_causal_lab.models.ssm.inference.schemas import TemporalEffect  # noqa: TC001

IS_INTERACTIVE_CONTEXT = True

ModelInfoSection = Literal[
    "overview",
    "variables",
    "measurement",
    "identifiability",
    "diagnostics",
    "baseline_effects",
    "capabilities",
]


def _default_model_info_sections() -> list[ModelInfoSection]:
    return ["overview", "variables", "capabilities"]


class GetModelInfoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[ModelInfoSection] = Field(
        default_factory=_default_model_info_sections,
        description="Named sections to include in the read-only model summary.",
    )
    names: list[str] = Field(
        default_factory=list,
        description="Optional construct or indicator names to focus the summary on.",
    )


class ScenarioStartInput(BaseModel):
    """Where the forward rollout begins (replaces the rung-2/rung-3 split)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["baseline", "abducted"] = Field(
        default="baseline",
        description=(
            "'baseline' starts from the population baseline steady state (an interventional, "
            "rung-2 query). 'abducted' conditions on the individual's observed evidence and starts "
            "from the recovered fitted latent state (a counterfactual, rung-3 query)."
        ),
    )
    time_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Abducted start only: observed fitted-state index to begin from. "
            "Defaults to the final retained fitted latent state."
        ),
    )
    time: str | None = Field(
        default=None,
        description=(
            "Abducted start only: ISO-8601 observed timestamp matching a retained fitted latent "
            "state. Use either time_index or time, not both."
        ),
    )

    @model_validator(mode="after")
    def validate_payload(self) -> ScenarioStartInput:
        if self.time_index is not None and self.time is not None:
            raise ValueError("Use either start.time_index or start.time, not both")
        if self.kind == "baseline" and (self.time_index is not None or self.time is not None):
            raise ValueError("start.kind='baseline' takes no time_index/time")
        return self


class LatentClampInput(BaseModel):
    """A do-operator on one latent variable over a time window.

    The window is ``[from_day, to_day)`` in days relative to the rollout start; outside
    the window the variable evolves under its natural dynamics. ``set`` pins to an absolute
    value, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly
    interpolates across the window, and ``trajectory`` tracks a list of values across it.
    """

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(description="Latent construct to clamp.")
    mode: Literal["set", "shift", "ramp", "trajectory"] = Field(
        description="How the clamped value is specified over the window."
    )
    value: float | None = Field(
        default=None, description="Required when mode='set'. Absolute latent-space value."
    )
    amount: float | None = Field(
        default=None,
        description="Required when mode='shift'. Additive delta from the start-state value.",
    )
    value_start: float | None = Field(
        default=None, description="Required when mode='ramp'. Value at from_day."
    )
    value_end: float | None = Field(
        default=None, description="Required when mode='ramp'. Value at to_day."
    )
    values: list[float] | None = Field(
        default=None,
        description="Required when mode='trajectory'. Values sampled evenly across the window.",
    )
    from_day: float = Field(
        default=0.0, ge=0.0, description="Window onset in days from the rollout start."
    )
    to_day: float | None = Field(
        default=None,
        description="Window end in days from the rollout start. Null runs through the horizon.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> LatentClampInput:
        if self.to_day is not None and self.to_day <= self.from_day:
            raise ValueError("clamp to_day must be greater than from_day")
        if self.mode == "set" and self.value is None:
            raise ValueError("mode='set' requires value")
        if self.mode == "shift" and self.amount is None:
            raise ValueError("mode='shift' requires amount")
        if self.mode == "ramp" and (self.value_start is None or self.value_end is None):
            raise ValueError("mode='ramp' requires value_start and value_end")
        if self.mode == "trajectory" and (self.values is None or len(self.values) < 2):
            raise ValueError("mode='trajectory' requires values with at least two points")
        return self


class ScenarioQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimand: Literal["end_state", "trajectory"] = Field(
        default="trajectory",
        description="Report the final-horizon outcome effect or the full effect trajectory.",
    )
    horizon_days: int = Field(
        default=30, ge=1, le=365, description="Forward horizon in days from the rollout start."
    )
    projection: Literal["latent", "manifest", "both"] = Field(
        default="latent",
        description="Report latent outcome effects, manifest projections, or both.",
    )


class SimulateScenarioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: ScenarioStartInput = Field(default_factory=ScenarioStartInput)
    clamps: list[LatentClampInput] = Field(
        min_length=1, description="One or more timed latent clamps composing the scenario."
    )
    outcome: str | None = Field(
        default=None, description="Outcome construct. Defaults to the latent-structure outcome."
    )
    query: ScenarioQueryInput = Field(default_factory=ScenarioQueryInput)


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


class BaselineReportVisualizationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_node_trajectories: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent trajectories for the reference (no-clamp) path aligned to "
            "effect_trajectory days."
        ),
    )
    action_node_trajectories: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent trajectories under the composed clamps aligned to "
            "effect_trajectory days."
        ),
    )
    node_effect_trajectories: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent effect trajectories aligned to effect_trajectory days. "
            "Values are causal deltas relative to the reference path."
        ),
    )
    start_state: dict[str, float] | None = Field(
        default=None,
        description="Posterior mean latent state the rollout started from.",
    )


class ScenarioStartResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["baseline", "abducted"]
    time_index: int | None = None
    time: str | None = None
    state_source: Literal["baseline_steady_state", "fitted_latent_paths"]


class SimulateScenarioResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: ScenarioStartResultContract
    clamps: list[LatentClampInput]
    outcome: str
    estimand: Literal["end_state", "trajectory"]
    summary: EffectSummaryContract
    effect_trajectory: list[EffectTrajectoryPointContract] | None = None
    visualization: BaselineReportVisualizationContract | None = None
    manifest_effects: dict[str, float] | None = None
    reference_mean: float = Field(
        description="Mean reference outcome (baseline steady state or factual forecast)."
    )
    warnings: list[str] = Field(default_factory=list)


class SimulateScenarioToolResultContract(
    RootModel[SimulateScenarioResultContract | ToolErrorContract]
):
    pass


ANALYSIS_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="get_model_info",
        description=(
            "Return a read-only summary of the fitted model, variables, identifiability status, "
            "diagnostics, and baseline effects."
        ),
        input_schema=GetModelInfoInput,
    ),
    ToolContract(
        name="simulate",
        description=(
            "Run a composable causal scenario on the fitted generative model. Start from the "
            "population baseline steady state (interventional) or an abducted fitted latent state "
            "(counterfactual), apply one or more timed latent clamps (do-operators), and read the "
            "effect on an outcome over a horizon."
        ),
        input_schema=SimulateScenarioInput,
        output_schema=SimulateScenarioToolResultContract,
    ),
]


EXPORTED_TOOL_RESULT_MODELS: tuple[type[BaseModel], ...] = (
    ToolErrorContract,
    EffectSummaryContract,
    EffectTrajectoryPointContract,
    BaselineReportVisualizationContract,
    ScenarioStartResultContract,
    SimulateScenarioResultContract,
    SimulateScenarioToolResultContract,
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


class BaselineReportContract(LLMArtifactContract):
    intervention_results: list[TreatmentEffectContract]
    saved_scenarios: list[SavedScenarioContract] | None = None
    final_summary: str | None = None
