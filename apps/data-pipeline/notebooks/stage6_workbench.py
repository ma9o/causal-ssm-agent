"""Interactive Stage 6 simulation workbench for notebooks.

Import from a notebook with:

    from stage6_workbench import Stage6Workbench

The widgets edit typed scenario specs and delegate execution to the existing
Stage 6 tool server functions. Simulation logic stays in the pipeline package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

import nof1_causal_lab.tool_server as tool_server
from nof1_causal_lab.flows.stages.stage6.contracts import (
    CounterfactualEvidenceInput,
    CounterfactualQueryInput,
    GetModelInfoInput,
    InterventionActionInput,
    InterventionQueryInput,
)


class InterventionScenarioSpec(BaseModel):
    """One rung-2 intervention simulation request."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["intervention"] = "intervention"
    name: str | None = None
    action: InterventionActionInput
    outcome: str | None = None
    query: InterventionQueryInput = Field(default_factory=InterventionQueryInput)

    def tool_input(self) -> dict[str, Any]:
        """Return the Stage 6 tool payload."""
        return {
            "action": self.action.model_dump(mode="json"),
            "outcome": self.outcome,
            "query": self.query.model_dump(mode="json"),
        }


class CounterfactualScenarioSpec(BaseModel):
    """One rung-3 counterfactual simulation request."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["counterfactual"] = "counterfactual"
    name: str | None = None
    evidence: CounterfactualEvidenceInput = Field(default_factory=CounterfactualEvidenceInput)
    action: InterventionActionInput
    outcome: str | None = None
    query: CounterfactualQueryInput = Field(default_factory=CounterfactualQueryInput)

    def tool_input(self) -> dict[str, Any]:
        """Return the Stage 6 tool payload."""
        return {
            "evidence": self.evidence.model_dump(mode="json"),
            "action": self.action.model_dump(mode="json"),
            "outcome": self.outcome,
            "query": self.query.model_dump(mode="json"),
        }


type ScenarioSpec = InterventionScenarioSpec | CounterfactualScenarioSpec
_SCENARIO_ADAPTER = TypeAdapter(ScenarioSpec)


@dataclass(frozen=True)
class VariableActionBuilder:
    """Small helper for readable notebook scenario definitions."""

    variable: str

    def set(self, value: float) -> InterventionActionInput:
        """Clamp the variable to an absolute latent-space value."""
        return InterventionActionInput(variable=self.variable, mode="set", value=value)

    def shift(self, amount: float) -> InterventionActionInput:
        """Clamp the variable to baseline/abducted value plus ``amount``."""
        return InterventionActionInput(variable=self.variable, mode="shift", amount=amount)


@dataclass(frozen=True)
class Stage6Result:
    """Result wrapper with notebook-friendly accessors."""

    scenario: ScenarioSpec
    raw: dict[str, Any]

    @property
    def summary(self) -> dict[str, float]:
        return dict(self.raw["summary"])

    @property
    def warnings(self) -> list[str]:
        return list(self.raw.get("warnings") or [])

    @property
    def effect_trajectory(self) -> list[dict[str, float]]:
        return list(self.raw.get("effect_trajectory") or [])

    @property
    def manifest_effects(self) -> dict[str, float] | None:
        manifest_effects = self.raw.get("manifest_effects")
        return dict(manifest_effects) if isinstance(manifest_effects, dict) else None

    def trajectory_figure(self, *, title: str | None = None) -> Any:
        """Return a Plotly figure for the mean effect trajectory."""
        trajectory = self.effect_trajectory
        if not trajectory:
            raise ValueError("This result has no effect_trajectory to plot.")

        import plotly.graph_objects as go

        x = [point["day"] for point in trajectory]
        y = [point["effect"] for point in trajectory]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=self.scenario.name or self.scenario.kind,
            )
        )
        fig.update_layout(
            title=title or self.scenario.name or "Stage 6 effect trajectory",
            xaxis_title="Days",
            yaxis_title="Effect",
            template="plotly_white",
        )
        return fig


class Stage6Session:
    """Load a fitted workspace and run typed Stage 6 scenarios."""

    def __init__(self, workspace_id: str, context: dict[str, Any]) -> None:
        self.workspace_id = workspace_id
        self._ctx = context

    @classmethod
    def from_workspace(cls, workspace_id: str) -> Stage6Session:
        """Load the Stage 6 simulation context from persisted workspace artifacts."""
        return cls(
            workspace_id=workspace_id, context=tool_server._build_stage6_context(workspace_id)
        )

    @classmethod
    def from_context(
        cls,
        context: dict[str, Any],
        *,
        workspace_id: str = "<memory>",
    ) -> Stage6Session:
        """Construct a session from an already-loaded Stage 6 context."""
        return cls(workspace_id=workspace_id, context=context)

    @property
    def identifiable_treatments(self) -> list[str]:
        return list(self._ctx["_identifiable_treatments"])

    @property
    def outcome(self) -> str | None:
        value = self._ctx.get("_outcome_name")
        return str(value) if value is not None else None

    @property
    def latent_names(self) -> list[str]:
        artifact = self._ctx["_fitted_artifact"]
        spec = artifact.builder.spec
        return list(spec.latent_names or [])

    def variable(self, name: str) -> VariableActionBuilder:
        """Return a builder for single-variable Stage 6 actions."""
        return VariableActionBuilder(name)

    def model_info(
        self,
        *,
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
        ]
        | None = None,
        names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the same model summary exposed by the Stage 6 tool."""
        payload = GetModelInfoInput(
            sections=sections
            if sections is not None
            else ["overview", "variables", "capabilities"],
            names=names or [],
        ).model_dump(mode="json")
        return tool_server._execute_get_model_info(self._ctx, payload)["result"]

    def capabilities(self) -> dict[str, Any]:
        """Return the Stage 6 simulation capability summary."""
        return self.model_info(sections=["capabilities"])["capabilities"]

    def intervention(
        self,
        *,
        action: InterventionActionInput,
        name: str | None = None,
        outcome: str | None = None,
        estimand: Literal["steady_state", "trajectory"] = "trajectory",
        horizon_days: int = 30,
        projection: Literal["latent", "manifest", "both"] = "latent",
    ) -> InterventionScenarioSpec:
        """Build a rung-2 intervention scenario."""
        return InterventionScenarioSpec(
            name=name,
            action=action,
            outcome=outcome,
            query=InterventionQueryInput(
                estimand=estimand,
                horizon_days=horizon_days,
                projection=projection,
            ),
        )

    def counterfactual(
        self,
        *,
        action: InterventionActionInput,
        name: str | None = None,
        outcome: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        variables: list[str] | None = None,
        estimand: Literal["end_state", "trajectory"] = "trajectory",
        horizon_days: int = 30,
        projection: Literal["latent", "manifest", "both"] = "latent",
    ) -> CounterfactualScenarioSpec:
        """Build a rung-3 counterfactual scenario."""
        return CounterfactualScenarioSpec(
            name=name,
            evidence=CounterfactualEvidenceInput(
                start_time=start_time,
                end_time=end_time,
                variables=variables or [],
            ),
            action=action,
            outcome=outcome,
            query=CounterfactualQueryInput(
                estimand=estimand,
                horizon_days=horizon_days,
                projection=projection,
            ),
        )

    def run(self, scenario: ScenarioSpec | dict[str, Any]) -> Stage6Result:
        """Execute one Stage 6 scenario and return a notebook result wrapper."""
        scenario = _SCENARIO_ADAPTER.validate_python(scenario)
        if scenario.kind == "intervention":
            response = tool_server._execute_simulate_intervention(self._ctx, scenario.tool_input())
        else:
            response = tool_server._execute_simulate_counterfactual(
                self._ctx, scenario.tool_input()
            )

        result = response["result"]
        if isinstance(result, dict) and "error" in result:
            raise ValueError(str(result["error"]))
        if not isinstance(result, dict):
            raise TypeError("Stage 6 tool returned a non-object result.")
        return Stage6Result(scenario=scenario, raw=result)


class Stage6Workbench:
    """Interactive notebook workbench backed by :class:`Stage6Session`."""

    def __init__(self, session: Stage6Session) -> None:
        self.session = session
        self.results: list[Stage6Result] = []
        self._build_widgets()

    @classmethod
    def from_workspace(cls, workspace_id: str) -> Stage6Workbench:
        """Load a workspace and return a ready-to-display workbench."""
        return cls(Stage6Session.from_workspace(workspace_id))

    def display(self) -> Any:
        """Display the workbench in a Jupyter notebook."""
        from IPython.display import display

        display(self.view)
        return self.view

    def _build_widgets(self) -> None:
        import ipywidgets as widgets

        treatments = self.session.identifiable_treatments
        latent_names = self.session.latent_names
        outcome = self.session.outcome
        outcome_options = latent_names or ([outcome] if outcome else [])
        treatment_options = treatments or latent_names

        self.kind = widgets.ToggleButtons(
            options=[("Intervention", "intervention"), ("Counterfactual", "counterfactual")],
            description="Query",
        )
        self.name = widgets.Text(value="", description="Name", placeholder="optional scenario name")
        self.treatment = widgets.Dropdown(
            options=treatment_options,
            description="Treatment",
            disabled=not treatment_options,
        )
        self.outcome = widgets.Dropdown(
            options=outcome_options,
            value=outcome
            if outcome in outcome_options
            else (outcome_options[0] if outcome_options else None),
            description="Outcome",
            disabled=not outcome_options,
        )
        self.mode = widgets.ToggleButtons(
            options=[("Shift", "shift"), ("Set", "set")], description="Action"
        )
        self.amount = widgets.FloatText(value=1.0, description="Amount")
        self.value = widgets.FloatText(value=0.0, description="Value", disabled=True)
        self.estimand = widgets.Dropdown(
            options=[("Trajectory", "trajectory"), ("Steady state", "steady_state")],
            description="Estimand",
        )
        self.horizon = widgets.IntSlider(
            value=30,
            min=1,
            max=365,
            step=1,
            description="Horizon",
            continuous_update=False,
        )
        self.projection = widgets.ToggleButtons(
            options=[("Latent", "latent"), ("Manifest", "manifest"), ("Both", "both")],
            description="Projection",
        )
        self.start_time = widgets.Text(
            value="",
            description="Start",
            placeholder="optional ISO-8601",
            disabled=True,
        )
        self.end_time = widgets.Text(
            value="",
            description="End",
            placeholder="optional ISO-8601",
            disabled=True,
        )
        self.evidence_variables = widgets.SelectMultiple(
            options=latent_names,
            description="Evidence",
            disabled=True,
        )
        self.run_button = widgets.Button(description="Run", button_style="primary")
        self.clear_button = widgets.Button(description="Clear")
        self.scenario_list = widgets.Select(options=[], description="Results", rows=6)
        self.output = widgets.Output()

        self.kind.observe(self._on_kind_change, names="value")
        self.mode.observe(self._on_mode_change, names="value")
        self.run_button.on_click(self._on_run)
        self.clear_button.on_click(self._on_clear)
        self.scenario_list.observe(self._on_select_result, names="value")

        controls = widgets.VBox(
            [
                widgets.HBox([self.kind, self.name]),
                widgets.HBox([self.treatment, self.outcome]),
                widgets.HBox([self.mode, self.amount, self.value]),
                widgets.HBox([self.estimand, self.horizon, self.projection]),
                widgets.Accordion(
                    children=[
                        widgets.VBox([self.start_time, self.end_time, self.evidence_variables])
                    ],
                    titles=("Counterfactual evidence",),
                ),
                widgets.HBox([self.run_button, self.clear_button]),
            ]
        )
        side = widgets.VBox([self.scenario_list])
        self.view = widgets.VBox(
            [
                widgets.HTML("<h3>Stage 6 simulation workbench</h3>"),
                widgets.HBox([controls, side]),
                self.output,
            ]
        )
        self._on_kind_change({"new": self.kind.value})

    def _on_kind_change(self, change: dict[str, Any]) -> None:
        kind = change["new"]
        is_counterfactual = kind == "counterfactual"
        self.start_time.disabled = not is_counterfactual
        self.end_time.disabled = not is_counterfactual
        self.evidence_variables.disabled = not is_counterfactual
        self.estimand.options = (
            [("Trajectory", "trajectory"), ("End state", "end_state")]
            if is_counterfactual
            else [("Trajectory", "trajectory"), ("Steady state", "steady_state")]
        )
        self.estimand.value = "trajectory"

    def _on_mode_change(self, change: dict[str, Any]) -> None:
        mode = change["new"]
        self.amount.disabled = mode != "shift"
        self.value.disabled = mode != "set"

    def _scenario_name(self) -> str | None:
        value = self.name.value.strip()
        return value or None

    def _action(self) -> Any:
        variable = str(self.treatment.value)
        if self.mode.value == "set":
            return self.session.variable(variable).set(float(self.value.value))
        return self.session.variable(variable).shift(float(self.amount.value))

    def _build_scenario(self) -> InterventionScenarioSpec | CounterfactualScenarioSpec:
        common = {
            "name": self._scenario_name(),
            "action": self._action(),
            "outcome": str(self.outcome.value) if self.outcome.value else None,
            "horizon_days": int(self.horizon.value),
            "projection": self.projection.value,
        }
        if self.kind.value == "counterfactual":
            return self.session.counterfactual(
                **common,
                start_time=self.start_time.value.strip() or None,
                end_time=self.end_time.value.strip() or None,
                variables=[str(value) for value in self.evidence_variables.value],
                estimand=self.estimand.value,
            )
        return self.session.intervention(
            **common,
            estimand=self.estimand.value,
        )

    def _on_run(self, _button: Any) -> None:
        scenario = self._build_scenario()
        with self.output:
            self.output.clear_output(wait=True)
            try:
                result = self.session.run(scenario)
            except Exception as exc:  # noqa: BLE001 - notebook UI must show user-facing errors
                print(f"Stage 6 simulation failed: {exc}")
                return

            self.results.append(result)
            self._refresh_result_list()
            self._display_result(result)

    def _on_clear(self, _button: Any) -> None:
        self.results.clear()
        self._refresh_result_list()
        self.output.clear_output(wait=True)

    def _on_select_result(self, change: dict[str, Any]) -> None:
        index = change.get("new")
        if index is None:
            return
        with self.output:
            self.output.clear_output(wait=True)
            self._display_result(self.results[int(index)])

    def _refresh_result_list(self) -> None:
        self.scenario_list.options = [
            (self._result_label(index, result), index) for index, result in enumerate(self.results)
        ]

    def _result_label(self, index: int, result: Stage6Result) -> str:
        name = result.scenario.name or f"{result.scenario.kind} {index + 1}"
        mean = result.summary.get("mean")
        return f"{name}: mean={mean:.3g}" if mean is not None else name

    def _display_result(self, result: Stage6Result) -> None:
        from IPython.display import JSON, display

        print(result.scenario.name or result.scenario.kind)
        display(JSON(result.summary))
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"- {warning}")
        if result.effect_trajectory:
            display(result.trajectory_figure())
        if result.manifest_effects:
            print("Manifest effects:")
            display(JSON(result.manifest_effects))


def launch(workspace_id: str) -> Stage6Workbench:
    """Create and display a Stage 6 workbench for a workspace."""
    workbench = Stage6Workbench.from_workspace(workspace_id)
    workbench.display()
    return workbench
