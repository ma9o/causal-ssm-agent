"""Latent causal-structure artifact models and validation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError, model_validator


class Role(StrEnum):
    """Whether a variable is modeled or treated as given."""

    ENDOGENOUS = "endogenous"
    EXOGENOUS = "exogenous"


class TemporalStatus(StrEnum):
    """Whether a construct changes within-person over time."""

    TIME_VARYING = "time_varying"
    TIME_INVARIANT = "time_invariant"


class Construct(BaseModel):
    """A theoretical entity in the latent causal model."""

    name: str = Field(description="Construct name (e.g., 'stress', 'sleep_quality')")
    description: str = Field(description="What this theoretical construct represents")
    role: Role = Field(description="'endogenous' (modeled) or 'exogenous' (given)")
    is_outcome: bool = Field(
        default=False,
        description="True if this is the primary outcome variable Y implied by the question",
    )
    temporal_status: TemporalStatus = Field(
        description="'time_varying' (changes over time) or 'time_invariant' (fixed)"
    )

    @model_validator(mode="after")
    def validate_construct(self) -> Construct:
        """Validate construct field consistency."""
        if self.is_outcome and self.role != Role.ENDOGENOUS:
            raise ValueError(
                f"Outcome construct '{self.name}' must be endogenous, got {self.role.value}"
            )
        return self


class CausalEdge(BaseModel):
    """A directed causal relationship between constructs."""

    cause: str = Field(description="Name of cause construct")
    effect: str = Field(description="Name of effect construct")
    description: str = Field(description="Theoretical justification for this causal link")
    lagged: bool = Field(
        default=True,
        description=(
            "If True, effect at t is caused by cause at t-1 (one model_clock tick delay). "
            "If False (contemporaneous), effect at t is caused by cause at t."
        ),
    )


def _check_edge_constraint(
    edge: CausalEdge,
    construct_map: dict[str, Construct],
) -> str | None:
    """Check a single edge against shared latent-structure constraints."""
    cause_construct = construct_map[edge.cause]
    effect_construct = construct_map[edge.effect]

    if effect_construct.role == Role.EXOGENOUS:
        return f"Exogenous construct '{edge.effect}' cannot be an effect"

    if (
        cause_construct.temporal_status == TemporalStatus.TIME_VARYING
        and effect_construct.temporal_status == TemporalStatus.TIME_INVARIANT
    ):
        return (
            f"Time-varying construct '{edge.cause}' cannot be a cause of "
            f"time-invariant construct '{edge.effect}'. Time-invariant constructs "
            "are fixed within person and cannot have time-varying parents."
        )

    both_time_varying = (
        cause_construct.temporal_status == TemporalStatus.TIME_VARYING
        and effect_construct.temporal_status == TemporalStatus.TIME_VARYING
    )
    both_endogenous = (
        cause_construct.role == Role.ENDOGENOUS and effect_construct.role == Role.ENDOGENOUS
    )
    if not edge.lagged and both_time_varying and both_endogenous:
        return (
            f"Directed contemporaneous edge '{edge.cause}' -> '{edge.effect}' "
            "between endogenous time-varying latent constructs is excluded by the "
            "latent-model contract. Represent directed effects between evolving "
            "latent states with lagged=True; reserve same-time dependence for "
            "explicit confounding or diffusion covariance."
        )

    return None


def _check_global_constraints(
    constructs: list[Construct],
    edges: list[CausalEdge],
) -> list[str]:
    """Check latent-model global constraints."""
    errors: list[str] = []

    outcomes = [construct for construct in constructs if construct.is_outcome]
    if len(outcomes) == 0:
        errors.append("Exactly one construct must have is_outcome=true")
    elif len(outcomes) > 1:
        names = [construct.name for construct in outcomes]
        errors.append(f"Only one outcome allowed, got {len(outcomes)}: {names}")

    if len(outcomes) == 1:
        outcome_name = outcomes[0].name
        incoming_to_outcome = [edge for edge in edges if edge.effect == outcome_name]
        if not incoming_to_outcome:
            errors.append(
                f"Outcome construct '{outcome_name}' has no incoming causal edges. "
                "The model must include at least one cause of the outcome."
            )

    contemporaneous_edges = [(edge.cause, edge.effect) for edge in edges if not edge.lagged]
    if contemporaneous_edges:
        import networkx as nx

        graph = nx.DiGraph(contemporaneous_edges)
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            errors.append(
                f"Contemporaneous edges form cycle(s) within time slice: {cycles}. "
                "Use lagged=true for feedback loops across time."
            )

    return errors


class LatentModel(BaseModel):
    """Theoretical causal structure over constructs."""

    constructs: list[Construct] = Field(description="Theoretical constructs in the model")
    edges: list[CausalEdge] = Field(description="Causal edges between constructs")

    @model_validator(mode="after")
    def validate_latent_model(self) -> LatentModel:
        """Validate latent model constraints."""
        construct_map = {construct.name: construct for construct in self.constructs}

        for edge in self.edges:
            if edge.cause not in construct_map:
                raise ValueError(f"Edge cause '{edge.cause}' not in constructs")
            if edge.effect not in construct_map:
                raise ValueError(f"Edge effect '{edge.effect}' not in constructs")

            error = _check_edge_constraint(edge, construct_map)
            if error:
                raise ValueError(error)

        global_errors = _check_global_constraints(self.constructs, self.edges)
        if global_errors:
            raise ValueError(global_errors[0])

        return self


def validate_latent_model(data: dict) -> tuple[LatentModel | None, list[str]]:
    """Validate a latent model dict, collecting all errors."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    constructs = data.get("constructs", [])
    edges = data.get("edges", [])

    if not isinstance(constructs, list):
        errors.append("'constructs' must be a list")
        constructs = []
    if not isinstance(edges, list):
        errors.append("'edges' must be a list")
        edges = []

    valid_constructs: list[Construct] = []
    construct_names: set[str] = set()

    for index, construct_data in enumerate(constructs):
        if not isinstance(construct_data, dict):
            errors.append(f"constructs[{index}]: must be a dictionary")
            continue

        name = construct_data.get("name", f"<unnamed_{index}>")
        if name in construct_names:
            errors.append(f"Duplicate construct name: '{name}'")
        construct_names.add(name)

        try:
            construct = Construct.model_validate(construct_data)
            valid_constructs.append(construct)
        except ValidationError as exc:
            error_msg = str(exc)
            if "validation error" in error_msg.lower():
                for line in error_msg.split("\n")[1:]:
                    line = line.strip()
                    if line and not line.startswith("For further"):
                        errors.append(f"constructs[{index}] ({name}): {line}")
            else:
                errors.append(f"constructs[{index}] ({name}): {error_msg}")

    construct_map = {construct.name: construct for construct in valid_constructs}
    valid_edges: list[CausalEdge] = []
    for index, edge_data in enumerate(edges):
        if not isinstance(edge_data, dict):
            errors.append(f"edges[{index}]: must be a dictionary")
            continue

        cause = edge_data.get("cause", "<missing>")
        effect = edge_data.get("effect", "<missing>")
        edge_label = f"edges[{index}] ({cause} -> {effect})"

        try:
            edge = CausalEdge.model_validate(edge_data)
        except ValidationError as exc:
            errors.append(f"{edge_label}: {exc}")
            continue

        if edge.cause not in construct_map:
            errors.append(f"{edge_label}: cause '{edge.cause}' not in constructs")
            continue
        if edge.effect not in construct_map:
            errors.append(f"{edge_label}: effect '{edge.effect}' not in constructs")
            continue

        constraint_error = _check_edge_constraint(edge, construct_map)
        if constraint_error:
            errors.append(f"{edge_label}: {constraint_error}")
            continue

        valid_edges.append(edge)

    errors.extend(_check_global_constraints(valid_constructs, valid_edges))

    if not errors:
        try:
            model = LatentModel(constructs=valid_constructs, edges=valid_edges)
            return model, []
        except ValidationError as exc:
            errors.append(f"Final validation failed: {exc}")

    return None, errors


__all__ = [
    "CausalEdge",
    "Construct",
    "LatentModel",
    "Role",
    "TemporalStatus",
    "validate_latent_model",
]
