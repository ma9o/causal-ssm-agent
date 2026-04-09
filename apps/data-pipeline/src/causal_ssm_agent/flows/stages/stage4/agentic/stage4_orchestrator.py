"""Stage 4 execution plan and repair topology.

Builds the immutable block plan and structural repair topology from
the deterministic skeleton.  Parameter enumeration lives in
``stage4_skeleton``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from causal_ssm_agent.utils.causal_spec import (
    get_estimation_edges,
    get_estimation_state_order,
    get_indicators,
)

from .stage4_skeleton import Stage4Skeleton, indicators_per_construct


@dataclass(frozen=True)
class Stage4FrontierBlock:
    """A single reducer-owned Stage 4 decision block."""

    id: str
    kind: str
    label: str
    construct_names: tuple[str, ...] = ()
    variable_names: tuple[str, ...] = ()
    parameter_names: tuple[str, ...] = ()
    expand_neighbor_topology: bool = True
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage4RepairTopology:
    """Deterministic structural repair topology projected from the Stage 4 plan."""

    parameter_to_block_id: dict[str, str] = field(default_factory=dict)
    indicator_to_decision_block_id: dict[str, str] = field(default_factory=dict)
    indicator_to_measurement_block_id: dict[str, str] = field(default_factory=dict)
    indicator_names_by_construct: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dynamics_block_id_by_construct: dict[str, str] = field(default_factory=dict)
    scc_id_by_construct: dict[str, str] = field(default_factory=dict)
    scc_construct_names_by_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    internal_effect_block_ids_by_scc_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reciprocal_parameter_by_parameter: dict[str, str] = field(default_factory=dict)
    parameter_construct_names: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def get_parameter_block_id(self, parameter_name: str) -> str | None:
        """Return the owning prompt block for a semantic parameter."""
        return self.parameter_to_block_id.get(parameter_name)

    def get_measurement_block_id(self, indicator_name: str) -> str | None:
        """Return the measurement block for a manifest indicator."""
        return self.indicator_to_measurement_block_id.get(indicator_name)

    def get_indicator_owner_block_id(self, indicator_name: str) -> str | None:
        """Return the owner block for a manifest indicator token."""
        return self.indicator_to_decision_block_id.get(
            indicator_name
        ) or self.get_measurement_block_id(indicator_name)

    def get_scc_id(self, construct_name: str) -> str | None:
        """Return the SCC id for a latent construct."""
        return self.scc_id_by_construct.get(construct_name)


@dataclass(frozen=True)
class Stage4Plan:
    """Immutable Stage 4 execution plan derived from the deterministic skeleton."""

    model_blocks: tuple[Stage4FrontierBlock, ...] = ()
    review_block: Stage4FrontierBlock | None = None
    prior_blocks: tuple[Stage4FrontierBlock, ...] = ()
    prior_review_block: Stage4FrontierBlock | None = None
    blocks_by_id: dict[str, Stage4FrontierBlock] = field(default_factory=dict)
    repair_topology: Stage4RepairTopology = field(default_factory=Stage4RepairTopology)

    @property
    def all_blocks(self) -> tuple[Stage4FrontierBlock, ...]:
        """Return all blocks in deterministic execution order."""
        review = (self.review_block,) if self.review_block is not None else ()
        prior_review = (self.prior_review_block,) if self.prior_review_block is not None else ()
        return (*self.model_blocks, *review, *self.prior_blocks, *prior_review)

    def get_block(self, block_id: str) -> Stage4FrontierBlock | None:
        """Return a block by id, if present."""
        return self.blocks_by_id.get(block_id)

    @property
    def prior_review_block_id(self) -> str | None:
        """Return the whole-system prior-review block id, if configured."""
        if self.prior_review_block is None:
            return None
        return self.prior_review_block.id


def build_stage4_plan(causal_spec: dict, skeleton: Stage4Skeleton) -> Stage4Plan:
    """Build the immutable Stage 4 block plan from the deterministic skeleton."""
    construct_order = get_estimation_state_order(causal_spec)
    indicators = get_indicators(causal_spec)
    indicator_lookup = {indicator["name"]: indicator for indicator in indicators}
    grouped_indicators = indicators_per_construct(indicators)
    param_order = {parameter["name"]: idx for idx, parameter in enumerate(skeleton.all_params)}

    model_blocks: list[Stage4FrontierBlock] = []
    for item in skeleton.ambiguous_indicators:
        variable = item["variable"]
        construct_name = (indicator_lookup.get(variable) or {}).get("construct_name")
        model_blocks.append(
            Stage4FrontierBlock(
                id=f"indicator:{variable}",
                kind="indicator_decision",
                label=variable.replace("_", " ").title(),
                construct_names=(construct_name,) if isinstance(construct_name, str) else (),
                variable_names=(variable,),
                payload=dict(item),
            )
        )

    measurement_parameter_names_by_construct: dict[str, list[str]] = {}
    for parameter in skeleton.loading_params:
        construct_name = parameter.get("construct")
        if isinstance(construct_name, str):
            measurement_parameter_names_by_construct.setdefault(construct_name, []).append(
                parameter["name"]
            )
    for parameter in skeleton.parameters:
        if parameter.get("role") != "measurement_error_sd":
            continue
        construct_name = parameter.get("construct")
        if isinstance(construct_name, str):
            measurement_parameter_names_by_construct.setdefault(construct_name, []).append(
                parameter["name"]
            )

    review_block = Stage4FrontierBlock(
        id="review:model_spec",
        kind="global_review",
        label="Model Specification",
        construct_names=tuple(construct_order),
        variable_names=tuple(item["variable"] for item in skeleton.ambiguous_indicators),
        parameter_names=tuple(
            sorted(
                (parameter["name"] for parameter in skeleton.loading_params),
                key=param_order.__getitem__,
            )
        ),
        payload={"reopenable_block_ids": tuple(block.id for block in model_blocks)},
    )

    prior_blocks: list[Stage4FrontierBlock] = []
    for construct_name in construct_order:
        names = measurement_parameter_names_by_construct.get(construct_name) or []
        if not names:
            continue
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"measurement:{construct_name}",
                kind="measurement_prior",
                label=construct_name,
                construct_names=(construct_name,),
                variable_names=tuple(grouped_indicators.get(construct_name) or ()),
                parameter_names=tuple(sorted(names, key=param_order.__getitem__)),
            )
        )

    observation_parameter_roles = {
        "observation_hyperparameter",
        "observation_hyperparameter_positive",
    }
    for parameter in skeleton.parameters:
        if parameter["role"] not in observation_parameter_roles:
            continue
        indicator_names = tuple(parameter.get("indicator_names") or ())
        construct_names = tuple(parameter.get("construct_names") or ())
        label = str(parameter.get("description") or parameter["name"])
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"observation:{parameter['name']}",
                kind="observation_prior",
                label=label,
                construct_names=construct_names,
                variable_names=indicator_names,
                parameter_names=(parameter["name"],),
                expand_neighbor_topology=False,
            )
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(construct_order)
    for edge in get_estimation_edges(causal_spec):
        graph.add_edge(edge["cause"], edge["effect"])
    order_lookup = {name: idx for idx, name in enumerate(construct_order)}
    dynamics_roles = {
        "ar_coefficient",
        "residual_sd",
        "initial_state_mean",
        "initial_state_sd",
    }
    scc_id_by_construct: dict[str, str] = {}
    scc_construct_names_by_id: dict[str, tuple[str, ...]] = {}
    for component in sorted(
        nx.strongly_connected_components(graph),
        key=lambda members: min(order_lookup[name] for name in members),
    ):
        ordered_members = tuple(name for name in construct_order if name in component)
        scc_id = "+".join(ordered_members)
        scc_construct_names_by_id[scc_id] = ordered_members
        for construct_name in ordered_members:
            scc_id_by_construct[construct_name] = scc_id
        names = [
            parameter["name"]
            for parameter in skeleton.parameters
            if parameter["role"] in dynamics_roles and parameter.get("construct") in component
        ]
        if not names:
            continue
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"dynamics:{'+'.join(ordered_members)}",
                kind="dynamics_prior",
                label=", ".join(ordered_members),
                construct_names=ordered_members,
                variable_names=tuple(
                    indicator_name
                    for construct_name in ordered_members
                    for indicator_name in grouped_indicators.get(construct_name, [])
                ),
                parameter_names=tuple(sorted(names, key=param_order.__getitem__)),
            )
        )

    effect_parameters_by_target: dict[str, list[dict[str, Any]]] = {}
    for parameter in skeleton.parameters:
        if parameter["role"] != "fixed_effect":
            continue
        effect_parameters_by_target.setdefault(parameter["effect"], []).append(parameter)
    for effect_name in construct_order:
        parameters = effect_parameters_by_target.get(effect_name) or []
        if not parameters:
            continue
        ordered_parameters = sorted(
            parameters, key=lambda parameter: param_order[parameter["name"]]
        )
        cause_names = tuple(dict.fromkeys(parameter["cause"] for parameter in ordered_parameters))
        construct_names = (*cause_names, effect_name)
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"effects:{effect_name}",
                kind="effect_prior",
                label=effect_name,
                construct_names=construct_names,
                variable_names=tuple(
                    indicator_name
                    for construct_name in construct_names
                    for indicator_name in grouped_indicators.get(construct_name, [])
                ),
                parameter_names=tuple(parameter["name"] for parameter in ordered_parameters),
                payload={
                    "target_construct": effect_name,
                    "cause_names": cause_names,
                },
            )
        )

    correlation_roles = {"correlation", "initial_state_correlation"}
    for parameter in skeleton.parameters:
        if parameter["role"] not in correlation_roles:
            continue
        construct_names = tuple(
            name
            for name in (parameter.get("construct_1"), parameter.get("construct_2"))
            if isinstance(name, str)
        )
        correlation_label = (
            f"{construct_names[0]} \u00d7 {construct_names[1]}"
            if len(construct_names) == 2
            else parameter["name"]
        )
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"correlation:{parameter['name']}",
                kind="correlation_prior",
                label=correlation_label,
                construct_names=construct_names,
                variable_names=tuple(
                    indicator_name
                    for construct_name in construct_names
                    for indicator_name in grouped_indicators.get(construct_name, [])
                ),
                parameter_names=(parameter["name"],),
            )
        )

    prior_review_block = Stage4FrontierBlock(
        id="review:prior_system",
        kind="global_prior_review",
        label="Full Prior System",
        construct_names=tuple(construct_order),
        variable_names=tuple(indicator["name"] for indicator in indicators),
        parameter_names=tuple(
            sorted(
                (parameter["name"] for parameter in skeleton.all_params),
                key=param_order.__getitem__,
            )
        ),
    )

    blocks_by_id = {
        block.id: block
        for block in [*model_blocks, review_block, *prior_blocks, prior_review_block]
    }
    parameter_to_block_id: dict[str, str] = {}
    indicator_to_decision_block_id: dict[str, str] = {}
    indicator_to_measurement_block_id: dict[str, str] = {}
    dynamics_block_id_by_construct: dict[str, str] = {}
    parameter_construct_names: dict[str, tuple[str, ...]] = {}
    reciprocal_parameter_by_parameter: dict[str, str] = {}
    effect_parameter_by_edge: dict[tuple[str, str], str] = {}

    for block in prior_blocks:
        for parameter_name in block.parameter_names:
            parameter_to_block_id.setdefault(parameter_name, block.id)
        if block.kind == "measurement_prior":
            for indicator_name in block.variable_names:
                indicator_to_measurement_block_id[indicator_name] = block.id
        if block.kind == "dynamics_prior":
            for construct_name in block.construct_names:
                dynamics_block_id_by_construct[construct_name] = block.id

    for block in model_blocks:
        for parameter_name in block.parameter_names:
            parameter_to_block_id.setdefault(parameter_name, block.id)
        if block.kind == "indicator_decision":
            for indicator_name in block.variable_names:
                indicator_to_decision_block_id[indicator_name] = block.id

    for parameter in skeleton.parameters:
        name = parameter["name"]
        role = parameter["role"]
        if role == "fixed_effect":
            cause_name = parameter.get("cause")
            effect_name = parameter.get("effect")
            construct_names = tuple(
                name_ for name_ in (cause_name, effect_name) if isinstance(name_, str)
            )
            parameter_construct_names[name] = construct_names
            if len(construct_names) == 2:
                effect_parameter_by_edge[(construct_names[0], construct_names[1])] = name
        elif (
            role
            in {
                "ar_coefficient",
                "residual_sd",
                "initial_state_mean",
                "initial_state_sd",
                "measurement_error_sd",
            }
            or role == "loading"
        ):
            construct_name = parameter.get("construct")
            if isinstance(construct_name, str):
                parameter_construct_names[name] = (construct_name,)
        elif role in {"observation_hyperparameter", "observation_hyperparameter_positive"}:
            construct_names = tuple(
                construct_name
                for construct_name in (parameter.get("construct_names") or ())
                if isinstance(construct_name, str)
            )
            if construct_names:
                parameter_construct_names[name] = construct_names
        elif role in {"correlation", "initial_state_correlation"}:
            construct_names = tuple(
                name_
                for name_ in (parameter.get("construct_1"), parameter.get("construct_2"))
                if isinstance(name_, str)
            )
            if construct_names:
                parameter_construct_names[name] = construct_names

    for (cause_name, effect_name), parameter_name in effect_parameter_by_edge.items():
        reciprocal_name = effect_parameter_by_edge.get((effect_name, cause_name))
        if reciprocal_name is None:
            continue
        reciprocal_parameter_by_parameter[parameter_name] = reciprocal_name

    internal_effect_block_ids_by_scc_id: dict[str, tuple[str, ...]] = {}
    for scc_id, construct_names in scc_construct_names_by_id.items():
        internal_blocks = tuple(
            block.id
            for block in prior_blocks
            if block.kind == "effect_prior"
            and isinstance(block.payload.get("target_construct"), str)
            and block.payload.get("target_construct") in construct_names
        )
        internal_effect_block_ids_by_scc_id[scc_id] = internal_blocks

    repair_topology = Stage4RepairTopology(
        parameter_to_block_id=parameter_to_block_id,
        indicator_to_decision_block_id=indicator_to_decision_block_id,
        indicator_to_measurement_block_id=indicator_to_measurement_block_id,
        indicator_names_by_construct={
            construct_name: tuple(grouped_indicators.get(construct_name, ()))
            for construct_name in construct_order
        },
        dynamics_block_id_by_construct=dynamics_block_id_by_construct,
        scc_id_by_construct=scc_id_by_construct,
        scc_construct_names_by_id=scc_construct_names_by_id,
        internal_effect_block_ids_by_scc_id=internal_effect_block_ids_by_scc_id,
        reciprocal_parameter_by_parameter=reciprocal_parameter_by_parameter,
        parameter_construct_names=parameter_construct_names,
    )

    return Stage4Plan(
        model_blocks=tuple(model_blocks),
        review_block=review_block,
        prior_blocks=tuple(prior_blocks),
        prior_review_block=prior_review_block,
        blocks_by_id=blocks_by_id,
        repair_topology=repair_topology,
    )
