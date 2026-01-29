"""DSEM domain schemas following Anderson & Gerbing two-step approach.

Separates:
1. LatentModel - theoretical constructs + causal edges (theory-driven)
2. MeasurementModel - observed indicators that reflect constructs (data-driven)
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from dsem_agent.utils.aggregations import AGGREGATION_REGISTRY


class Role(str, Enum):
    """Whether a variable is modeled (endogenous) or given (exogenous)."""

    ENDOGENOUS = "endogenous"  # Has inbound edges, is modeled
    EXOGENOUS = "exogenous"  # No inbound edges, given/external


class TemporalStatus(str, Enum):
    """Whether a variable changes over time."""

    TIME_VARYING = "time_varying"  # Changes within person over time
    TIME_INVARIANT = "time_invariant"  # Fixed for each person


# Hours per granularity unit
GRANULARITY_HOURS = {
    "hourly": 1,
    "daily": 24,
    "weekly": 168,
    "monthly": 720,  # 30 days
    "yearly": 8760,
}


# ══════════════════════════════════════════════════════════════════════════════
# LATENT MODEL (theoretical - what exists and how it relates)
# ══════════════════════════════════════════════════════════════════════════════


class Construct(BaseModel):
    """A theoretical entity in the causal model.

    Constructs are conceptually 'latent' - they represent theoretical entities
    that may be measured by one or more observed indicators.
    """

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
    causal_granularity: str | None = Field(
        default=None,
        description=(
            "'hourly', 'daily', 'weekly', 'monthly', 'yearly'. Required for time-varying constructs. "
            "The timescale at which causal dynamics operate."
        ),
    )

    @model_validator(mode="after")
    def validate_construct(self):
        """Validate construct field consistency."""
        is_time_varying = self.temporal_status == TemporalStatus.TIME_VARYING

        if is_time_varying:
            if self.causal_granularity is None:
                raise ValueError(
                    f"Time-varying construct '{self.name}' requires causal_granularity"
                )
            if self.causal_granularity not in GRANULARITY_HOURS:
                raise ValueError(
                    f"Invalid causal_granularity '{self.causal_granularity}' for '{self.name}'. "
                    f"Must be one of: {', '.join(sorted(GRANULARITY_HOURS.keys()))}"
                )
        else:
            if self.causal_granularity is not None:
                raise ValueError(
                    f"Time-invariant construct '{self.name}' must not have causal_granularity"
                )

        # Outcomes must be endogenous
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
            "If True, effect at t is caused by cause at t-1. "
            "If False (contemporaneous), effect at t is caused by cause at t. "
            "Cross-timescale edges are always lagged."
        ),
    )


class LatentModel(BaseModel):
    """Theoretical causal structure over constructs (the latent model).

    This is the output of Stage 1a - proposed based on domain knowledge alone,
    without seeing data. Defines the topological structure among latent constructs.
    """

    constructs: list[Construct] = Field(description="Theoretical constructs in the model")
    edges: list[CausalEdge] = Field(description="Causal edges between constructs")

    @model_validator(mode="after")
    def validate_latent_model(self):
        """Validate latent model constraints."""
        # Exactly one outcome required
        outcomes = [c for c in self.constructs if c.is_outcome]
        if len(outcomes) == 0:
            raise ValueError("Exactly one construct must have is_outcome=true")
        if len(outcomes) > 1:
            names = [c.name for c in outcomes]
            raise ValueError(f"Only one outcome allowed, got {len(outcomes)}: {names}")

        construct_map = {c.name: c for c in self.constructs}

        for edge in self.edges:
            # Check constructs exist
            if edge.cause not in construct_map:
                raise ValueError(f"Edge cause '{edge.cause}' not in constructs")
            if edge.effect not in construct_map:
                raise ValueError(f"Edge effect '{edge.effect}' not in constructs")

            cause_construct = construct_map[edge.cause]
            effect_construct = construct_map[edge.effect]

            # No inbound edges to exogenous
            if effect_construct.role == Role.EXOGENOUS:
                raise ValueError(f"Exogenous construct '{edge.effect}' cannot be an effect")

            cause_gran = cause_construct.causal_granularity
            effect_gran = effect_construct.causal_granularity

            # Contemporaneous (lagged=False) requires same timescale
            # Exception: time-invariant causes (granularity=None) can affect any timescale
            both_time_varying = cause_gran is not None and effect_gran is not None
            if not edge.lagged and both_time_varying and cause_gran != effect_gran:
                raise ValueError(
                    f"Contemporaneous edge (lagged=false) requires same timescale: "
                    f"{edge.cause} ({cause_gran}) -> {edge.effect} ({effect_gran})"
                )

        # Check acyclicity within time slice (contemporaneous edges only)
        contemporaneous_edges = [(e.cause, e.effect) for e in self.edges if not e.lagged]
        if contemporaneous_edges:
            import networkx as nx

            G = nx.DiGraph(contemporaneous_edges)
            if not nx.is_directed_acyclic_graph(G):
                cycles = list(nx.simple_cycles(G))
                raise ValueError(
                    f"Contemporaneous edges form cycle(s) within time slice: {cycles}. "
                    "Use lagged=true for feedback loops across time."
                )

        return self

    def to_networkx(self):
        """Convert to NetworkX DiGraph."""
        import networkx as nx

        G = nx.DiGraph()
        for construct in self.constructs:
            G.add_node(construct.name, **construct.model_dump())
        for edge in self.edges:
            G.add_edge(
                edge.cause,
                edge.effect,
                description=edge.description,
                lagged=edge.lagged,
            )
        return G


# ══════════════════════════════════════════════════════════════════════════════
# MEASUREMENT MODEL (operational - how constructs are observed)
# ══════════════════════════════════════════════════════════════════════════════


class Indicator(BaseModel):
    """An observed variable that reflects a construct.

    Following the reflective measurement model (A1), causality flows from
    construct to indicator: the latent construct causes the observed values.
    """

    model_config = {"populate_by_name": True}

    name: str = Field(description="Indicator name (e.g., 'hrv', 'self_reported_stress')")
    construct_name: str = Field(
        description="Which construct this indicator measures",
        alias="construct",
    )
    how_to_measure: str = Field(
        description="Instructions for workers on how to extract this from data"
    )
    measurement_granularity: str = Field(
        description=(
            "'finest' (one datapoint per raw entry) or 'hourly', 'daily', 'weekly', 'monthly', 'yearly'. "
            "The resolution at which workers extract measurements."
        ),
    )
    measurement_dtype: str = Field(
        description="'continuous', 'binary', 'count', 'ordinal', 'categorical'"
    )
    aggregation: str = Field(
        description=f"Aggregation function to collapse to causal_granularity. Available: {', '.join(sorted(AGGREGATION_REGISTRY.keys()))}",
    )

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str) -> str:
        if v not in AGGREGATION_REGISTRY:
            available = ", ".join(sorted(AGGREGATION_REGISTRY.keys()))
            raise ValueError(f"Unknown aggregation '{v}'. Available: {available}")
        return v

    @field_validator("measurement_granularity")
    @classmethod
    def validate_measurement_granularity(cls, v: str) -> str:
        valid = {"finest"} | set(GRANULARITY_HOURS.keys())
        if v not in valid:
            raise ValueError(
                f"Invalid measurement_granularity '{v}'. "
                f"Must be 'finest' or one of: {', '.join(sorted(GRANULARITY_HOURS.keys()))}"
            )
        return v

    @field_validator("measurement_dtype")
    @classmethod
    def validate_measurement_dtype(cls, v: str) -> str:
        valid = {"continuous", "binary", "count", "ordinal", "categorical"}
        if v not in valid:
            raise ValueError(f"Invalid measurement_dtype '{v}'. Must be one of: {', '.join(sorted(valid))}")
        return v


class MeasurementModel(BaseModel):
    """Operationalization of constructs into observed indicators.

    This is the output of Stage 1b - proposed after seeing data sample,
    given the latent model from Stage 1a.

    Each construct from the latent model must have at least one indicator.
    """

    indicators: list[Indicator] = Field(
        description="Observed indicators, each measuring a construct"
    )

    def get_indicators_for_construct(self, construct_name: str) -> list[Indicator]:
        """Get all indicators that measure a given construct."""
        return [i for i in self.indicators if i.construct_name == construct_name]


# ══════════════════════════════════════════════════════════════════════════════
# FULL DSEM MODEL (composition of latent + measurement)
# ══════════════════════════════════════════════════════════════════════════════


def compute_lag_hours(
    cause_granularity: str | None,
    effect_granularity: str | None,
    lagged: bool,
) -> int:
    """Compute lag in hours based on granularities and lagged flag.

    Rules (Markov property):
    - Same timescale, contemporaneous: lag = 0
    - Same timescale, lagged: lag = 1 granularity unit
    - Cross timescale: lag = coarser granularity (always lagged)
    """
    cause_hours = GRANULARITY_HOURS.get(cause_granularity, 0) if cause_granularity else 0
    effect_hours = GRANULARITY_HOURS.get(effect_granularity, 0) if effect_granularity else 0

    # Cross-timescale: always use coarser granularity
    if cause_granularity != effect_granularity:
        return max(cause_hours, effect_hours)

    # Same timescale: depends on lagged flag
    if lagged:
        return cause_hours  # 1 unit of the shared granularity
    return 0  # contemporaneous


class IdentifiabilityStatus(BaseModel):
    """Status of causal effect identifiability.

    Since there's exactly one outcome in the model, we only track which
    treatments have identifiable effects (not treatment-outcome pairs).
    """

    outcome: str = Field(description="The outcome construct name")
    all_identifiable: bool = Field(description="Whether all treatment effects are identifiable")
    non_identifiable_treatments: set[str] = Field(
        default_factory=set,
        description="Treatments whose effect on outcome is NOT identifiable"
    )
    blocking_confounders: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map of non-identifiable treatment to blocking confounders"
    )


class DSEMModel(BaseModel):
    """Complete DSEM specification combining latent and measurement models.

    This is the full model after both Stage 1a (latent) and Stage 1b (measurement).
    Includes identifiability status for target causal effects.
    """

    latent: LatentModel = Field(description="Theoretical causal structure (topological)")
    measurement: MeasurementModel = Field(description="Operationalization into indicators")
    identifiability: IdentifiabilityStatus | None = Field(
        default=None,
        description="Identifiability status of target causal effects"
    )

    @model_validator(mode="after")
    def validate_dsem_model(self):
        """Validate that measurement model covers all constructs."""
        construct_names = {c.name for c in self.latent.constructs}
        measured_constructs = {i.construct_name for i in self.measurement.indicators}

        # Check all indicator references are valid
        for indicator in self.measurement.indicators:
            if indicator.construct_name not in construct_names:
                raise ValueError(
                    f"Indicator '{indicator.name}' references unknown construct '{indicator.construct_name}'"
                )

            # Validate measurement_granularity vs causal_granularity
            construct = next(c for c in self.latent.constructs if c.name == indicator.construct_name)
            if construct.temporal_status == TemporalStatus.TIME_VARYING:
                causal_gran = construct.causal_granularity
                meas_gran = indicator.measurement_granularity
                # measurement_granularity must be finer than or equal to causal_granularity
                if meas_gran != "finest" and causal_gran is not None:
                    meas_hours = GRANULARITY_HOURS.get(meas_gran, 0)
                    causal_hours = GRANULARITY_HOURS.get(causal_gran, 0)
                    if meas_hours > causal_hours:
                        raise ValueError(
                            f"Indicator '{indicator.name}' has measurement_granularity '{meas_gran}' "
                            f"coarser than construct '{construct.name}' causal_granularity '{causal_gran}'"
                        )

        # TODO: A2 validation commented out. Whether a latent construct without indicators
        # is problematic depends on the topological structure (e.g., is it a confounder on
        # a backdoor path?).
        # This should be checked by DoWhy in Stage 3 using Pearl's identification rules,
        # not enforced here unconditionally.
        #
        # for construct in self.latent.constructs:
        #     if construct.temporal_status == TemporalStatus.TIME_VARYING:
        #         if construct.name not in measured_constructs:
        #             raise ValueError(
        #                 f"Time-varying construct '{construct.name}' has no indicators. "
        #                 "Per A2, latent time-varying constructs require at least one indicator."
        #             )

        return self

    def get_edge_lag_hours(self, edge: CausalEdge) -> int:
        """Compute lag in hours for a causal edge."""
        construct_map = {c.name: c for c in self.latent.constructs}
        cause = construct_map[edge.cause]
        effect = construct_map[edge.effect]
        return compute_lag_hours(
            cause.causal_granularity,
            effect.causal_granularity,
            edge.lagged,
        )

    def to_networkx(self):
        """Convert to NetworkX DiGraph with computed lag_hours."""
        import networkx as nx

        G = nx.DiGraph()

        # Add construct nodes
        for construct in self.latent.constructs:
            G.add_node(construct.name, node_type="construct", **construct.model_dump())

        # Add indicator nodes
        for indicator in self.measurement.indicators:
            G.add_node(indicator.name, node_type="indicator", **indicator.model_dump())
            # Add loading edge: construct → indicator (reflective model)
            G.add_edge(indicator.construct_name, indicator.name, edge_type="loading")

        # Add causal edges with computed lag_hours
        for edge in self.latent.edges:
            G.add_edge(
                edge.cause,
                edge.effect,
                edge_type="causal",
                description=edge.description,
                lagged=edge.lagged,
                lag_hours=self.get_edge_lag_hours(edge),
            )

        return G


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS (for LLM tool use)
# ══════════════════════════════════════════════════════════════════════════════


def validate_latent_model(data: dict) -> tuple[LatentModel | None, list[str]]:
    """Validate a latent model dict, collecting ALL errors.

    Args:
        data: Dictionary to validate as LatentModel

    Returns:
        Tuple of (validated model or None, list of error messages)
    """
    errors = []

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

    # Validate each construct individually
    valid_constructs = []
    construct_names = set()

    for i, construct_data in enumerate(constructs):
        if not isinstance(construct_data, dict):
            errors.append(f"constructs[{i}]: must be a dictionary")
            continue

        name = construct_data.get("name", f"<unnamed_{i}>")

        if name in construct_names:
            errors.append(f"Duplicate construct name: '{name}'")
        construct_names.add(name)

        try:
            construct = Construct.model_validate(construct_data)
            valid_constructs.append(construct)
        except Exception as e:
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                for line in error_msg.split("\n")[1:]:
                    line = line.strip()
                    if line and not line.startswith("For further"):
                        errors.append(f"constructs[{i}] ({name}): {line}")
            else:
                errors.append(f"constructs[{i}] ({name}): {error_msg}")

    # Build construct map for edge validation
    construct_map = {c.name: c for c in valid_constructs}

    # Validate each edge individually
    valid_edges = []
    for i, edge_data in enumerate(edges):
        if not isinstance(edge_data, dict):
            errors.append(f"edges[{i}]: must be a dictionary")
            continue

        cause = edge_data.get("cause", "<missing>")
        effect = edge_data.get("effect", "<missing>")
        edge_label = f"edges[{i}] ({cause} -> {effect})"

        try:
            edge = CausalEdge.model_validate(edge_data)
        except Exception as e:
            errors.append(f"{edge_label}: {e}")
            continue

        if edge.cause not in construct_map:
            errors.append(f"{edge_label}: cause '{edge.cause}' not in constructs")
            continue
        if edge.effect not in construct_map:
            errors.append(f"{edge_label}: effect '{edge.effect}' not in constructs")
            continue

        cause_construct = construct_map[edge.cause]
        effect_construct = construct_map[edge.effect]

        if effect_construct.role == Role.EXOGENOUS:
            errors.append(f"{edge_label}: exogenous construct '{edge.effect}' cannot be an effect")
            continue

        cause_gran = cause_construct.causal_granularity
        effect_gran = effect_construct.causal_granularity
        both_time_varying = cause_gran is not None and effect_gran is not None
        if not edge.lagged and both_time_varying and cause_gran != effect_gran:
            errors.append(
                f"{edge_label}: contemporaneous edge requires same timescale, "
                f"got {cause_gran} -> {effect_gran}"
            )
            continue

        valid_edges.append(edge)

    # Check outcome constraints
    outcomes = [c for c in valid_constructs if c.is_outcome]
    if len(outcomes) == 0:
        errors.append("Exactly one construct must have is_outcome=true (none found)")
    elif len(outcomes) > 1:
        names = [c.name for c in outcomes]
        errors.append(f"Only one outcome allowed, got {len(outcomes)}: {names}")

    # Check acyclicity of contemporaneous edges
    contemporaneous_edges = [(e.cause, e.effect) for e in valid_edges if not e.lagged]
    if contemporaneous_edges:
        import networkx as nx

        G = nx.DiGraph(contemporaneous_edges)
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            errors.append(
                f"Contemporaneous edges form cycle(s): {cycles}. "
                "Use lagged=true for feedback loops."
            )

    if not errors:
        try:
            model = LatentModel(constructs=valid_constructs, edges=valid_edges)
            return model, []
        except Exception as e:
            errors.append(f"Final validation failed: {e}")

    return None, errors


def validate_measurement_model(
    data: dict,
    latent: LatentModel,
) -> tuple[MeasurementModel | None, list[str]]:
    """Validate a measurement model dict against a latent model.

    Args:
        data: Dictionary to validate as MeasurementModel
        latent: The latent model this measurement model operationalizes

    Returns:
        Tuple of (validated model or None, list of error messages)
    """
    errors = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    indicators = data.get("indicators", [])

    if not isinstance(indicators, list):
        errors.append("'indicators' must be a list")
        indicators = []

    construct_names = {c.name for c in latent.constructs}
    construct_map = {c.name: c for c in latent.constructs}

    # Validate each indicator
    valid_indicators = []
    indicator_names = set()

    for i, indicator_data in enumerate(indicators):
        if not isinstance(indicator_data, dict):
            errors.append(f"indicators[{i}]: must be a dictionary")
            continue

        name = indicator_data.get("name", f"<unnamed_{i}>")

        if name in indicator_names:
            errors.append(f"Duplicate indicator name: '{name}'")
        indicator_names.add(name)

        try:
            indicator = Indicator.model_validate(indicator_data)
        except Exception as e:
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                for line in error_msg.split("\n")[1:]:
                    line = line.strip()
                    if line and not line.startswith("For further"):
                        errors.append(f"indicators[{i}] ({name}): {line}")
            else:
                errors.append(f"indicators[{i}] ({name}): {error_msg}")
            continue

        # Check construct reference
        if indicator.construct_name not in construct_names:
            errors.append(
                f"indicators[{i}] ({name}): references unknown construct '{indicator.construct_name}'"
            )
            continue

        # Check granularity compatibility
        construct = construct_map[indicator.construct_name]
        if construct.temporal_status == TemporalStatus.TIME_VARYING:
            causal_gran = construct.causal_granularity
            meas_gran = indicator.measurement_granularity
            if meas_gran != "finest" and causal_gran is not None:
                meas_hours = GRANULARITY_HOURS.get(meas_gran, 0)
                causal_hours = GRANULARITY_HOURS.get(causal_gran, 0)
                if meas_hours > causal_hours:
                    errors.append(
                        f"indicators[{i}] ({name}): measurement_granularity '{meas_gran}' "
                        f"coarser than construct '{construct.name}' causal_granularity '{causal_gran}'"
                    )
                    continue

        valid_indicators.append(indicator)

    # TODO: A2 validation commented out. Whether a latent construct without indicators
    # is problematic depends on the topological structure (e.g., is it a confounder on
    # a backdoor path?). This should be checked by DoWhy in Stage 3 using Pearl's
    # identification rules, not enforced here unconditionally.
    #
    # measured_constructs = {i.construct_name for i in valid_indicators}
    # for construct in latent.constructs:
    #     if construct.temporal_status == TemporalStatus.TIME_VARYING:
    #         if construct.name not in measured_constructs:
    #             errors.append(
    #                 f"Time-varying construct '{construct.name}' has no indicators. "
    #                 "Per A2, time-varying constructs require at least one indicator."
    #             )

    if not errors:
        try:
            model = MeasurementModel(indicators=valid_indicators)
            return model, []
        except Exception as e:
            errors.append(f"Final validation failed: {e}")

    return None, errors


def validate_dsem_model(
    latent_data: dict,
    measurement_data: dict,
) -> tuple[DSEMModel | None, list[str]]:
    """Validate both latent and measurement models together.

    Args:
        latent_data: Dictionary to validate as LatentModel
        measurement_data: Dictionary to validate as MeasurementModel

    Returns:
        Tuple of (validated DSEMModel or None, list of error messages)
    """
    latent, latent_errors = validate_latent_model(latent_data)
    if latent is None:
        return None, ["Latent model errors:"] + latent_errors

    measurement, measurement_errors = validate_measurement_model(measurement_data, latent)
    if measurement is None:
        return None, ["Measurement model errors:"] + measurement_errors

    try:
        model = DSEMModel(latent=latent, measurement=measurement)
        return model, []
    except Exception as e:
        return None, [f"DSEMModel validation failed: {e}"]
