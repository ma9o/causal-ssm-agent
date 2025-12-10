from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from causal_agent.utils.aggregations import AGGREGATION_REGISTRY


class Role(str, Enum):
    """Whether a variable is modeled (endogenous) or given (exogenous)."""

    ENDOGENOUS = "endogenous"  # Has inbound edges, is modeled
    EXOGENOUS = "exogenous"  # No inbound edges, given/external


class Observability(str, Enum):
    """Whether a variable is directly measurable."""

    OBSERVED = "observed"  # Directly measured in data
    LATENT = "latent"  # Not directly measured, inferred


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


class Dimension(BaseModel):
    """A variable in the causal model."""

    name: str = Field(description="Variable name (e.g., 'sleep_quality')")
    description: str = Field(description="What this variable represents")
    role: Role = Field(description="'endogenous' (modeled) or 'exogenous' (given)")
    observability: Observability = Field(description="'observed' (measured) or 'latent' (inferred)")
    temporal_status: TemporalStatus = Field(
        description="'time_varying' (changes over time) or 'time_invariant' (fixed)"
    )
    causal_granularity: str | None = Field(
        default=None,
        description=(
            "'hourly', 'daily', 'weekly', 'monthly', 'yearly'. Required for time-varying variables. "
            "The granularity at which causal relationships make sense."
        ),
    )
    base_dtype: str = Field(
        description="'continuous', 'binary', 'count', 'ordinal', 'categorical'"
    )
    aggregation: str | None = Field(
        default=None,
        description=f"Aggregation function from registry. Available: {', '.join(sorted(AGGREGATION_REGISTRY.keys()))}",
    )

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str | None) -> str | None:
        if v is not None and v not in AGGREGATION_REGISTRY:
            available = ", ".join(sorted(AGGREGATION_REGISTRY.keys()))
            raise ValueError(f"Unknown aggregation '{v}'. Available: {available}")
        return v

    @model_validator(mode="after")
    def validate_field_consistency(self):
        """Validate that causal_granularity and aggregation are consistent with temporal_status."""
        is_time_varying = self.temporal_status == TemporalStatus.TIME_VARYING

        if is_time_varying:
            if self.causal_granularity is None:
                raise ValueError(
                    f"Time-varying variable '{self.name}' requires causal_granularity"
                )
            if self.aggregation is None:
                raise ValueError(
                    f"Time-varying variable '{self.name}' requires aggregation (how to collapse raw data)"
                )
        else:
            if self.causal_granularity is not None:
                raise ValueError(
                    f"Time-invariant variable '{self.name}' must not have causal_granularity"
                )
            if self.aggregation is not None:
                raise ValueError(
                    f"Time-invariant variable '{self.name}' must not have aggregation"
                )

        return self


class CausalEdge(BaseModel):
    """A directed causal edge between variables."""

    cause: str = Field(description="Name of cause variable")
    effect: str = Field(description="Name of effect variable")
    description: str = Field(description="Why this causal relationship exists")
    lagged: bool = Field(
        default=True,
        description=(
            "If True, effect at t is caused by cause at t-1. "
            "If False (contemporaneous), effect at t is caused by cause at t. "
            "Cross-timescale edges are always lagged."
        ),
    )
    aggregation: str | None = Field(
        default=None,
        description=f"Required when cause is finer-grained than effect. Available: {', '.join(sorted(AGGREGATION_REGISTRY.keys()))}",
    )
    # Computed field - set by DSEMStructure validator
    lag_hours: int | None = Field(
        default=None,
        description="Lag in hours. Computed from granularities - do not set manually.",
    )

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str | None) -> str | None:
        if v is not None and v not in AGGREGATION_REGISTRY:
            available = ", ".join(sorted(AGGREGATION_REGISTRY.keys()))
            raise ValueError(f"Unknown aggregation '{v}'. Available: {available}")
        return v


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


class DSEMStructure(BaseModel):
    """Complete DSEM specification."""

    dimensions: list[Dimension] = Field(description="Variables in the model")
    edges: list[CausalEdge] = Field(description="Causal edges including cross-lags")

    @model_validator(mode="after")
    def validate_and_compute_lags(self):
        """Validate structure and compute lag_hours for each edge."""
        dim_map = {d.name: d for d in self.dimensions}

        for edge in self.edges:
            # Check variables exist
            if edge.cause not in dim_map:
                raise ValueError(f"Edge cause '{edge.cause}' not in dimensions")
            if edge.effect not in dim_map:
                raise ValueError(f"Edge effect '{edge.effect}' not in dimensions")

            cause_dim = dim_map[edge.cause]
            effect_dim = dim_map[edge.effect]

            # No inbound edges to exogenous
            if effect_dim.role == Role.EXOGENOUS:
                raise ValueError(f"Exogenous variable '{edge.effect}' cannot be an effect")

            cause_gran = cause_dim.causal_granularity
            effect_gran = effect_dim.causal_granularity

            # Contemporaneous (lagged=False) requires same timescale
            if not edge.lagged and cause_gran != effect_gran:
                raise ValueError(
                    f"Contemporaneous edge (lagged=false) requires same timescale: "
                    f"{edge.cause} ({cause_gran}) -> {edge.effect} ({effect_gran})"
                )

            # Cross-timescale aggregation rules
            if cause_gran != effect_gran and cause_gran is not None and effect_gran is not None:
                cause_hours = GRANULARITY_HOURS.get(cause_gran, 0)
                effect_hours = GRANULARITY_HOURS.get(effect_gran, 0)

                # Aggregation required when finer cause -> coarser effect
                if cause_hours < effect_hours and edge.aggregation is None:
                    raise ValueError(
                        f"Aggregation required for finer->coarser edge: "
                        f"{edge.cause} ({cause_gran}) -> {edge.effect} ({effect_gran})"
                    )

                # Aggregation prohibited when coarser cause -> finer effect
                if cause_hours >= effect_hours and edge.aggregation is not None:
                    raise ValueError(
                        f"Aggregation not allowed for coarser->finer edge: "
                        f"{edge.cause} ({cause_gran}) -> {edge.effect} ({effect_gran})"
                    )

            # Compute and set lag_hours
            edge.lag_hours = compute_lag_hours(cause_gran, effect_gran, edge.lagged)

        return self

    def to_networkx(self):
        """Convert to NetworkX DiGraph (compatible with DoWhy)."""
        import networkx as nx

        G = nx.DiGraph()
        for dim in self.dimensions:
            G.add_node(dim.name, **dim.model_dump())
        for edge in self.edges:
            G.add_edge(
                edge.cause,
                edge.effect,
                description=edge.description,
                lag_hours=edge.lag_hours,
                lagged=edge.lagged,
                aggregation=edge.aggregation,
            )
        return G

    def to_edge_list(self) -> list[tuple[str, str]]:
        """Convert to edge list format."""
        return [(e.cause, e.effect) for e in self.edges]
