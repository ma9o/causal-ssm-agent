from pydantic import BaseModel, Field, model_validator


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
    time_granularity: str | None = Field(
        description="'hourly', 'daily', 'weekly', 'monthly', 'yearly', or None for time-invariant"
    )
    dtype: str = Field(description="'continuous', 'binary', 'ordinal', 'categorical'")
    role: str = Field(description="'endogenous' or 'exogenous'")
    is_latent: bool = Field(
        default=False,
        description="True for random effects. Only valid when role='exogenous' and time_granularity=None",
    )
    aggregation: str | None = Field(
        default=None,
        description="Aggregation function: 'mean', 'sum', 'max', 'min', 'last', 'variance'. Required when raw data is finer than time_granularity.",
    )

    @model_validator(mode="after")
    def validate_dimension(self):
        # Latent validity: is_latent requires exogenous + time-invariant
        if self.is_latent:
            if self.role != "exogenous":
                raise ValueError("is_latent=True requires role='exogenous'")
            if self.time_granularity is not None:
                raise ValueError("is_latent=True requires time_granularity=None")

        # Endogenous requires time-varying
        if self.role == "endogenous" and self.time_granularity is None:
            raise ValueError("Endogenous variables must be time-varying (time_granularity cannot be None)")

        return self


class CausalEdge(BaseModel):
    """A directed causal edge between variables."""

    cause: str = Field(description="Name of cause variable")
    effect: str = Field(description="Name of effect variable")
    lag: int = Field(
        description=(
            "Lag in hours. Same timescale: 0 (contemporaneous) or 1 granularity unit. "
            "Cross-timescale: must equal the coarser variable's granularity in hours."
        )
    )
    aggregation: str | None = Field(
        default=None,
        description="Required when cause is finer-grained than effect: 'mean', 'sum', 'max', 'min', 'last', 'variance'.",
    )


class DSEMStructure(BaseModel):
    """Complete DSEM specification."""

    dimensions: list[Dimension] = Field(description="Variables in the model")
    edges: list[CausalEdge] = Field(description="Causal edges including cross-lags")

    @model_validator(mode="after")
    def validate_structure(self):
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
            if effect_dim.role == "exogenous":
                raise ValueError(f"Exogenous variable '{edge.effect}' cannot be an effect")

            cause_gran = cause_dim.time_granularity
            effect_gran = effect_dim.time_granularity

            # Contemporaneous requires same timescale
            if edge.lag == 0:
                if cause_gran != effect_gran:
                    raise ValueError(
                        f"Contemporaneous edge (lag=0) requires same timescale: "
                        f"{edge.cause} ({cause_gran}) -> {edge.effect} ({effect_gran})"
                    )

            # Same-scale lag constraint (Markov): must be 0 or exactly 1 unit
            if cause_gran == effect_gran and cause_gran is not None:
                unit_hours = GRANULARITY_HOURS.get(cause_gran, 0)
                if edge.lag != 0 and edge.lag != unit_hours:
                    raise ValueError(
                        f"Same-timescale edge must have lag=0 or lag={unit_hours}h (1 {cause_gran} unit): "
                        f"{edge.cause} -> {edge.effect} has lag={edge.lag}"
                    )

            # Cross-scale lag constraint (Markov): lag = coarser granularity
            if cause_gran != effect_gran and cause_gran is not None and effect_gran is not None:
                cause_hours = GRANULARITY_HOURS.get(cause_gran, 0)
                effect_hours = GRANULARITY_HOURS.get(effect_gran, 0)
                required_lag = max(cause_hours, effect_hours)
                if edge.lag != required_lag:
                    raise ValueError(
                        f"Cross-timescale edge must have lag={required_lag}h: "
                        f"{edge.cause} ({cause_gran}) -> {edge.effect} ({effect_gran}) has lag={edge.lag}"
                    )

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

        return self

    def to_networkx(self):
        """Convert to NetworkX DiGraph (compatible with DoWhy)."""
        import networkx as nx

        G = nx.DiGraph()
        for dim in self.dimensions:
            G.add_node(dim.name, **dim.model_dump())
        for edge in self.edges:
            G.add_edge(edge.cause, edge.effect, lag=edge.lag, aggregation=edge.aggregation)
        return G

    def to_edge_list(self) -> list[tuple[str, str]]:
        """Convert to edge list format."""
        return [(e.cause, e.effect) for e in self.edges]
