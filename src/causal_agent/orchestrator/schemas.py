from pydantic import BaseModel, Field


class Dimension(BaseModel):
    """A candidate dimension/variable for the causal model."""

    name: str = Field(description="Variable name (snake_case)")
    description: str = Field(description="What this variable represents")
    dtype: str = Field(description="Data type: 'continuous', 'categorical', 'binary', 'ordinal'")
    time_granularity: str = Field(
        description="Time granularity: 'hourly', 'daily', 'weekly', 'monthly', 'yearly', or 'none'"
    )
    autocorrelated: bool = Field(description="Whether this variable has temporal autocorrelation")


class CausalEdge(BaseModel):
    """A directed edge in the causal DAG."""

    cause: str = Field(description="The cause variable (parent node)")
    effect: str = Field(description="The effect variable (child node)")


class CrossLaggedEdge(BaseModel):
    """A cross-lagged causal edge: cause at t-lag affects effect at t."""

    cause: str = Field(description="The cause variable")
    effect: str = Field(description="The effect variable")
    lag: int = Field(description="Time lag (e.g., 1 means cause at t-1 affects effect at t)")


class ProposedStructure(BaseModel):
    """Output schema for the structure proposal stage."""

    dimensions: list[Dimension] = Field(
        description="Candidate variables/dimensions to extract from data"
    )
    edges: list[CausalEdge] = Field(
        description="Contemporaneous causal edges (same time point)"
    )
    cross_lags: list[CrossLaggedEdge] = Field(
        description="Cross-lagged causal edges (cause at t-lag affects effect at t)"
    )

    def to_networkx(self):
        """Convert to NetworkX DiGraph (compatible with DoWhy)."""
        import networkx as nx

        G = nx.DiGraph()
        # Add all dimension nodes
        for dim in self.dimensions:
            G.add_node(dim.name, **dim.model_dump())
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.cause, edge.effect)
        return G

    def to_edge_list(self) -> list[tuple[str, str]]:
        """Convert to edge list format."""
        return [(e.cause, e.effect) for e in self.edges]
