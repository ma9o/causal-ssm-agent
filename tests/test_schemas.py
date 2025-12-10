"""Tests for DSEM schema validation."""

import pytest

from causal_agent.orchestrator.schemas import (
    CausalEdge,
    Dimension,
    DSEMStructure,
    GRANULARITY_HOURS,
    VariableType,
    compute_lag_hours,
)


class TestDimension:
    """Tests for Dimension validation."""

    def test_outcome_derives_endogenous(self):
        """Outcome type derives role=endogenous, is_latent=False."""
        dim = Dimension(
            name="mood",
            description="Daily mood rating",
            variable_type=VariableType.OUTCOME,
            causal_granularity="daily",
            base_dtype="continuous",
        )
        assert dim.role == "endogenous"
        assert dim.is_latent is False

    def test_input_derives_exogenous(self):
        """Input type derives role=exogenous, is_latent=False."""
        dim = Dimension(
            name="weather",
            description="Daily temperature",
            variable_type=VariableType.INPUT,
            causal_granularity="daily",
            base_dtype="continuous",
        )
        assert dim.role == "exogenous"
        assert dim.is_latent is False

    def test_covariate_derives_exogenous_time_invariant(self):
        """Covariate type derives role=exogenous, is_latent=False, no causal_granularity."""
        dim = Dimension(
            name="age",
            description="Participant age",
            variable_type=VariableType.COVARIATE,
            base_dtype="continuous",
        )
        assert dim.role == "exogenous"
        assert dim.is_latent is False
        assert dim.causal_granularity is None

    def test_random_effect_derives_latent(self):
        """Random effect type derives role=exogenous, is_latent=True."""
        dim = Dimension(
            name="person_intercept",
            description="Person-specific baseline",
            variable_type=VariableType.RANDOM_EFFECT,
            base_dtype="continuous",
        )
        assert dim.role == "exogenous"
        assert dim.is_latent is True
        assert dim.causal_granularity is None

    def test_outcome_requires_causal_granularity(self):
        """Outcome type requires causal_granularity."""
        with pytest.raises(ValueError, match="Outcome .* requires causal_granularity"):
            Dimension(
                name="mood",
                description="Invalid",
                variable_type=VariableType.OUTCOME,
                base_dtype="continuous",
            )

    def test_input_requires_causal_granularity(self):
        """Input type requires causal_granularity."""
        with pytest.raises(ValueError, match="Input .* requires causal_granularity"):
            Dimension(
                name="weather",
                description="Invalid",
                variable_type=VariableType.INPUT,
                base_dtype="continuous",
            )

    def test_covariate_forbids_causal_granularity(self):
        """Covariate type must not have causal_granularity."""
        with pytest.raises(ValueError, match="Covariate .* must not have causal_granularity"):
            Dimension(
                name="age",
                description="Invalid",
                variable_type=VariableType.COVARIATE,
                causal_granularity="daily",
                base_dtype="continuous",
            )

    def test_random_effect_forbids_causal_granularity(self):
        """Random effect type must not have causal_granularity."""
        with pytest.raises(ValueError, match="Random effect .* must not have causal_granularity"):
            Dimension(
                name="intercept",
                description="Invalid",
                variable_type=VariableType.RANDOM_EFFECT,
                causal_granularity="daily",
                base_dtype="continuous",
            )


class TestCausalEdge:
    """Tests for CausalEdge."""

    def test_contemporaneous_edge(self):
        """Contemporaneous edge (lagged=False) is valid."""
        edge = CausalEdge(cause="stress", effect="mood", lagged=False)
        assert edge.lagged is False

    def test_lagged_edge(self):
        """Lagged edge (default) is valid."""
        edge = CausalEdge(cause="sleep", effect="mood")
        assert edge.lagged is True

    def test_edge_with_aggregation(self):
        """Edge with aggregation is valid."""
        edge = CausalEdge(cause="hourly_stress", effect="daily_mood", aggregation="mean")
        assert edge.aggregation == "mean"


class TestDSEMStructure:
    """Tests for DSEMStructure validation."""

    def _make_dims(self, *specs):
        """Helper to create dimensions from (name, granularity, variable_type) tuples."""
        return [
            Dimension(
                name=name,
                description=f"{name} description",
                variable_type=vtype,
                causal_granularity=gran,
                base_dtype="continuous",
            )
            for name, gran, vtype in specs
        ]

    def test_valid_simple_structure(self):
        """Simple valid structure passes validation."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("stress", "daily", VariableType.INPUT),
                ("mood", "daily", VariableType.OUTCOME),
            ),
            edges=[CausalEdge(cause="stress", effect="mood", lagged=False)],
        )
        assert len(structure.dimensions) == 2
        assert len(structure.edges) == 1
        assert structure.edges[0].lag_hours == 0  # contemporaneous

    def test_valid_lagged_edge(self):
        """Same-timescale lagged edge computes correct lag_hours."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("sleep", "daily", VariableType.OUTCOME),
                ("mood", "daily", VariableType.OUTCOME),
            ),
            edges=[CausalEdge(cause="sleep", effect="mood", lagged=True)],
        )
        assert structure.edges[0].lag_hours == 24  # 1 day

    def test_invalid_edge_cause_not_in_dimensions(self):
        """Edge cause must exist in dimensions."""
        with pytest.raises(ValueError, match="Edge cause 'unknown' not in dimensions"):
            DSEMStructure(
                dimensions=self._make_dims(("mood", "daily", VariableType.OUTCOME)),
                edges=[CausalEdge(cause="unknown", effect="mood")],
            )

    def test_invalid_edge_effect_not_in_dimensions(self):
        """Edge effect must exist in dimensions."""
        with pytest.raises(ValueError, match="Edge effect 'unknown' not in dimensions"):
            DSEMStructure(
                dimensions=self._make_dims(("stress", "daily", VariableType.INPUT)),
                edges=[CausalEdge(cause="stress", effect="unknown")],
            )

    def test_invalid_exogenous_cannot_be_effect(self):
        """Exogenous variable (input) cannot be an effect."""
        with pytest.raises(ValueError, match="Exogenous variable 'weather' cannot be an effect"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("mood", "daily", VariableType.OUTCOME),
                    ("weather", "daily", VariableType.INPUT),
                ),
                edges=[CausalEdge(cause="mood", effect="weather", lagged=False)],
            )

    def test_invalid_contemporaneous_cross_timescale(self):
        """Contemporaneous edge requires same timescale."""
        with pytest.raises(ValueError, match="Contemporaneous edge.*requires same timescale"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("hourly_stress", "hourly", VariableType.INPUT),
                    ("daily_mood", "daily", VariableType.OUTCOME),
                ),
                edges=[CausalEdge(cause="hourly_stress", effect="daily_mood", lagged=False)],
            )

    def test_valid_cross_scale_coarser_to_finer(self):
        """Coarser cause -> finer effect computes correct lag_hours."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("weekly_stress", "weekly", VariableType.INPUT),
                ("daily_mood", "daily", VariableType.OUTCOME),
            ),
            edges=[CausalEdge(cause="weekly_stress", effect="daily_mood")],  # lagged=True by default
        )
        assert structure.edges[0].lag_hours == 168  # 1 week (coarser granularity)

    def test_valid_cross_scale_finer_to_coarser(self):
        """Finer cause -> coarser effect computes correct lag_hours."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("hourly_activity", "hourly", VariableType.INPUT),
                ("daily_mood", "daily", VariableType.OUTCOME),
            ),
            edges=[CausalEdge(cause="hourly_activity", effect="daily_mood", aggregation="mean")],
        )
        assert structure.edges[0].lag_hours == 24  # 1 day (coarser granularity)

    def test_invalid_finer_to_coarser_needs_aggregation(self):
        """Finer cause -> coarser effect requires aggregation."""
        with pytest.raises(ValueError, match="Aggregation required for finer->coarser edge"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("hourly_activity", "hourly", VariableType.INPUT),
                    ("daily_mood", "daily", VariableType.OUTCOME),
                ),
                edges=[CausalEdge(cause="hourly_activity", effect="daily_mood")],
            )

    def test_valid_finer_to_coarser_with_aggregation(self):
        """Finer cause -> coarser effect with aggregation is valid."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("hourly_activity", "hourly", VariableType.INPUT),
                ("daily_mood", "daily", VariableType.OUTCOME),
            ),
            edges=[CausalEdge(cause="hourly_activity", effect="daily_mood", aggregation="mean")],
        )
        assert structure.edges[0].aggregation == "mean"
        assert structure.edges[0].lag_hours == 24

    def test_invalid_coarser_to_finer_no_aggregation_allowed(self):
        """Coarser cause -> finer effect must not have aggregation."""
        with pytest.raises(ValueError, match="Aggregation not allowed for coarser->finer edge"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("weekly_stress", "weekly", VariableType.INPUT),
                    ("daily_mood", "daily", VariableType.OUTCOME),
                ),
                edges=[CausalEdge(cause="weekly_stress", effect="daily_mood", aggregation="mean")],
            )

    def test_to_networkx(self):
        """Structure converts to NetworkX graph with computed lag_hours."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("stress", "daily", VariableType.INPUT),
                ("mood", "daily", VariableType.OUTCOME),
            ),
            edges=[CausalEdge(cause="stress", effect="mood", lagged=False)],
        )
        G = structure.to_networkx()
        assert "stress" in G.nodes
        assert "mood" in G.nodes
        assert ("stress", "mood") in G.edges
        assert G.edges["stress", "mood"]["lag_hours"] == 0


class TestGranularityHours:
    """Tests for granularity constants."""

    def test_granularity_values(self):
        """Granularity hours are correct."""
        assert GRANULARITY_HOURS["hourly"] == 1
        assert GRANULARITY_HOURS["daily"] == 24
        assert GRANULARITY_HOURS["weekly"] == 168
        assert GRANULARITY_HOURS["monthly"] == 720
        assert GRANULARITY_HOURS["yearly"] == 8760


class TestComputeLagHours:
    """Tests for compute_lag_hours function."""

    def test_same_timescale_contemporaneous(self):
        """Same timescale with lagged=False returns 0."""
        assert compute_lag_hours("daily", "daily", lagged=False) == 0
        assert compute_lag_hours("hourly", "hourly", lagged=False) == 0

    def test_same_timescale_lagged(self):
        """Same timescale with lagged=True returns 1 unit."""
        assert compute_lag_hours("daily", "daily", lagged=True) == 24
        assert compute_lag_hours("hourly", "hourly", lagged=True) == 1
        assert compute_lag_hours("weekly", "weekly", lagged=True) == 168

    def test_cross_timescale_coarser_to_finer(self):
        """Cross-timescale returns coarser granularity regardless of lagged flag."""
        assert compute_lag_hours("weekly", "daily", lagged=True) == 168
        assert compute_lag_hours("weekly", "daily", lagged=False) == 168  # cross-scale always uses max
        assert compute_lag_hours("daily", "hourly", lagged=True) == 24

    def test_cross_timescale_finer_to_coarser(self):
        """Finer to coarser also returns coarser granularity."""
        assert compute_lag_hours("hourly", "daily", lagged=True) == 24
        assert compute_lag_hours("daily", "weekly", lagged=True) == 168
