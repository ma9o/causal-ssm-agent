"""Tests for DSEM schema validation."""

import pytest

from causal_agent.orchestrator.schemas import (
    CausalEdge,
    Dimension,
    DSEMStructure,
    GRANULARITY_HOURS,
    Observability,
    Role,
    TemporalStatus,
    compute_lag_hours,
)


class TestDimension:
    """Tests for Dimension validation."""

    def test_endogenous_observed_time_varying(self):
        """Endogenous, observed, time-varying variable (classic outcome)."""
        dim = Dimension(
            name="mood",
            description="Daily mood rating",
            role=Role.ENDOGENOUS,
            observability=Observability.OBSERVED,
            how_to_measure="Extract mood ratings from survey responses",
            temporal_status=TemporalStatus.TIME_VARYING,
            causal_granularity="daily",
            measurement_dtype="continuous",
            aggregation="mean",
        )
        assert dim.role == Role.ENDOGENOUS
        assert dim.observability == Observability.OBSERVED
        assert dim.temporal_status == TemporalStatus.TIME_VARYING

    def test_exogenous_observed_time_varying(self):
        """Exogenous, observed, time-varying variable (classic input)."""
        dim = Dimension(
            name="weather",
            description="Daily temperature",
            role=Role.EXOGENOUS,
            observability=Observability.OBSERVED,
            how_to_measure="Get temperature from weather API",
            temporal_status=TemporalStatus.TIME_VARYING,
            causal_granularity="daily",
            measurement_dtype="continuous",
            aggregation="mean",
        )
        assert dim.role == Role.EXOGENOUS
        assert dim.observability == Observability.OBSERVED
        assert dim.temporal_status == TemporalStatus.TIME_VARYING

    def test_exogenous_observed_time_invariant(self):
        """Exogenous, observed, time-invariant variable (classic covariate)."""
        dim = Dimension(
            name="age",
            description="Participant age",
            role=Role.EXOGENOUS,
            observability=Observability.OBSERVED,
            how_to_measure="Get age from participant intake form",
            temporal_status=TemporalStatus.TIME_INVARIANT,
            measurement_dtype="continuous",
        )
        assert dim.role == Role.EXOGENOUS
        assert dim.observability == Observability.OBSERVED
        assert dim.temporal_status == TemporalStatus.TIME_INVARIANT
        assert dim.causal_granularity is None

    def test_exogenous_latent_time_invariant(self):
        """Exogenous, latent, time-invariant variable (classic random effect)."""
        dim = Dimension(
            name="person_intercept",
            description="Person-specific baseline",
            role=Role.EXOGENOUS,
            observability=Observability.LATENT,
            temporal_status=TemporalStatus.TIME_INVARIANT,
            measurement_dtype="continuous",
        )
        assert dim.role == Role.EXOGENOUS
        assert dim.observability == Observability.LATENT
        assert dim.temporal_status == TemporalStatus.TIME_INVARIANT
        assert dim.causal_granularity is None

    def test_time_varying_requires_causal_granularity(self):
        """Time-varying variable requires causal_granularity."""
        with pytest.raises(ValueError, match="Time-varying .* requires causal_granularity"):
            Dimension(
                name="mood",
                description="Invalid",
                role=Role.ENDOGENOUS,
                observability=Observability.OBSERVED,
                temporal_status=TemporalStatus.TIME_VARYING,
                measurement_dtype="continuous",
                aggregation="mean",
            )

    def test_time_varying_requires_aggregation(self):
        """Time-varying variable requires aggregation."""
        with pytest.raises(ValueError, match="Time-varying .* requires aggregation"):
            Dimension(
                name="mood",
                description="Invalid",
                role=Role.ENDOGENOUS,
                observability=Observability.OBSERVED,
                temporal_status=TemporalStatus.TIME_VARYING,
                causal_granularity="daily",
                measurement_dtype="continuous",
            )

    def test_time_invariant_forbids_aggregation(self):
        """Time-invariant variable must not have aggregation."""
        with pytest.raises(ValueError, match="Time-invariant .* must not have aggregation"):
            Dimension(
                name="age",
                description="Invalid",
                role=Role.EXOGENOUS,
                observability=Observability.OBSERVED,
                temporal_status=TemporalStatus.TIME_INVARIANT,
                measurement_dtype="continuous",
                aggregation="mean",
            )

    def test_time_invariant_forbids_causal_granularity(self):
        """Time-invariant variable must not have causal_granularity."""
        with pytest.raises(ValueError, match="Time-invariant .* must not have causal_granularity"):
            Dimension(
                name="age",
                description="Invalid",
                role=Role.EXOGENOUS,
                observability=Observability.OBSERVED,
                temporal_status=TemporalStatus.TIME_INVARIANT,
                causal_granularity="daily",
                measurement_dtype="continuous",
            )


class TestCausalEdge:
    """Tests for CausalEdge."""

    def test_contemporaneous_edge(self):
        """Contemporaneous edge (lagged=False) is valid."""
        edge = CausalEdge(cause="stress", effect="mood", description="Stress affects mood", lagged=False)
        assert edge.lagged is False

    def test_lagged_edge(self):
        """Lagged edge (default) is valid."""
        edge = CausalEdge(cause="sleep", effect="mood", description="Sleep quality affects next day mood")
        assert edge.lagged is True



class TestDSEMStructure:
    """Tests for DSEMStructure validation."""

    def _make_dim(self, name, granularity, role, observability=Observability.OBSERVED, temporal_status=None, is_outcome=False):
        """Helper to create a dimension.

        If temporal_status is None, infers from granularity presence.
        """
        if temporal_status is None:
            temporal_status = TemporalStatus.TIME_VARYING if granularity else TemporalStatus.TIME_INVARIANT
        agg = "mean" if temporal_status == TemporalStatus.TIME_VARYING else None
        # Only observed variables need how_to_measure
        how_to_measure = f"Extract {name} from data" if observability == Observability.OBSERVED else None
        return Dimension(
            name=name,
            description=f"{name} description",
            role=role,
            is_outcome=is_outcome,
            observability=observability,
            how_to_measure=how_to_measure,
            temporal_status=temporal_status,
            causal_granularity=granularity,
            measurement_dtype="continuous",
            aggregation=agg,
        )

    def test_valid_simple_structure(self):
        """Simple valid structure passes validation."""
        structure = DSEMStructure(
            dimensions=[
                self._make_dim("stress", "daily", Role.EXOGENOUS),
                self._make_dim("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Stress affects mood", lagged=False)],
        )
        assert len(structure.dimensions) == 2
        assert len(structure.edges) == 1
        assert structure.edges[0].lag_hours == 0  # contemporaneous

    def test_valid_lagged_edge(self):
        """Same-timescale lagged edge computes correct lag_hours."""
        structure = DSEMStructure(
            dimensions=[
                self._make_dim("sleep", "daily", Role.ENDOGENOUS),
                self._make_dim("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="sleep", effect="mood", description="Sleep affects mood", lagged=True)],
        )
        assert structure.edges[0].lag_hours == 24  # 1 day

    def test_invalid_edge_cause_not_in_dimensions(self):
        """Edge cause must exist in dimensions."""
        with pytest.raises(ValueError, match="Edge cause 'unknown' not in dimensions"):
            DSEMStructure(
                dimensions=[self._make_dim("mood", "daily", Role.ENDOGENOUS, is_outcome=True)],
                edges=[CausalEdge(cause="unknown", effect="mood", description="Test edge")],
            )

    def test_invalid_edge_effect_not_in_dimensions(self):
        """Edge effect must exist in dimensions."""
        with pytest.raises(ValueError, match="Edge effect 'unknown' not in dimensions"):
            DSEMStructure(
                dimensions=[
                    self._make_dim("stress", "daily", Role.EXOGENOUS),
                    self._make_dim("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="stress", effect="unknown", description="Test edge")],
            )

    def test_invalid_exogenous_cannot_be_effect(self):
        """Exogenous variable cannot be an effect."""
        with pytest.raises(ValueError, match="Exogenous variable 'weather' cannot be an effect"):
            DSEMStructure(
                dimensions=[
                    self._make_dim("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                    self._make_dim("weather", "daily", Role.EXOGENOUS),
                ],
                edges=[CausalEdge(cause="mood", effect="weather", description="Invalid edge", lagged=False)],
            )

    def test_invalid_contemporaneous_cross_timescale(self):
        """Contemporaneous edge requires same timescale."""
        with pytest.raises(ValueError, match="Contemporaneous edge.*requires same timescale"):
            DSEMStructure(
                dimensions=[
                    self._make_dim("hourly_stress", "hourly", Role.EXOGENOUS),
                    self._make_dim("daily_mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="hourly_stress", effect="daily_mood", description="Invalid cross-timescale", lagged=False)],
            )

    def test_valid_cross_scale_coarser_to_finer(self):
        """Coarser cause -> finer effect computes correct lag_hours."""
        structure = DSEMStructure(
            dimensions=[
                self._make_dim("weekly_stress", "weekly", Role.EXOGENOUS),
                self._make_dim("daily_mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="weekly_stress", effect="daily_mood", description="Weekly stress affects daily mood")],  # lagged=True by default
        )
        assert structure.edges[0].lag_hours == 168  # 1 week (coarser granularity)

    def test_valid_cross_scale_finer_to_coarser(self):
        """Finer cause -> coarser effect computes correct lag_hours."""
        structure = DSEMStructure(
            dimensions=[
                self._make_dim("hourly_activity", "hourly", Role.EXOGENOUS),
                self._make_dim("daily_mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="hourly_activity", effect="daily_mood", description="Activity affects mood")],
        )
        assert structure.edges[0].lag_hours == 24  # 1 day (coarser granularity)

    def test_to_networkx(self):
        """Structure converts to NetworkX graph with computed lag_hours."""
        structure = DSEMStructure(
            dimensions=[
                self._make_dim("stress", "daily", Role.EXOGENOUS),
                self._make_dim("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Stress affects mood", lagged=False)],
        )
        G = structure.to_networkx()
        assert "stress" in G.nodes
        assert "mood" in G.nodes
        assert ("stress", "mood") in G.edges
        assert G.edges["stress", "mood"]["lag_hours"] == 0

    def test_invalid_no_outcome(self):
        """Structure must have exactly one outcome."""
        with pytest.raises(ValueError, match="Exactly one dimension must have is_outcome=true"):
            DSEMStructure(
                dimensions=[
                    self._make_dim("stress", "daily", Role.EXOGENOUS),
                    self._make_dim("mood", "daily", Role.ENDOGENOUS),
                ],
                edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
            )

    def test_invalid_multiple_outcomes(self):
        """Structure must have exactly one outcome."""
        with pytest.raises(ValueError, match="Only one outcome allowed"):
            DSEMStructure(
                dimensions=[
                    self._make_dim("stress", "daily", Role.ENDOGENOUS, is_outcome=True),
                    self._make_dim("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
            )

    def test_invalid_exogenous_outcome(self):
        """Outcome must be endogenous."""
        with pytest.raises(ValueError, match="Outcome variable .* must be endogenous"):
            Dimension(
                name="weather",
                description="Weather",
                role=Role.EXOGENOUS,
                is_outcome=True,
                observability=Observability.OBSERVED,
                how_to_measure="Get weather from API",
                temporal_status=TemporalStatus.TIME_VARYING,
                causal_granularity="daily",
                measurement_dtype="continuous",
                aggregation="mean",
            )


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
