"""Tests for causal spec schema computed properties and utility functions.

Object-level construction validation (Construct, LatentModel, Indicator,
MeasurementModel) is covered by test_schema_validators.py via dict validators.
This file tests CausalSpec composition, computed properties, and utility
functions that are not exercised through dict validation.
"""

from causal_ssm_agent.orchestrator.schemas import (
    CausalEdge,
    CausalSpec,
    LatentModel,
    MeasurementModel,
    ObservationKind,
    Role,
    check_semantic_collisions,
    compute_lag_hours,
    derive_observation_kind,
)


class TestCausalSpec:
    """Tests for CausalSpec validation."""

    def test_latent_construct_without_indicator_is_valid(
        self, construct_factory, indicator_factory
    ):
        """Latent constructs without indicators are allowed (A2 deferred to y0)."""
        latent = LatentModel(
            constructs=[
                construct_factory("stress", "daily", Role.EXOGENOUS),
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
        )
        measurement = MeasurementModel(
            indicators=[
                indicator_factory("mood_rating", "mood"),
                # stress has no indicator - it's a latent construct
            ]
        )
        # This should now be valid - y0 will check identification in Stage 3
        causal_spec = CausalSpec(latent=latent, measurement=measurement)
        assert len(causal_spec.latent.constructs) == 2
        assert len(causal_spec.measurement.indicators) == 1

    def test_to_networkx_includes_loading_edges(self, construct_factory, indicator_factory):
        """CausalSpec.to_networkx includes construct→indicator loading edges."""
        latent = LatentModel(
            constructs=[
                construct_factory("stress", "daily", Role.EXOGENOUS),
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
        )
        measurement = MeasurementModel(
            indicators=[
                indicator_factory("mood_rating", "mood"),
            ]
        )
        causal_spec = CausalSpec(latent=latent, measurement=measurement)
        G = causal_spec.to_networkx()

        # Both construct and indicator nodes exist
        assert "mood" in G.nodes
        assert "mood_rating" in G.nodes

        # Loading edge exists
        assert ("mood", "mood_rating") in G.edges
        assert G.edges["mood", "mood_rating"]["edge_type"] == "loading"

    def test_get_edge_lag_hours(self, construct_factory, indicator_factory):
        """CausalSpec.get_edge_lag_hours computes correct lag."""
        latent = LatentModel(
            constructs=[
                construct_factory("sleep", "daily", Role.EXOGENOUS),
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[
                CausalEdge(
                    cause="sleep", effect="mood", description="Sleep affects mood", lagged=True
                )
            ],
        )
        measurement = MeasurementModel(
            indicators=[
                indicator_factory("sleep_hours", "sleep"),
                indicator_factory("mood_rating", "mood"),
            ]
        )
        causal_spec = CausalSpec(latent=latent, measurement=measurement)
        lag = causal_spec.get_edge_lag_hours(latent.edges[0])
        assert lag == 24  # 1 day


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
        assert (
            compute_lag_hours("weekly", "daily", lagged=False) == 168
        )  # cross-scale always uses max
        assert compute_lag_hours("daily", "hourly", lagged=True) == 24

    def test_cross_timescale_finer_to_coarser(self):
        """Finer to coarser also returns coarser granularity."""
        assert compute_lag_hours("hourly", "daily", lagged=True) == 24
        assert compute_lag_hours("daily", "weekly", lagged=True) == 168

    def test_time_invariant_both_none(self):
        """Both granularities None (time-invariant constructs) returns 0."""
        assert compute_lag_hours(None, None, lagged=False) == 0
        assert compute_lag_hours(None, None, lagged=True) == 0

    def test_time_invariant_mixed(self):
        """One granularity None, other non-None uses cross-timescale logic."""
        assert compute_lag_hours("daily", None, lagged=True) == 24
        assert compute_lag_hours(None, "daily", lagged=True) == 24
        assert compute_lag_hours(None, "hourly", lagged=False) == 1


class TestDeriveObservationKind:
    """Tests for derive_observation_kind function."""

    def test_cumulative(self):
        """Sum aggregation → cumulative."""
        assert derive_observation_kind("sum") == ObservationKind.CUMULATIVE

    def test_point_in_time(self):
        """First/last/min/max → point_in_time."""
        assert derive_observation_kind("first") == ObservationKind.POINT_IN_TIME
        assert derive_observation_kind("last") == ObservationKind.POINT_IN_TIME
        assert derive_observation_kind("min") == ObservationKind.POINT_IN_TIME
        assert derive_observation_kind("max") == ObservationKind.POINT_IN_TIME

    def test_variability(self):
        """Variability aggregations classified correctly."""
        for agg in ("std", "var", "range", "cv", "iqr", "instability"):
            assert derive_observation_kind(agg) == ObservationKind.VARIABILITY

    def test_frequency(self):
        """Count/n_unique → frequency."""
        assert derive_observation_kind("count") == ObservationKind.FREQUENCY
        assert derive_observation_kind("n_unique") == ObservationKind.FREQUENCY

    def test_window_average_default(self):
        """Mean, median, percentiles → window_average."""
        for agg in ("mean", "median", "p10", "p75", "entropy", "trend"):
            assert derive_observation_kind(agg) == ObservationKind.WINDOW_AVERAGE

    def test_ordinal_overrides_aggregation(self):
        """Ordinal dtype → ordinal, regardless of aggregation."""
        assert derive_observation_kind("mean", "ordinal") == ObservationKind.ORDINAL
        assert derive_observation_kind("last", "ordinal") == ObservationKind.ORDINAL
        assert derive_observation_kind("median", "ordinal") == ObservationKind.ORDINAL


class TestSemanticCollisions:
    """Tests for check_semantic_collisions function."""

    def test_count_text_mean_agg_collision(self):
        """'count' in how_to_measure + mean aggregation → warning."""
        warnings = check_semantic_collisions("Count the number of exercise sessions", "mean")
        assert len(warnings) >= 1
        assert "counting" in warnings[0].lower() or "count" in warnings[0].lower()

    def test_no_collision(self):
        """Consistent text and aggregation → no warnings."""
        warnings = check_semantic_collisions("Average daily mood rating", "mean")
        assert len(warnings) == 0

    def test_total_text_mean_agg_collision(self):
        """'total' in text + mean aggregation → warning."""
        warnings = check_semantic_collisions("Total steps walked during the day", "mean")
        assert len(warnings) >= 1

    def test_last_text_sum_agg_collision(self):
        """'most recent' in text + sum aggregation → warning."""
        warnings = check_semantic_collisions("The most recent blood pressure reading", "sum")
        assert len(warnings) >= 1


class TestIndicatorObservationKind:
    """Tests for Indicator.observation_kind property."""

    def test_observation_kind_property(self, indicator_factory):
        """Indicator.observation_kind returns derived kind."""
        ind = indicator_factory("steps", "activity", aggregation="sum", dtype="count")
        assert ind.observation_kind == ObservationKind.CUMULATIVE

    def test_requires_integral_measurement(self, indicator_factory):
        """Cumulative indicators require integral measurement equation."""
        cumulative = indicator_factory("steps", "activity", aggregation="sum", dtype="count")
        assert cumulative.requires_integral_measurement is True

        average = indicator_factory("mood_rating", "mood", aggregation="mean", dtype="continuous")
        assert average.requires_integral_measurement is False

        point = indicator_factory("last_bp", "bp", aggregation="last", dtype="continuous")
        assert point.requires_integral_measurement is False

    def test_ordinal_indicator(self, indicator_factory):
        """Ordinal dtype → ordinal observation kind."""
        ind = indicator_factory("pain_level", "pain", aggregation="median", dtype="ordinal")
        assert ind.observation_kind == ObservationKind.ORDINAL
        assert ind.requires_integral_measurement is False

    def test_min_max_are_point_in_time(self, indicator_factory):
        """Min/max are instantaneous extremals, not window averages."""
        ind_min = indicator_factory("min_hr", "hr", aggregation="min", dtype="continuous")
        ind_max = indicator_factory("max_hr", "hr", aggregation="max", dtype="continuous")
        assert ind_min.observation_kind == ObservationKind.POINT_IN_TIME
        assert ind_max.observation_kind == ObservationKind.POINT_IN_TIME
