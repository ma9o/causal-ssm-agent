"""Tests for causal spec schema validation.

Two-stage schema:
- LatentModel: theoretical constructs + causal edges
- MeasurementModel: indicators that operationalize constructs
- CausalSpec: composition of latent + measurement
"""

import pytest

from causal_ssm_agent.orchestrator.schemas import (
    GRANULARITY_HOURS,
    CausalEdge,
    CausalSpec,
    Construct,
    Indicator,
    LatentModel,
    MeasurementModel,
    ObservationKind,
    Role,
    TemporalStatus,
    check_semantic_collisions,
    compute_lag_hours,
    derive_observation_kind,
)


class TestConstruct:
    """Tests for Construct validation."""

    def test_endogenous_time_varying(self):
        """Endogenous, time-varying construct (classic outcome)."""
        c = Construct(
            name="mood",
            description="Daily mood state",
            role=Role.ENDOGENOUS,
            temporal_status=TemporalStatus.TIME_VARYING,
            temporal_scale="daily",
        )
        assert c.role == Role.ENDOGENOUS
        assert c.temporal_status == TemporalStatus.TIME_VARYING
        assert c.temporal_scale == "daily"

    def test_exogenous_time_varying(self):
        """Exogenous, time-varying construct (classic input)."""
        c = Construct(
            name="weather",
            description="Daily temperature",
            role=Role.EXOGENOUS,
            temporal_status=TemporalStatus.TIME_VARYING,
            temporal_scale="daily",
        )
        assert c.role == Role.EXOGENOUS
        assert c.temporal_status == TemporalStatus.TIME_VARYING

    def test_exogenous_time_invariant(self):
        """Exogenous, time-invariant construct (classic covariate)."""
        c = Construct(
            name="age",
            description="Participant age",
            role=Role.EXOGENOUS,
            temporal_status=TemporalStatus.TIME_INVARIANT,
        )
        assert c.role == Role.EXOGENOUS
        assert c.temporal_status == TemporalStatus.TIME_INVARIANT
        assert c.temporal_scale is None

    def test_time_varying_requires_temporal_scale(self):
        """Time-varying construct requires temporal_scale."""
        with pytest.raises(ValueError, match=r"Time-varying construct .* requires temporal_scale"):
            Construct(
                name="mood",
                description="Invalid",
                role=Role.ENDOGENOUS,
                temporal_status=TemporalStatus.TIME_VARYING,
            )

    def test_time_invariant_forbids_temporal_scale(self):
        """Time-invariant construct must not have temporal_scale."""
        with pytest.raises(
            ValueError, match=r"Time-invariant construct .* must not have temporal_scale"
        ):
            Construct(
                name="age",
                description="Invalid",
                role=Role.EXOGENOUS,
                temporal_status=TemporalStatus.TIME_INVARIANT,
                temporal_scale="daily",
            )

    def test_exogenous_cannot_be_outcome(self):
        """Exogenous construct cannot be outcome."""
        with pytest.raises(ValueError, match=r"Outcome construct .* must be endogenous"):
            Construct(
                name="weather",
                description="Invalid",
                role=Role.EXOGENOUS,
                is_outcome=True,
                temporal_status=TemporalStatus.TIME_VARYING,
                temporal_scale="daily",
            )


class TestCausalEdge:
    """Tests for CausalEdge."""

    def test_contemporaneous_edge(self):
        """Contemporaneous edge (lagged=False) is valid."""
        edge = CausalEdge(
            cause="stress", effect="mood", description="Stress affects mood", lagged=False
        )
        assert edge.lagged is False

    def test_lagged_edge(self):
        """Lagged edge (default) is valid."""
        edge = CausalEdge(
            cause="sleep", effect="mood", description="Sleep quality affects next day mood"
        )
        assert edge.lagged is True


class TestLatentModel:
    """Tests for LatentModel validation."""

    def test_valid_simple_structure(self, construct_factory):
        """Simple valid structure passes validation."""
        structure = LatentModel(
            constructs=[
                construct_factory("stress", "daily", Role.EXOGENOUS),
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[
                CausalEdge(
                    cause="stress", effect="mood", description="Stress affects mood", lagged=False
                )
            ],
        )
        assert len(structure.constructs) == 2
        assert len(structure.edges) == 1

    def test_invalid_edge_cause_not_in_constructs(self, construct_factory):
        """Edge cause must exist in constructs."""
        with pytest.raises(ValueError, match="Edge cause 'unknown' not in constructs"):
            LatentModel(
                constructs=[construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True)],
                edges=[CausalEdge(cause="unknown", effect="mood", description="Test edge")],
            )

    def test_invalid_edge_effect_not_in_constructs(self, construct_factory):
        """Edge effect must exist in constructs."""
        with pytest.raises(ValueError, match="Edge effect 'unknown' not in constructs"):
            LatentModel(
                constructs=[
                    construct_factory("stress", "daily", Role.EXOGENOUS),
                    construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="stress", effect="unknown", description="Test edge")],
            )

    def test_invalid_exogenous_cannot_be_effect(self, construct_factory):
        """Exogenous construct cannot be an effect."""
        with pytest.raises(ValueError, match="Exogenous construct 'weather' cannot be an effect"):
            LatentModel(
                constructs=[
                    construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                    construct_factory("weather", "daily", Role.EXOGENOUS),
                ],
                edges=[
                    CausalEdge(
                        cause="mood", effect="weather", description="Invalid edge", lagged=False
                    )
                ],
            )

    def test_invalid_contemporaneous_cross_timescale(self, construct_factory):
        """Contemporaneous edge requires same timescale."""
        with pytest.raises(ValueError, match=r"Contemporaneous edge.*requires same timescale"):
            LatentModel(
                constructs=[
                    construct_factory("hourly_stress", "hourly", Role.EXOGENOUS),
                    construct_factory("daily_mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[
                    CausalEdge(
                        cause="hourly_stress",
                        effect="daily_mood",
                        description="Invalid",
                        lagged=False,
                    )
                ],
            )

    def test_invalid_outcome_no_incoming_edges(self, construct_factory):
        """Outcome must have at least one incoming causal edge."""
        with pytest.raises(ValueError, match="has no incoming causal edges"):
            LatentModel(
                constructs=[
                    construct_factory("stress", "daily", Role.EXOGENOUS),
                    construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[],
            )

    def test_invalid_no_outcome(self, construct_factory):
        """Structure must have exactly one outcome."""
        with pytest.raises(ValueError, match="Exactly one construct must have is_outcome=true"):
            LatentModel(
                constructs=[
                    construct_factory("stress", "daily", Role.EXOGENOUS),
                    construct_factory("mood", "daily", Role.ENDOGENOUS),
                ],
                edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
            )

    def test_invalid_multiple_outcomes(self, construct_factory):
        """Structure must have exactly one outcome."""
        with pytest.raises(ValueError, match="Only one outcome allowed"):
            LatentModel(
                constructs=[
                    construct_factory("stress", "daily", Role.ENDOGENOUS, is_outcome=True),
                    construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
            )

    def test_to_networkx(self, construct_factory):
        """Structure converts to NetworkX graph."""
        structure = LatentModel(
            constructs=[
                construct_factory("stress", "daily", Role.EXOGENOUS),
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[
                CausalEdge(
                    cause="stress", effect="mood", description="Stress affects mood", lagged=False
                )
            ],
        )
        G = structure.to_networkx()
        assert "stress" in G.nodes
        assert "mood" in G.nodes
        assert ("stress", "mood") in G.edges


class TestIndicator:
    """Tests for Indicator validation."""

    def test_valid_indicator(self):
        """Valid indicator passes validation."""
        ind = Indicator(
            name="mood_rating",
            construct_name="mood",
            how_to_measure="Extract mood ratings (1-10 scale)",
            measurement_dtype="continuous",
            aggregation="mean",
        )
        assert ind.name == "mood_rating"
        assert ind.construct_name == "mood"

    def test_invalid_aggregation(self):
        """Invalid aggregation is rejected."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            Indicator(
                name="mood_rating",
                construct_name="mood",
                how_to_measure="Extract mood",
                measurement_dtype="continuous",
                aggregation="invalid_agg",
            )

    def test_invalid_measurement_dtype(self):
        """Invalid measurement_dtype is rejected."""
        with pytest.raises(ValueError, match="Invalid measurement_dtype"):
            Indicator(
                name="mood_rating",
                construct_name="mood",
                how_to_measure="Extract mood",
                measurement_dtype="invalid_type",
                aggregation="mean",
            )


class TestMeasurementModel:
    """Tests for MeasurementModel."""

    def test_get_indicators_for_construct(self):
        """get_indicators_for_construct returns correct indicators."""
        model = MeasurementModel(
            indicators=[
                Indicator(
                    name="mood_rating",
                    construct_name="mood",
                    how_to_measure="Extract mood ratings",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
                Indicator(
                    name="mood_text",
                    construct_name="mood",
                    how_to_measure="Extract mood from text",
                    measurement_dtype="ordinal",
                    aggregation="mean",
                ),
                Indicator(
                    name="stress_level",
                    construct_name="stress",
                    how_to_measure="Extract stress ratings",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ]
        )
        mood_indicators = model.get_indicators_for_construct("mood")
        assert len(mood_indicators) == 2
        assert all(i.construct_name == "mood" for i in mood_indicators)


class TestCausalSpec:
    """Tests for CausalSpec validation."""

    def test_valid_causal_spec(self, construct_factory, indicator_factory):
        """Valid CausalSpec passes validation."""
        latent = LatentModel(
            constructs=[
                construct_factory("stress", "daily", Role.EXOGENOUS),
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Stress affects mood")],
        )
        measurement = MeasurementModel(
            indicators=[
                indicator_factory("stress_rating", "stress"),
                indicator_factory("mood_rating", "mood"),
            ]
        )
        causal_spec = CausalSpec(latent=latent, measurement=measurement)
        assert len(causal_spec.latent.constructs) == 2
        assert len(causal_spec.measurement.indicators) == 2

    def test_invalid_indicator_references_unknown_construct(
        self, construct_factory, indicator_factory
    ):
        """Indicator must reference a valid construct."""
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
                indicator_factory("unknown_indicator", "unknown"),  # invalid reference
            ]
        )
        with pytest.raises(ValueError, match="references unknown construct 'unknown'"):
            CausalSpec(latent=latent, measurement=measurement)

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
        assert (
            compute_lag_hours("weekly", "daily", lagged=False) == 168
        )  # cross-scale always uses max
        assert compute_lag_hours("daily", "hourly", lagged=True) == 24

    def test_cross_timescale_finer_to_coarser(self):
        """Finer to coarser also returns coarser granularity."""
        assert compute_lag_hours("hourly", "daily", lagged=True) == 24
        assert compute_lag_hours("daily", "weekly", lagged=True) == 168


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
