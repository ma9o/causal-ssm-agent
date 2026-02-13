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
    Role,
    TemporalStatus,
    compute_lag_hours,
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
            causal_granularity="daily",
        )
        assert c.role == Role.ENDOGENOUS
        assert c.temporal_status == TemporalStatus.TIME_VARYING
        assert c.causal_granularity == "daily"

    def test_exogenous_time_varying(self):
        """Exogenous, time-varying construct (classic input)."""
        c = Construct(
            name="weather",
            description="Daily temperature",
            role=Role.EXOGENOUS,
            temporal_status=TemporalStatus.TIME_VARYING,
            causal_granularity="daily",
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
        assert c.causal_granularity is None

    def test_time_varying_requires_causal_granularity(self):
        """Time-varying construct requires causal_granularity."""
        with pytest.raises(
            ValueError, match=r"Time-varying construct .* requires causal_granularity"
        ):
            Construct(
                name="mood",
                description="Invalid",
                role=Role.ENDOGENOUS,
                temporal_status=TemporalStatus.TIME_VARYING,
            )

    def test_time_invariant_forbids_causal_granularity(self):
        """Time-invariant construct must not have causal_granularity."""
        with pytest.raises(
            ValueError, match=r"Time-invariant construct .* must not have causal_granularity"
        ):
            Construct(
                name="age",
                description="Invalid",
                role=Role.EXOGENOUS,
                temporal_status=TemporalStatus.TIME_INVARIANT,
                causal_granularity="daily",
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
                causal_granularity="daily",
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
            construct="mood",
            how_to_measure="Extract mood ratings (1-10 scale)",
            measurement_granularity="daily",
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
                construct="mood",
                how_to_measure="Extract mood",
                measurement_granularity="daily",
                measurement_dtype="continuous",
                aggregation="invalid_agg",
            )

    def test_invalid_measurement_granularity(self):
        """Invalid measurement_granularity is rejected."""
        with pytest.raises(ValueError, match="Invalid measurement_granularity"):
            Indicator(
                name="mood_rating",
                construct="mood",
                how_to_measure="Extract mood",
                measurement_granularity="invalid_value",
                measurement_dtype="continuous",
                aggregation="mean",
            )

    def test_invalid_measurement_dtype(self):
        """Invalid measurement_dtype is rejected."""
        with pytest.raises(ValueError, match="Invalid measurement_dtype"):
            Indicator(
                name="mood_rating",
                construct="mood",
                how_to_measure="Extract mood",
                measurement_granularity="daily",
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
                    construct="mood",
                    how_to_measure="Extract mood ratings",
                    measurement_granularity="daily",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
                Indicator(
                    name="mood_text",
                    construct="mood",
                    how_to_measure="Extract mood from text",
                    measurement_granularity="daily",
                    measurement_dtype="ordinal",
                    aggregation="mean",
                ),
                Indicator(
                    name="stress_level",
                    construct="stress",
                    how_to_measure="Extract stress ratings",
                    measurement_granularity="daily",
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
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[],
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

    def test_invalid_measurement_granularity_coarser_than_causal(
        self, construct_factory, indicator_factory
    ):
        """Indicator measurement_granularity must be finer than construct's causal_granularity."""
        latent = LatentModel(
            constructs=[
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[],
        )
        measurement = MeasurementModel(
            indicators=[
                indicator_factory(
                    "mood_rating", "mood", granularity="weekly"
                ),  # coarser than daily
            ]
        )
        with pytest.raises(ValueError, match="coarser than construct"):
            CausalSpec(latent=latent, measurement=measurement)

    def test_to_networkx_includes_loading_edges(self, construct_factory, indicator_factory):
        """CausalSpec.to_networkx includes construct→indicator loading edges."""
        latent = LatentModel(
            constructs=[
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[],
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
