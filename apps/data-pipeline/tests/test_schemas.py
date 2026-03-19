"""Tests for causal spec schema computed properties and utility functions.

Object-level construction validation (Construct, LatentModel, Indicator,
MeasurementModel) is covered by test_schema_validators.py via dict validators.
This file tests CausalSpec composition, computed properties, and utility
functions that are not exercised through dict validation.
"""

import pytest

from causal_ssm_agent.orchestrator.schemas import (
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
    derive_observation_kind,
    parse_duration_to_hours,
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
        )
        assert c.role == Role.ENDOGENOUS
        assert c.temporal_status == TemporalStatus.TIME_VARYING

    def test_exogenous_time_varying(self):
        """Exogenous, time-varying construct (classic input)."""
        c = Construct(
            name="weather",
            description="Daily temperature",
            role=Role.EXOGENOUS,
            temporal_status=TemporalStatus.TIME_VARYING,
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

    def test_exogenous_cannot_be_outcome(self):
        """Exogenous construct cannot be outcome."""
        with pytest.raises(ValueError, match=r"Outcome construct .* must be endogenous"):
            Construct(
                name="weather",
                description="Invalid",
                role=Role.EXOGENOUS,
                is_outcome=True,
                temporal_status=TemporalStatus.TIME_VARYING,
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
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
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
                constructs=[construct_factory("mood", Role.ENDOGENOUS, is_outcome=True)],
                edges=[CausalEdge(cause="unknown", effect="mood", description="Test edge")],
            )

    def test_invalid_edge_effect_not_in_constructs(self, construct_factory):
        """Edge effect must exist in constructs."""
        with pytest.raises(ValueError, match="Edge effect 'unknown' not in constructs"):
            LatentModel(
                constructs=[
                    construct_factory("stress", Role.EXOGENOUS),
                    construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="stress", effect="unknown", description="Test edge")],
            )

    def test_invalid_exogenous_cannot_be_effect(self, construct_factory):
        """Exogenous construct cannot be an effect."""
        with pytest.raises(ValueError, match="Exogenous construct 'weather' cannot be an effect"):
            LatentModel(
                constructs=[
                    construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
                    construct_factory("weather", Role.EXOGENOUS),
                ],
                edges=[
                    CausalEdge(
                        cause="mood", effect="weather", description="Invalid edge", lagged=False
                    )
                ],
            )

    def test_valid_exogenous_to_endogenous_contemporaneous(self, construct_factory):
        """Exogenous → endogenous contemporaneous edge is valid."""
        model = LatentModel(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[
                CausalEdge(
                    cause="stress",
                    effect="mood",
                    description="Contemporaneous exo→endo",
                    lagged=False,
                )
            ],
        )
        assert len(model.edges) == 1
        assert model.edges[0].lagged is False

    def test_invalid_time_varying_to_time_invariant_edge(self, construct_factory):
        """Time-varying constructs cannot cause time-invariant constructs."""
        with pytest.raises(ValueError, match="cannot be a cause of time-invariant construct"):
            LatentModel(
                constructs=[
                    construct_factory("habit", Role.ENDOGENOUS),
                    construct_factory(
                        "trait",
                        Role.ENDOGENOUS,
                        is_outcome=True,
                        temporal_status=TemporalStatus.TIME_INVARIANT,
                    ),
                ],
                edges=[
                    CausalEdge(
                        cause="habit",
                        effect="trait",
                        description="Habit changes a fixed trait",
                    )
                ],
            )

    def test_invalid_outcome_no_incoming_edges(self, construct_factory):
        """Outcome must have at least one incoming causal edge."""
        with pytest.raises(ValueError, match="has no incoming causal edges"):
            LatentModel(
                constructs=[
                    construct_factory("stress", Role.EXOGENOUS),
                    construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[],
            )

    def test_invalid_no_outcome(self, construct_factory):
        """Structure must have exactly one outcome."""
        with pytest.raises(ValueError, match="Exactly one construct must have is_outcome=true"):
            LatentModel(
                constructs=[
                    construct_factory("stress", Role.EXOGENOUS),
                    construct_factory("mood", Role.ENDOGENOUS),
                ],
                edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
            )

    def test_invalid_multiple_outcomes(self, construct_factory):
        """Structure must have exactly one outcome."""
        with pytest.raises(ValueError, match="Only one outcome allowed"):
            LatentModel(
                constructs=[
                    construct_factory("stress", Role.ENDOGENOUS, is_outcome=True),
                    construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
            )

    def test_build_digraph(self, construct_factory):
        """Latent model dict converts to NetworkX graph via build_digraph."""
        from causal_ssm_agent.utils.causal_spec import build_digraph

        structure = LatentModel(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[
                CausalEdge(
                    cause="stress", effect="mood", description="Stress affects mood", lagged=False
                )
            ],
        )
        G = build_digraph(structure.model_dump())
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

    def test_ordinal_requires_levels(self):
        """Ordinal dtype without ordinal_levels is rejected."""
        with pytest.raises(ValueError, match="ordinal_levels is required"):
            Indicator(
                name="pain",
                construct_name="pain",
                how_to_measure="Extract pain level",
                measurement_dtype="ordinal",
                aggregation="median",
            )

    def test_ordinal_needs_at_least_two_levels(self):
        """Ordinal with only one level is rejected."""
        with pytest.raises(ValueError, match="at least 2 items"):
            Indicator(
                name="pain",
                construct_name="pain",
                how_to_measure="Extract pain level",
                measurement_dtype="ordinal",
                aggregation="median",
                ordinal_levels=["only_one"],
            )

    def test_ordinal_no_duplicate_levels(self):
        """Ordinal with duplicate levels is rejected."""
        with pytest.raises(ValueError, match="duplicate labels"):
            Indicator(
                name="pain",
                construct_name="pain",
                how_to_measure="Extract pain level",
                measurement_dtype="ordinal",
                aggregation="median",
                ordinal_levels=["low", "low", "high"],
            )

    def test_ordinal_valid_levels(self):
        """Ordinal with valid levels passes."""
        ind = Indicator(
            name="pain",
            construct_name="pain",
            how_to_measure="Extract pain level",
            measurement_dtype="ordinal",
            aggregation="median",
            ordinal_levels=["low", "medium", "high"],
        )
        assert ind.ordinal_levels == ["low", "medium", "high"]

    def test_non_ordinal_ignores_levels(self):
        """Non-ordinal dtype doesn't require ordinal_levels."""
        ind = Indicator(
            name="weight",
            construct_name="health",
            how_to_measure="Extract weight",
            measurement_dtype="continuous",
            aggregation="mean",
        )
        assert ind.ordinal_levels is None

    def test_semantic_default(self):
        """Extraction mode defaults to 'semantic'."""
        ind = Indicator(
            name="mood_rating",
            construct_name="mood",
            how_to_measure="Extract mood",
            measurement_dtype="continuous",
            aggregation="mean",
        )
        assert ind.extraction_mode == "semantic"

    def test_invalid_extraction_mode(self):
        """Invalid extraction_mode is rejected."""
        with pytest.raises(ValueError, match="extraction_mode must be"):
            Indicator(
                name="mood_rating",
                construct_name="mood",
                how_to_measure="Extract mood",
                measurement_dtype="continuous",
                aggregation="mean",
                extraction_mode="invalid",
            )

    def test_computed_valid(self):
        """Computed indicator with single source column and continuous dtype passes."""
        ind = Indicator(
            name="avg_heart_rate",
            construct_name="health",
            how_to_measure="Use heart_rate column directly",
            measurement_dtype="continuous",
            aggregation="mean",
            source_columns=["heart_rate"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_count_dtype(self):
        """Computed indicator with count dtype passes."""
        ind = Indicator(
            name="total_steps",
            construct_name="activity",
            how_to_measure="Use steps column directly",
            measurement_dtype="count",
            aggregation="sum",
            source_columns=["steps"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_requires_single_source_column(self):
        """Computed with 0 or 2+ source_columns is rejected."""
        with pytest.raises(ValueError, match="exactly 1 source_column"):
            Indicator(
                name="avg_hr",
                construct_name="health",
                how_to_measure="Use heart_rate",
                measurement_dtype="continuous",
                aggregation="mean",
                source_columns=[],
                extraction_mode="computed",
            )
        with pytest.raises(ValueError, match="exactly 1 source_column"):
            Indicator(
                name="avg_hr",
                construct_name="health",
                how_to_measure="Compute from systolic and diastolic",
                measurement_dtype="continuous",
                aggregation="mean",
                source_columns=["systolic_bp", "diastolic_bp"],
                extraction_mode="computed",
            )

    def test_computed_rejects_non_numeric_dtype(self):
        """Computed with binary/ordinal/categorical dtype is rejected."""
        for dtype in ("binary", "ordinal", "categorical"):
            kwargs = {
                "name": "test_ind",
                "construct_name": "test",
                "how_to_measure": "Extract test",
                "measurement_dtype": dtype,
                "aggregation": "mean",
                "source_columns": ["col"],
                "extraction_mode": "computed",
            }
            if dtype == "ordinal":
                kwargs["ordinal_levels"] = ["low", "medium", "high"]
            with pytest.raises(ValueError, match="'continuous' or 'count'"):
                Indicator(**kwargs)


class TestMeasurementModel:
    """Tests for MeasurementModel."""

    def test_get_indicators_for_construct(self):
        """get_indicators_for_construct returns correct indicators."""
        model = MeasurementModel(
            model_clock="1d",
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
                    ordinal_levels=["low", "medium", "high"],
                ),
                Indicator(
                    name="stress_level",
                    construct_name="stress",
                    how_to_measure="Extract stress ratings",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ],
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
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Stress affects mood")],
        )
        measurement = MeasurementModel(
            model_clock="1d",
            indicators=[
                indicator_factory("stress_rating", "stress"),
                indicator_factory("mood_rating", "mood"),
            ],
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
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
        )
        measurement = MeasurementModel(
            model_clock="1d",
            indicators=[
                indicator_factory("mood_rating", "mood"),
                indicator_factory("unknown_indicator", "unknown"),  # invalid reference
            ],
        )
        with pytest.raises(ValueError, match="references unknown construct 'unknown'"):
            CausalSpec(latent=latent, measurement=measurement)

    def test_latent_construct_without_indicator_is_valid(
        self, construct_factory, indicator_factory
    ):
        """Latent constructs without indicators are allowed (A2 deferred to y0)."""
        latent = LatentModel(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
        )
        measurement = MeasurementModel(
            model_clock="1d",
            indicators=[
                indicator_factory("mood_rating", "mood"),
                # stress has no indicator - it's a latent construct
            ],
        )
        # This should now be valid - y0 will check identification in Stage 3
        causal_spec = CausalSpec(latent=latent, measurement=measurement)
        assert len(causal_spec.latent.constructs) == 2
        assert len(causal_spec.measurement.indicators) == 1

    def test_build_digraph_from_causal_spec(self, construct_factory, indicator_factory):
        """build_digraph produces correct topology from a CausalSpec's latent model."""
        from causal_ssm_agent.utils.causal_spec import build_digraph

        latent = LatentModel(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
        )
        measurement = MeasurementModel(
            model_clock="1d",
            indicators=[
                indicator_factory("mood_rating", "mood"),
            ],
        )
        causal_spec = CausalSpec(latent=latent, measurement=measurement)
        G = build_digraph(causal_spec.latent.model_dump())

        assert "stress" in G.nodes
        assert "mood" in G.nodes
        assert ("stress", "mood") in G.edges

    def test_get_edge_lag_hours(self, construct_factory, indicator_factory):
        """CausalSpec.get_edge_lag_hours returns model_clock_hours for lagged, 0 for contemporaneous."""
        latent = LatentModel(
            constructs=[
                construct_factory("sleep", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[
                CausalEdge(
                    cause="sleep", effect="mood", description="Sleep affects mood", lagged=True
                )
            ],
        )
        measurement = MeasurementModel(
            model_clock="1d",
            indicators=[
                indicator_factory("sleep_hours", "sleep"),
                indicator_factory("mood_rating", "mood"),
            ],
        )
        causal_spec = CausalSpec(latent=latent, measurement=measurement)
        lag = causal_spec.get_edge_lag_hours(latent.edges[0])
        assert lag == 24  # 1 day


class TestParseDurationToHours:
    """Tests for parse_duration_to_hours function."""

    def test_seconds(self):
        assert parse_duration_to_hours("3600s") == 1.0

    def test_minutes(self):
        assert parse_duration_to_hours("60m") == 1.0

    def test_hours(self):
        assert parse_duration_to_hours("4h") == 4.0
        assert parse_duration_to_hours("1h") == 1.0

    def test_days(self):
        assert parse_duration_to_hours("1d") == 24.0
        assert parse_duration_to_hours("7d") == 168.0

    def test_weeks(self):
        assert parse_duration_to_hours("1w") == 168.0
        assert parse_duration_to_hours("2w") == 336.0

    def test_months(self):
        assert parse_duration_to_hours("1mo") == 720.0

    def test_quarters(self):
        assert parse_duration_to_hours("1q") == 2160.0

    def test_years(self):
        assert parse_duration_to_hours("1y") == 8760.0

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration_to_hours("abc")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration_to_hours("5x")

    def test_zero_duration(self):
        with pytest.raises(ValueError, match="positive"):
            parse_duration_to_hours("0d")

    def test_no_number(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration_to_hours("d")

    def test_fractional_days(self):
        """model_clock_days property converts correctly."""
        m = MeasurementModel(
            model_clock="6h",
            indicators=[
                Indicator(
                    name="x",
                    construct_name="X",
                    how_to_measure="test",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ],
        )
        assert m.model_clock_hours == 6.0
        assert m.model_clock_days == 0.25

    def test_invalid_model_clock_on_measurement_model(self):
        """MeasurementModel rejects invalid model_clock."""
        with pytest.raises(ValueError, match="Invalid duration"):
            MeasurementModel(
                model_clock="bad",
                indicators=[
                    Indicator(
                        name="x",
                        construct_name="X",
                        how_to_measure="test",
                        measurement_dtype="continuous",
                        aggregation="mean",
                    ),
                ],
            )


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
