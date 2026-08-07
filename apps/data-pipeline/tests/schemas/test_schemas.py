"""Tests for causal design schema computed properties and utility functions.

Object-level construction validation (Construct, LatentStructure, Indicator,
MeasurementStructure) is covered by test_schema_validators.py via dict validators.
This file tests CausalDesign composition, computed properties, and utility
functions that are not exercised through dict validation.
"""

from typing import Any

import pytest

from nof1_causal_lab.artifacts import (
    CausalDesign,
    CausalEdge,
    ComputedRule,
    Construct,
    LatentStructure,
    MeasurementStructure,
    Role,
    TemporalStatus,
    check_semantic_collisions,
    parse_duration_to_hours,
)
from nof1_causal_lab.artifacts import (
    Indicator as IndicatorModel,
)
from nof1_causal_lab.utils.observation_semantics import (
    AnchorPolicy,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
)


def Indicator(**kwargs: Any) -> IndicatorModel:
    """Build test indicators with the current required schema defaults."""
    kwargs.setdefault("construct_polarity", "positive")
    computed_rule = kwargs.get("computed_rule")
    if isinstance(computed_rule, dict):
        kwargs["computed_rule"] = ComputedRule(**computed_rule)
    return IndicatorModel(**kwargs)


class TestConstruct:
    """Tests for Construct validation."""

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


class TestLatentStructure:
    """Tests for LatentStructure validation."""

    def test_valid_simple_structure(self, construct_factory):
        """Simple valid structure passes validation."""
        structure = LatentStructure(
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
            LatentStructure(
                constructs=[construct_factory("mood", Role.ENDOGENOUS, is_outcome=True)],
                edges=[CausalEdge(cause="unknown", effect="mood", description="Test edge")],
            )

    def test_invalid_edge_effect_not_in_constructs(self, construct_factory):
        """Edge effect must exist in constructs."""
        with pytest.raises(ValueError, match="Edge effect 'unknown' not in constructs"):
            LatentStructure(
                constructs=[
                    construct_factory("stress", Role.EXOGENOUS),
                    construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="stress", effect="unknown", description="Test edge")],
            )

    def test_invalid_exogenous_cannot_be_effect(self, construct_factory):
        """Exogenous construct cannot be an effect."""
        with pytest.raises(ValueError, match="Exogenous construct 'weather' cannot be an effect"):
            LatentStructure(
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
        model = LatentStructure(
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
            LatentStructure(
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
            LatentStructure(
                constructs=[
                    construct_factory("stress", Role.EXOGENOUS),
                    construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[],
            )

    def test_invalid_no_outcome(self, construct_factory):
        """Structure must have exactly one outcome."""
        with pytest.raises(ValueError, match="Exactly one construct must have is_outcome=true"):
            LatentStructure(
                constructs=[
                    construct_factory("stress", Role.EXOGENOUS),
                    construct_factory("mood", Role.ENDOGENOUS),
                ],
                edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
            )

    def test_invalid_multiple_outcomes(self, construct_factory):
        """Structure must have exactly one outcome."""
        with pytest.raises(ValueError, match="Only one outcome allowed"):
            LatentStructure(
                constructs=[
                    construct_factory("stress", Role.ENDOGENOUS, is_outcome=True),
                    construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
                ],
                edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
            )


class TestIndicator:
    """Tests for Indicator validation."""

    def test_invalid_aggregation(self):
        """Invalid aggregation is rejected."""
        with pytest.raises(ValueError, match="aggregation"):
            Indicator(
                name="mood_rating",
                construct_name="mood",
                how_to_measure="Extract mood",
                measurement_dtype="continuous",
                aggregation="invalid_agg",
            )

    def test_invalid_measurement_dtype(self):
        """Invalid measurement_dtype is rejected."""
        with pytest.raises(ValueError, match="measurement_dtype"):
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
                aggregation="last",
            )

    def test_ordinal_needs_at_least_two_levels(self):
        """Ordinal with only one level is rejected."""
        with pytest.raises(ValueError, match="at least 2 items"):
            Indicator(
                name="pain",
                construct_name="pain",
                how_to_measure="Extract pain level",
                measurement_dtype="ordinal",
                aggregation="last",
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
                aggregation="last",
                ordinal_levels=["low", "low", "high"],
            )

    def test_ordinal_valid_levels(self):
        """Ordinal with valid levels passes."""
        ind = Indicator(
            name="pain",
            construct_name="pain",
            how_to_measure="Extract pain level",
            measurement_dtype="ordinal",
            aggregation="last",
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
        with pytest.raises(ValueError, match="extraction_mode"):
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

    def test_computed_binary_point_dtype(self):
        """Computed indicator with binary dtype passes for direct point aggregation."""
        ind = Indicator(
            name="alarm_state",
            construct_name="monitoring",
            how_to_measure="Use the last observed alarm_state value directly",
            measurement_dtype="binary",
            aggregation="last",
            source_columns=["alarm_state"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_ordinal_point_dtype(self):
        """Computed indicator with ordinal dtype passes for direct point aggregation."""
        ind = Indicator(
            name="mood_label",
            construct_name="mood",
            how_to_measure="Use the last observed mood_label value directly",
            measurement_dtype="ordinal",
            aggregation="last",
            ordinal_levels=["bad", "ok", "good"],
            source_columns=["mood_label"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_categorical_point_dtype(self):
        """Computed indicator with categorical dtype passes for direct point aggregation."""
        ind = Indicator(
            name="care_setting",
            construct_name="care_context",
            how_to_measure="Use the first observed care_setting value directly",
            measurement_dtype="categorical",
            aggregation="first",
            categorical_levels=["home", "clinic"],
            source_columns=["care_setting"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_requires_single_source_column(self):
        """Direct computed indicators with 0 or 2+ source_columns are rejected."""
        with pytest.raises(ValueError, match="exactly 1 direct source_column"):
            Indicator(
                name="avg_hr",
                construct_name="health",
                how_to_measure="Use heart_rate",
                measurement_dtype="continuous",
                aggregation="mean",
                source_columns=[],
                extraction_mode="computed",
            )
        with pytest.raises(ValueError, match="exactly 1 direct source_column"):
            Indicator(
                name="avg_hr",
                construct_name="health",
                how_to_measure="Compute from systolic and diastolic",
                measurement_dtype="continuous",
                aggregation="mean",
                source_columns=["systolic_bp", "diastolic_bp"],
                extraction_mode="computed",
            )

    def test_computed_rule_allows_multi_source_deterministic_formula(self):
        """Computed rules can reference multiple source columns deterministically."""
        ind = Indicator(
            name="mean_arterial_pressure",
            construct_name="cardiovascular_health",
            how_to_measure="Compute deterministically from systolic and diastolic blood pressure",
            measurement_dtype="continuous",
            aggregation="mean",
            source_columns=["systolic_bp", "diastolic_bp"],
            computed_rule={"window_expr": "mean(diastolic_bp + (systolic_bp - diastolic_bp) / 3)"},
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"
        assert ind.computed_rule is not None

    def test_computed_rule_rejects_semantic_mode(self):
        """computed_rule is only valid when extraction_mode='computed'."""
        with pytest.raises(ValueError, match="computed_rule but extraction_mode is 'semantic'"):
            Indicator(
                name="low_spo2",
                construct_name="respiratory_status",
                how_to_measure="Deterministically compute low SpO2 from spo2_pct",
                measurement_dtype="binary",
                aggregation="last",
                source_columns=["spo2_pct"],
                computed_rule={
                    "window_expr": "1 if any(spo2_pct < 92) else (0 if count_non_null(spo2_pct) > 0 else None)"
                },
                extraction_mode="semantic",
            )

    def test_computed_rule_rejects_undeclared_source_column(self):
        """computed_rule must reference only declared source_columns."""
        with pytest.raises(ValueError, match="references undeclared source_columns"):
            Indicator(
                name="glucose_out_of_range",
                construct_name="glycemic_control",
                how_to_measure="Count out-of-range glucose values deterministically",
                measurement_dtype="count",
                aggregation="sum",
                source_columns=["glucose_mg_dl"],
                computed_rule={
                    "window_expr": "None if count_non_null(glucose_mg_dl) == 0 else sum(1 if (glucose_mg_dl < 70 or serum_glucose > 180) else 0)"
                },
                extraction_mode="computed",
            )

    def test_computed_rule_requires_source_reference(self):
        """computed_rule must actually use at least one declared source column."""
        with pytest.raises(ValueError, match="does not reference any source_columns"):
            Indicator(
                name="constant_flag",
                construct_name="monitoring",
                how_to_measure="Always emit a constant flag",
                measurement_dtype="binary",
                aggregation="last",
                source_columns=["spo2_pct"],
                computed_rule={"window_expr": "1"},
                extraction_mode="computed",
            )

    def test_computed_still_rejects_invalid_semantics(self):
        """Computed indicators still respect the measurement-semantics grid."""
        with pytest.raises(
            ValueError, match="aggregation 'mean' requires measurement_dtype='continuous'"
        ):
            Indicator(
                name="alarm_state",
                construct_name="monitoring",
                how_to_measure="Use alarm_state directly",
                measurement_dtype="binary",
                aggregation="mean",
                source_columns=["alarm_state"],
                extraction_mode="computed",
            )


class TestMeasurementStructure:
    """Tests for MeasurementStructure."""

    def test_get_indicators_for_construct(self):
        """get_indicators_for_construct returns correct indicators."""
        model = MeasurementStructure(
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
                    aggregation="last",
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


class TestCausalDesign:
    """Tests for CausalDesign validation."""

    def test_valid_causal_design(self, construct_factory, indicator_factory):
        """Valid CausalDesign passes validation."""
        latent = LatentStructure(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Stress affects mood")],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("stress_rating", "stress"),
                indicator_factory("mood_rating", "mood"),
            ],
        )
        causal_design = CausalDesign(latent=latent, measurement=measurement)
        assert len(causal_design.latent.constructs) == 2
        assert len(causal_design.measurement.indicators) == 2

    def test_invalid_indicator_references_unknown_construct(
        self, construct_factory, indicator_factory
    ):
        """Indicator must reference a valid construct."""
        latent = LatentStructure(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("mood_rating", "mood"),
                indicator_factory("unknown_indicator", "unknown"),  # invalid reference
            ],
        )
        with pytest.raises(ValueError, match="references unknown construct 'unknown'"):
            CausalDesign(latent=latent, measurement=measurement)

    def test_latent_construct_without_indicator_is_valid(
        self, construct_factory, indicator_factory
    ):
        """Latent constructs without indicators are allowed (A2 deferred to y0)."""
        latent = LatentStructure(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("mood_rating", "mood"),
                # stress has no indicator - it's a latent construct
            ],
        )
        # This should now be valid - y0 will check identification in validation
        causal_design = CausalDesign(latent=latent, measurement=measurement)
        assert len(causal_design.latent.constructs) == 2
        assert len(causal_design.measurement.indicators) == 1

    def test_known_input_cannot_also_be_scientific_only(self, construct_factory, indicator_factory):
        """An authored construct must have exactly one executable disposition."""
        latent = LatentStructure(
            constructs=[
                construct_factory("x", Role.EXOGENOUS),
                construct_factory("y", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[
                CausalEdge(cause="x", effect="y", description="Treatment path"),
            ],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("x_obs", "x"),
                indicator_factory("y_obs", "y"),
            ],
        )
        with pytest.raises(ValueError, match="both known inputs and scientific-only"):
            CausalDesign.model_validate(
                {
                    "latent": latent.model_dump(),
                    "measurement": measurement.model_dump(),
                    "known_inputs": [{"construct": "x", "source_indicator": "x_obs"}],
                    "scientific_only_constructs": [{"construct": "x", "reason": "context only"}],
                }
            )

    def test_lagged_edge_uses_measurement_clock(self, construct_factory, indicator_factory):
        latent = LatentStructure(
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
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("sleep_hours", "sleep"),
                indicator_factory("mood_rating", "mood"),
            ],
        )
        causal_design = CausalDesign(latent=latent, measurement=measurement)
        assert latent.edges[0].lagged is True
        assert causal_design.measurement.model_clock_hours == 24


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
        m = MeasurementStructure(
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

    def test_invalid_model_clock_on_measurement_structure(self):
        """MeasurementStructure rejects invalid model_clock."""
        with pytest.raises(ValueError, match="Invalid duration"):
            MeasurementStructure(
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


class TestDeriveObservationSemantics:
    """Tests for derive_indicator_observation_semantics."""

    def test_first_maps_to_point_at_window_start(self):
        semantics = derive_indicator_observation_semantics("first", "continuous")
        assert semantics.support_kind == SupportKind.POINT
        assert semantics.summary_operator == SummaryOperator.FIRST
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_START

    def test_last_maps_to_point_at_window_end(self):
        semantics = derive_indicator_observation_semantics("last", "continuous")
        assert semantics.support_kind == SupportKind.POINT
        assert semantics.summary_operator == SummaryOperator.LAST
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_interval_summary_operator_maps_to_interval_support(self):
        semantics = derive_indicator_observation_semantics("sum", "count")
        assert semantics.support_kind == SupportKind.INTERVAL
        assert semantics.summary_operator == SummaryOperator.SUM
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_std_requires_continuous_measurements(self):
        with pytest.raises(
            ValueError, match="aggregation 'std' requires measurement_dtype='continuous'"
        ):
            derive_indicator_observation_semantics("std", "count")

    def test_ordinal_indicators_only_support_point_operators(self):
        with pytest.raises(
            ValueError, match="ordinal indicators currently support only first/last"
        ):
            derive_indicator_observation_semantics("mean", "ordinal")

    def test_unsupported_aggregations_fail_fast(self):
        with pytest.raises(ValueError, match="not yet supported by the measurement structure"):
            derive_indicator_observation_semantics("median", "continuous")


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


class TestIndicatorObservationSemantics:
    """Tests for Indicator computed observation semantics."""

    def test_interval_indicator_serializes_semantics(self, indicator_factory):
        ind = indicator_factory("steps", "activity", aggregation="sum", dtype="count")
        assert ind.support_kind == SupportKind.INTERVAL
        assert ind.summary_operator == SummaryOperator.SUM
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END
        assert ind.requires_interval_summary_measurement is True

    def test_point_indicator_serializes_semantics(self, indicator_factory):
        ind = indicator_factory("last_bp", "bp", aggregation="last", dtype="continuous")
        assert ind.support_kind == SupportKind.POINT
        assert ind.summary_operator == SummaryOperator.LAST
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END
        assert ind.requires_interval_summary_measurement is False

    def test_ordinal_indicator_uses_point_semantics(self, indicator_factory):
        ind = indicator_factory("pain_level", "pain", aggregation="last", dtype="ordinal")
        assert ind.support_kind == SupportKind.POINT
        assert ind.summary_operator == SummaryOperator.LAST
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_unsupported_aggregation_is_rejected_on_indicator(self, indicator_factory):
        with pytest.raises(ValueError, match="not yet supported by the measurement structure"):
            indicator_factory("median_hr", "hr", aggregation="median", dtype="continuous")

    def test_ordinal_interval_summary_is_rejected_on_indicator(self, indicator_factory):
        with pytest.raises(
            ValueError, match="ordinal indicators currently support only first/last"
        ):
            indicator_factory("pain_level", "pain", aggregation="mean", dtype="ordinal")


class TestIndicatorObservationWindow:
    def test_valid_observation_window(self):
        indicator = Indicator(
            name="monthly_mood",
            construct_name="mood",
            how_to_measure="Average mood over the last month",
            measurement_dtype="continuous",
            aggregation="mean",
            observation_window="1mo",
        )

        assert indicator.observation_window == "1mo"

    def test_invalid_observation_window(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            Indicator(
                name="monthly_mood",
                construct_name="mood",
                how_to_measure="Average mood over the last month",
                measurement_dtype="continuous",
                aggregation="mean",
                observation_window="monthly",
            )
