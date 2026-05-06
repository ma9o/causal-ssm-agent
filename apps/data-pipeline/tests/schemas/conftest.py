"""Factory fixtures for Construct/Indicator schema tests."""

import pytest

from causal_ssm_agent.artifacts import (
    Construct,
    Indicator,
    IndicatorPolarity,
    Role,
    TemporalStatus,
)


@pytest.fixture
def construct_factory():
    """Factory for creating Construct objects.

    Usage:
        def test_something(construct_factory):
            stress = construct_factory("stress", Role.EXOGENOUS)
            mood = construct_factory("mood", Role.ENDOGENOUS, is_outcome=True)
    """

    def _make(
        name: str,
        role: Role = Role.ENDOGENOUS,
        is_outcome: bool = False,
        temporal_status: TemporalStatus = TemporalStatus.TIME_VARYING,
    ) -> Construct:
        return Construct(
            name=name,
            description=f"{name} description",
            role=role,
            is_outcome=is_outcome,
            temporal_status=temporal_status,
        )

    return _make


@pytest.fixture
def indicator_factory():
    """Factory for creating Indicator objects.

    Usage:
        def test_something(indicator_factory):
            ind = indicator_factory("mood_rating", "mood")
    """

    def _make(
        name: str,
        construct_name: str,
        dtype: str = "continuous",
        aggregation: str = "mean",
        construct_polarity: IndicatorPolarity = IndicatorPolarity.POSITIVE,
        ordinal_levels: list[str] | None = None,
        source_columns: list[str] | None = None,
        extraction_mode: str = "semantic",
    ) -> Indicator:
        # Auto-provide ordinal_levels for ordinal dtype if not specified
        if dtype == "ordinal" and ordinal_levels is None:
            ordinal_levels = ["low", "medium", "high"]
        return Indicator(
            name=name,
            construct_name=construct_name,
            construct_polarity=construct_polarity,
            how_to_measure=f"Extract {name}",
            measurement_dtype=dtype,
            aggregation=aggregation,
            ordinal_levels=ordinal_levels,
            source_columns=source_columns or [name],
            extraction_mode=extraction_mode,
        )

    return _make
