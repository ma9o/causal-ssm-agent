"""Shared fixtures for causal SSM tests.

This module provides reusable fixtures to reduce duplication across test files:
- Factory fixtures for creating schema objects (constructs, indicators)
- Stage 1b fixtures (identifiability / proxy resolution)

For LLM/session fakes, see helpers.py. For SSM data builders and recovery
assertions (e.g. make_lgss_data, assert_recovery_ci), see
ssm_test_utils.py.
"""

import pytest

from causal_ssm_agent.artifacts import (
    Construct,
    Indicator,
    IndicatorPolarity,
    Role,
    TemporalStatus,
)

# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1B FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def stage1b_simple_latent():
    """Simple chain: Treatment -> Outcome (all observable)."""
    return {
        "constructs": [
            {
                "name": "Treatment",
                "role": "exogenous",
                "description": "The intervention",
                "temporal_status": "time_invariant",
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
            },
        ],
        "edges": [
            {
                "cause": "Treatment",
                "effect": "Outcome",
                "description": "Treatment causes Outcome",
            },
        ],
    }


@pytest.fixture
def stage1b_confounded_latent():
    """Confounded: Treatment -> Outcome, Confounder -> Treatment, Confounder -> Outcome."""
    return {
        "constructs": [
            {
                "name": "Treatment",
                "role": "endogenous",
                "description": "The intervention",
                "temporal_status": "time_varying",
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
            },
            {
                "name": "Confounder",
                "role": "exogenous",
                "description": "Unmeasured common cause",
                "temporal_status": "time_invariant",
            },
        ],
        "edges": [
            {
                "cause": "Treatment",
                "effect": "Outcome",
                "description": "Treatment causes Outcome",
            },
            {
                "cause": "Confounder",
                "effect": "Treatment",
                "description": "Confounder affects Treatment",
            },
            {
                "cause": "Confounder",
                "effect": "Outcome",
                "description": "Confounder affects Outcome",
            },
        ],
    }


@pytest.fixture
def stage1b_measurement_all_observed():
    """Measurement model with indicators for Treatment and Outcome."""
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "name": "treatment_dose",
                "construct_name": "Treatment",
                "construct_polarity": "positive",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["treatment_dose"],
            },
            {
                "name": "outcome_score",
                "construct_name": "Outcome",
                "construct_polarity": "positive",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["outcome_score"],
            },
        ],
    }


@pytest.fixture
def stage1b_dummy_chunks():
    """Dummy data chunks for measurement model proposal."""
    return [
        "Day 1: Patient took 10mg treatment, outcome score was 5.",
        "Day 2: Patient took 15mg treatment, outcome score was 7.",
        "Day 3: Patient took 10mg treatment, outcome score was 6.",
    ]

