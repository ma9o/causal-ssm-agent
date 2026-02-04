"""Shared fixtures for DSEM tests.

This module provides reusable fixtures to reduce duplication across test files:
- Factory fixtures for creating schema objects (constructs, indicators)
- Canonical graph fixtures (simple chains, confounded graphs, front-door)
- Sample DataFrames for aggregation tests

For non-fixture helpers (MockPrediction, make_mock_generate), see helpers.py.
"""

import polars as pl
import pytest

from dsem_agent.orchestrator.schemas import (
    CausalEdge,
    Construct,
    Indicator,
    LatentModel,
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
            stress = construct_factory("stress", "daily", Role.EXOGENOUS)
            mood = construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True)
    """

    def _make(
        name: str,
        granularity: str | None = "daily",
        role: Role = Role.ENDOGENOUS,
        is_outcome: bool = False,
    ) -> Construct:
        temporal_status = (
            TemporalStatus.TIME_VARYING if granularity else TemporalStatus.TIME_INVARIANT
        )
        return Construct(
            name=name,
            description=f"{name} description",
            role=role,
            is_outcome=is_outcome,
            temporal_status=temporal_status,
            causal_granularity=granularity,
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
        construct: str,
        granularity: str = "daily",
        dtype: str = "continuous",
        aggregation: str = "mean",
    ) -> Indicator:
        return Indicator(
            name=name,
            construct=construct,
            how_to_measure=f"Extract {name}",
            measurement_granularity=granularity,
            measurement_dtype=dtype,
            aggregation=aggregation,
        )

    return _make


@pytest.fixture
def latent_model_factory(construct_factory):
    """Factory for creating LatentModel objects.

    Usage:
        def test_something(latent_model_factory):
            model = latent_model_factory(
                constructs=[("X", "daily", Role.EXOGENOUS), ("Y", "daily", Role.ENDOGENOUS, True)],
                edges=[("X", "Y")]
            )
    """

    def _make(
        constructs: list[tuple],
        edges: list[tuple[str, str]],
    ) -> LatentModel:
        construct_objs = []
        for c in constructs:
            if len(c) == 3:
                name, gran, role = c
                is_outcome = False
            else:
                name, gran, role, is_outcome = c
            construct_objs.append(construct_factory(name, gran, role, is_outcome))

        edge_objs = [
            CausalEdge(cause=cause, effect=effect, description=f"{cause} causes {effect}")
            for cause, effect in edges
        ]

        return LatentModel(constructs=construct_objs, edges=edge_objs)

    return _make


# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL GRAPH FIXTURES (dict format for identifiability tests)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_chain_latent():
    """Simple chain: A -> B -> C (all observable, always identifiable)."""
    return {
        "constructs": [
            {"name": "A", "role": "exogenous"},
            {"name": "B", "role": "endogenous"},
            {"name": "C", "role": "endogenous", "is_outcome": True},
        ],
        "edges": [
            {"cause": "A", "effect": "B", "description": "A causes B"},
            {"cause": "B", "effect": "C", "description": "B causes C"},
        ],
    }


@pytest.fixture
def simple_chain_measurement():
    """Measurement model for simple_chain_latent (all observed)."""
    return {
        "indicators": [
            {"name": "a_ind", "construct": "A", "how_to_measure": "test"},
            {"name": "b_ind", "construct": "B", "how_to_measure": "test"},
            {"name": "c_ind", "construct": "C", "how_to_measure": "test"},
        ]
    }


@pytest.fixture
def confounded_latent():
    """Confounded graph: A -> B with U -> A, U -> B (U unobserved).

    Creates A <-> B in projected ADMG, blocking identification.
    """
    return {
        "constructs": [
            {"name": "A", "role": "endogenous"},
            {"name": "B", "role": "endogenous", "is_outcome": True},
            {"name": "U", "role": "exogenous"},  # Unobserved confounder
        ],
        "edges": [
            {"cause": "A", "effect": "B", "description": "A causes B"},
            {"cause": "U", "effect": "A", "description": "U causes A"},
            {"cause": "U", "effect": "B", "description": "U causes B"},
        ],
    }


@pytest.fixture
def confounded_measurement():
    """Measurement model for confounded_latent (U unobserved)."""
    return {
        "indicators": [
            {"name": "a_ind", "construct": "A", "how_to_measure": "test"},
            {"name": "b_ind", "construct": "B", "how_to_measure": "test"},
        ]
    }


@pytest.fixture
def frontdoor_latent():
    """Front-door graph: X -> M -> Y with U -> X, U -> Y.

    X -> Y is identifiable via front-door criterion through M.
    """
    return {
        "constructs": [
            {"name": "X", "role": "endogenous"},
            {"name": "M", "role": "endogenous"},  # Mediator
            {"name": "Y", "role": "endogenous", "is_outcome": True},
            {"name": "U", "role": "exogenous"},  # Unobserved confounder
        ],
        "edges": [
            {"cause": "X", "effect": "M", "description": "X causes M"},
            {"cause": "M", "effect": "Y", "description": "M causes Y"},
            {"cause": "U", "effect": "X", "description": "U causes X"},
            {"cause": "U", "effect": "Y", "description": "U causes Y"},
        ],
    }


@pytest.fixture
def frontdoor_measurement():
    """Measurement model for frontdoor_latent (U unobserved)."""
    return {
        "indicators": [
            {"name": "x_ind", "construct": "X", "how_to_measure": "test"},
            {"name": "m_ind", "construct": "M", "how_to_measure": "test"},
            {"name": "y_ind", "construct": "Y", "how_to_measure": "test"},
        ]
    }


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
                "causal_granularity": "daily",
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
                "causal_granularity": "daily",
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
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
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def stage1b_measurement_missing_confounder():
    """Measurement model missing the confounder (non-identifiable)."""
    return {
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def stage1b_measurement_with_confounder():
    """Measurement model with confounder indicator (identifiable)."""
    return {
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "confounder_proxy",
                "construct": "Confounder",
                "how_to_measure": "Proxy measurement for the confounder",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def stage1b_proxy_response_success():
    """Successful proxy response that adds confounder indicator."""
    return {
        "new_proxies": [
            {
                "construct": "Confounder",
                "indicators": ["confounder_proxy"],
                "justification": "This proxy captures the confounder",
            }
        ],
        "unfeasible_confounders": [],
    }


@pytest.fixture
def stage1b_proxy_response_empty():
    """Empty proxy response (no proxies found)."""
    return {
        "new_proxies": [],
        "unfeasible_confounders": ["Confounder"],
    }


@pytest.fixture
def stage1b_dummy_chunks():
    """Dummy data chunks for measurement model proposal."""
    return [
        "Day 1: Patient took 10mg treatment, outcome score was 5.",
        "Day 2: Patient took 15mg treatment, outcome score was 7.",
        "Day 3: Patient took 10mg treatment, outcome score was 6.",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# POLARS DATAFRAME FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_aggregation_df():
    """Sample DataFrame for aggregation tests with group/value columns."""
    return pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )


@pytest.fixture
def sample_grouped_df():
    """Sample DataFrame for grouped aggregation tests."""
    return pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1.0, 3.0, 10.0, 20.0],
        }
    )


@pytest.fixture
def daily_aggregation_schema():
    """DSEMModel schema with daily causal_granularity constructs."""
    return {
        "latent": {
            "constructs": [
                {"name": "body_temp", "causal_granularity": "daily"},
                {"name": "activity", "causal_granularity": "daily"},
            ],
            "edges": [],
        },
        "measurement": {
            "indicators": [
                {
                    "name": "temperature",
                    "construct_name": "body_temp",
                    "aggregation": "mean",
                },
                {
                    "name": "step_count",
                    "construct_name": "activity",
                    "aggregation": "sum",
                },
            ],
        },
    }


@pytest.fixture
def worker_measurement_dfs():
    """Sample worker dataframes with timestamps.

    Data layout:
    - 2024-01-01: temperature=[20, 22], step_count=[5000, 2000]
    - 2024-01-02: temperature=[24], step_count=[3000, 4000]
    """
    df1 = pl.DataFrame(
        {
            "indicator": [
                "temperature",
                "temperature",
                "step_count",
                "step_count",
            ],
            "value": [20.0, 22.0, 5000, 3000],
            "timestamp": [
                "2024-01-01 10:00",
                "2024-01-01 14:00",
                "2024-01-01 08:00",
                "2024-01-02 09:00",
            ],
        },
        schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
    )

    df2 = pl.DataFrame(
        {
            "indicator": ["temperature", "step_count", "step_count"],
            "value": [24.0, 2000, 4000],
            "timestamp": ["2024-01-02 12:00", "2024-01-01 20:00", "2024-01-02 18:00"],
        },
        schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
    )

    return [df1, df2]
