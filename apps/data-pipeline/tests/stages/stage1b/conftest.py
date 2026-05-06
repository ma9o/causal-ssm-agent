"""Stage 1b fixtures (identifiability / proxy resolution)."""

import pytest


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
