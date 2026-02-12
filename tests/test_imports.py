"""Minimal test suite - verify code interprets correctly."""


def test_import_pipeline():
    from dsem_agent.flows.pipeline import causal_inference_pipeline

    assert callable(causal_inference_pipeline)


def test_import_orchestrator():
    from dsem_agent.orchestrator.agents import propose_latent_model, propose_measurement_model

    assert callable(propose_latent_model)
    assert callable(propose_measurement_model)


def test_import_workers():
    from dsem_agent.workers import process_chunk, process_chunks

    assert callable(process_chunk)
    assert callable(process_chunks)


def test_import_utils():
    from dsem_agent.utils.data import (
        load_text_chunks,
    )

    assert callable(load_text_chunks)


def test_preprocessing_script():
    from evals.scripts.preprocess_google_takeout import (
        parse_takeout_zip,
    )

    assert callable(parse_takeout_zip)


def test_schema_to_networkx():
    from dsem_agent.orchestrator.schemas import (
        CausalEdge,
        CausalSpec,
        Construct,
        Indicator,
        LatentModel,
        MeasurementModel,
        Role,
        TemporalStatus,
    )

    latent = LatentModel(
        constructs=[
            Construct(
                name="X",
                description="cause variable",
                role=Role.EXOGENOUS,
                temporal_status=TemporalStatus.TIME_VARYING,
                causal_granularity="hourly",
            ),
            Construct(
                name="Y",
                description="effect variable",
                role=Role.ENDOGENOUS,
                is_outcome=True,
                temporal_status=TemporalStatus.TIME_VARYING,
                causal_granularity="hourly",
            ),
        ],
        edges=[CausalEdge(cause="X", effect="Y", description="X causes Y", lagged=True)],
    )

    measurement = MeasurementModel(
        indicators=[
            Indicator(
                name="x_indicator",
                construct="X",
                how_to_measure="Extract X from data",
                measurement_granularity="finest",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
            Indicator(
                name="y_indicator",
                construct="Y",
                how_to_measure="Extract Y from data",
                measurement_granularity="finest",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
        ]
    )

    causal_spec = CausalSpec(latent=latent, measurement=measurement)
    G = causal_spec.to_networkx()

    # Construct nodes exist
    assert "X" in G.nodes
    assert "Y" in G.nodes

    # Indicator nodes exist
    assert "x_indicator" in G.nodes
    assert "y_indicator" in G.nodes

    # Causal edge exists
    assert ("X", "Y") in G.edges
    assert G.edges["X", "Y"]["lag_hours"] == 1  # hourly lag

    # Loading edges exist (construct -> indicator)
    assert ("X", "x_indicator") in G.edges
    assert ("Y", "y_indicator") in G.edges
