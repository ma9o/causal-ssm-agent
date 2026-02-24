"""Minimal test suite - verify code interprets correctly."""


def test_import_pipeline():
    from causal_ssm_agent.flows.pipeline import causal_inference_pipeline

    assert callable(causal_inference_pipeline)


def test_import_orchestrator():
    from causal_ssm_agent.orchestrator.agents import propose_latent_model, propose_measurement_model

    assert callable(propose_latent_model)
    assert callable(propose_measurement_model)


def test_import_workers():
    from causal_ssm_agent.workers import process_chunk, process_chunks

    assert callable(process_chunk)
    assert callable(process_chunks)


def test_import_utils():
    from causal_ssm_agent.utils.data import (
        load_text_chunks,
    )

    assert callable(load_text_chunks)


def test_import_stage0():
    from causal_ssm_agent.flows.stages.stage0_preprocess import preprocess_raw_input

    assert callable(preprocess_raw_input)


def test_schema_to_networkx():
    from causal_ssm_agent.orchestrator.schemas import (
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
                temporal_scale="hourly",
            ),
            Construct(
                name="Y",
                description="effect variable",
                role=Role.ENDOGENOUS,
                is_outcome=True,
                temporal_status=TemporalStatus.TIME_VARYING,
                temporal_scale="hourly",
            ),
        ],
        edges=[CausalEdge(cause="X", effect="Y", description="X causes Y", lagged=True)],
    )

    measurement = MeasurementModel(
        indicators=[
            Indicator(
                name="x_indicator",
                construct_name="X",
                how_to_measure="Extract X from data",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
            Indicator(
                name="y_indicator",
                construct_name="Y",
                how_to_measure="Extract Y from data",
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
