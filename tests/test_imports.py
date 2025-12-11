"""Minimal test suite - verify code interprets correctly."""

import pytest


def test_import_pipeline():
    from causal_agent.flows.pipeline import causal_inference_pipeline
    assert callable(causal_inference_pipeline)


def test_import_orchestrator():
    from causal_agent.orchestrator.agents import propose_structure
    from causal_agent.orchestrator.schemas import DSEMStructure, CausalEdge
    assert callable(propose_structure)


def test_import_workers():
    from causal_agent.workers import process_chunk, process_chunks, WorkerOutput
    assert callable(process_chunk)
    assert callable(process_chunks)


def test_import_utils():
    from causal_agent.utils.data import (
        load_text_chunks,
        resolve_input_path,
        load_query,
        get_latest_preprocessed_file,
    )
    assert callable(load_text_chunks)


def test_preprocessing_script():
    from evals.scripts.preprocess_google_takeout import (
        parse_takeout_zip,
        export_as_text_chunks,
    )
    assert callable(parse_takeout_zip)


def test_schema_to_networkx():
    from causal_agent.orchestrator.schemas import DSEMStructure, Dimension, CausalEdge, Role, Observability, TemporalStatus

    structure = DSEMStructure(
        dimensions=[
            Dimension(
                name="X",
                description="cause variable",
                role=Role.EXOGENOUS,
                observability=Observability.OBSERVED,
                how_to_measure="Extract X from data",
                temporal_status=TemporalStatus.TIME_VARYING,
                causal_granularity="hourly",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
            Dimension(
                name="Y",
                description="effect variable",
                role=Role.ENDOGENOUS,
                is_outcome=True,
                observability=Observability.OBSERVED,
                how_to_measure="Extract Y from data",
                temporal_status=TemporalStatus.TIME_VARYING,
                causal_granularity="hourly",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
        ],
        edges=[CausalEdge(cause="X", effect="Y", description="X causes Y", lagged=True)],
    )

    G = structure.to_networkx()
    assert "X" in G.nodes
    assert "Y" in G.nodes
    assert ("X", "Y") in G.edges
    assert G.edges["X", "Y"]["lag_hours"] == 1  # hourly lag
