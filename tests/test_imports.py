"""Minimal test suite - verify code interprets correctly."""

import pytest


def test_import_pipeline():
    from causal_agent.flows.pipeline import causal_inference_pipeline
    assert callable(causal_inference_pipeline)


def test_import_orchestrator():
    from causal_agent.orchestrator.agents import propose_structure
    from causal_agent.orchestrator.schemas import DSEMStructure, CausalEdge
    assert callable(propose_structure)


def test_import_utils():
    from causal_agent.utils.data import (
        load_text_chunks,
        resolve_input_path,
        load_query,
        get_latest_preprocessed_file,
    )
    assert callable(load_text_chunks)


def test_preprocessing_script():
    from scripts.preprocess_google_takeout import (
        parse_takeout_zip,
        export_as_text_chunks,
    )
    assert callable(parse_takeout_zip)


def test_schema_to_networkx():
    from causal_agent.orchestrator.schemas import DSEMStructure, Dimension, CausalEdge

    structure = DSEMStructure(
        dimensions=[
            Dimension(
                name="X",
                description="cause variable",
                time_granularity="hourly",
                dtype="continuous",
                role="exogenous",
            ),
            Dimension(
                name="Y",
                description="effect variable",
                time_granularity="hourly",
                dtype="continuous",
                role="endogenous",
            ),
        ],
        edges=[CausalEdge(cause="X", effect="Y", lag=1)],
    )

    G = structure.to_networkx()
    assert "X" in G.nodes
    assert "Y" in G.nodes
    assert ("X", "Y") in G.edges
