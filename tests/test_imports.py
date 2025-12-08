"""Minimal test suite - verify code interprets correctly."""

import pytest


def test_import_pipeline():
    from causal_agent.flows.pipeline import causal_inference_pipeline
    assert callable(causal_inference_pipeline)


def test_import_orchestrator():
    from causal_agent.orchestrator.agents import propose_structure
    from causal_agent.orchestrator.schemas import ProposedStructure, CausalEdge
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
    from causal_agent.orchestrator.schemas import ProposedStructure, Dimension, CausalEdge

    structure = ProposedStructure(
        dimensions=[
            Dimension(name="X", description="cause", dtype="continuous", example_values=["1.0", "2.0"]),
            Dimension(name="Y", description="effect", dtype="continuous", example_values=["3.0", "4.0"]),
        ],
        time_granularity="hourly",
        autocorrelations=["X"],
        edges=[CausalEdge(cause="X", effect="Y")],
        reasoning="test",
    )

    G = structure.to_networkx()
    assert "X" in G.nodes
    assert "Y" in G.nodes
    assert ("X", "Y") in G.edges
