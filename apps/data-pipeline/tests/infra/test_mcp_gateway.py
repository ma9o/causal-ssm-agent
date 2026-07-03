"""MCP gateway: static machine description stays in lockstep with the graph."""

from nof1_causal_lab.machine.graph import ARTIFACT_GRAPH
from nof1_causal_lab.machine.runners import STAGE_EXECUTION_CLASS
from nof1_causal_lab.mcp_gateway import describe_machine


def _run(coro):
    import asyncio

    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_execution_classes_cover_exactly_the_graph():
    assert set(STAGE_EXECUTION_CLASS) == {spec.stage_id for spec in ARTIFACT_GRAPH}


def test_describe_machine_serves_graph_and_classes():
    description = _run(describe_machine())

    stages = {entry["stage_id"]: entry for entry in description["stages"]}
    assert set(stages) == {spec.stage_id for spec in ARTIFACT_GRAPH}
    for spec in ARTIFACT_GRAPH:
        entry = stages[spec.stage_id]
        assert entry["consumes"] == list(spec.consumes)
        assert entry["produces"] == list(spec.produces)
        assert entry["execution_class"] == STAGE_EXECUTION_CLASS[spec.stage_id]
    assert description["topological_stage_order"][0] == "stage-0"
    assert "question" in description["artifact_ids"]
