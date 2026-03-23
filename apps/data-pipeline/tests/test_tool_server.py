from fastapi.testclient import TestClient

import causal_ssm_agent.tool_server as tool_server


def test_execute_tool_rejects_invalid_input_before_invoking_tool(monkeypatch):
    client = TestClient(tool_server.app)
    called = False

    def fake_impl(_ctx, _args):
        nonlocal called
        called = True
        return {"result": "should not run"}

    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("stage-1a", "validate_latent_model"),
        fake_impl,
    )

    response = client.post(
        "/api/tools/stage-1a/validate_latent_model",
        json={"workspace_id": "user-123", "input": {}},
    )

    assert response.status_code == 422
    assert called is False


def test_persist_stage_web_patch_uses_shared_persistence_helper(monkeypatch):
    client = TestClient(tool_server.app)

    called_with = {}

    def fake_persist_web_patch(stage_id, patch, workspace_id):
        called_with["stage_id"] = stage_id
        called_with["patch"] = patch
        called_with["workspace_id"] = workspace_id
        return {"outcome": "success", **patch}

    monkeypatch.setattr(tool_server, "persist_web_patch", fake_persist_web_patch)

    response = client.post(
        "/api/stages/stage-6/persist-web-patch",
        json={"workspace_id": "user-123", "patch": {"llm_trace": {"messages": []}}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "payload": {"outcome": "success", "llm_trace": {"messages": []}},
    }
    assert called_with == {
        "stage_id": "stage-6",
        "patch": {"llm_trace": {"messages": []}},
        "workspace_id": "user-123",
    }
