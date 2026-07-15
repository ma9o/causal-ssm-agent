"""Tests for utils/llm.py pure utility functions.

Covers validation formatting and the OpenRouter client.
"""

import asyncio
import logging
from typing import Any, cast

import pytest

from nof1_causal_lab.utils.llm import (
    _validate_json_and_format,
)
from tests.agent._support import make_worker_tool as _make_worker_tool
from tests.helpers import run_async as _run


def _require_mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


# =============================================================================
# _validate_json_and_format
# =============================================================================


class TestValidateJsonAndFormat:
    def test_valid_returns_valid_string(self):
        def validate(data):
            return data, []

        result = _validate_json_and_format('{"key": "value"}', validate)
        assert result == "VALID"

    def test_errors_returned_as_string(self):
        def validate(data):
            return None, ["Field 'x' is required", "Field 'y' must be positive"]

        result = _validate_json_and_format('{"key": "value"}', validate)
        assert "VALIDATION ERRORS" in result
        assert "Field 'x' is required" in result
        assert "Field 'y' must be positive" in result

    def test_invalid_json_returns_parse_error(self):
        def validate(data):
            return data, []

        result = _validate_json_and_format("not json", validate)
        assert "JSON parse error" in result

    def test_capture_stores_observation_rows(self):
        def validate(data):
            return "validated_result", []

        capture = {}
        _validate_json_and_format(
            '{"key": "value"}',
            validate,
            capture=capture,
            capture_key="test",
            capture_result=False,
        )
        assert capture["test"] == {"key": "value"}

    def test_capture_stores_result(self):
        def validate(data):
            return "validated_result", []

        capture = {}
        _validate_json_and_format(
            '{"key": "value"}',
            validate,
            capture=capture,
            capture_key="test",
            capture_result=True,
        )
        assert capture["test"] == "validated_result"

    def test_no_capture_on_errors(self):
        def validate(data):
            return None, ["error"]

        capture = {}
        _validate_json_and_format(
            '{"key": "value"}',
            validate,
            capture=capture,
            capture_key="test",
        )
        assert "test" not in capture


class TestWorkerValidationTools:
    def test_validate_worker_tool_stops_on_valid_output(self):
        tool, _capture = _make_worker_tool()
        assert tool.stop_on_success is True
        assert tool.success_output == "VALID"


class _FakeChatCompletions:
    def __init__(self, response: dict[str, object], seen: dict[str, object]):
        self._response = response
        self._seen = seen

    async def create(self, **kwargs):
        self._seen["kwargs"] = kwargs
        return self._response


class _FakeOpenRouterClient:
    def __init__(self, response: dict[str, object], seen: dict[str, object]):
        self.chat = type(
            "_FakeChatNamespace",
            (),
            {"completions": _FakeChatCompletions(response, seen)},
        )()


class TestOpenRouterClient:
    def test_call_model_enforces_local_timeout(self, monkeypatch):
        from nof1_causal_lab.utils import openrouter_client

        seen: dict[str, object] = {}

        class _SlowChatCompletions:
            async def create(self, **kwargs):
                seen["kwargs"] = kwargs
                await asyncio.sleep(0.05)
                return {"choices": []}

        slow_client = type(
            "_SlowOpenRouterClient",
            (),
            {
                "chat": type(
                    "_SlowChatNamespace",
                    (),
                    {"completions": _SlowChatCompletions()},
                )()
            },
        )()

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda _api_key=None: slow_client,
        )

        timeout_seconds = cast("Any", 0.01)
        with pytest.raises(TimeoutError, match=r"call_model timed out after 0\.01s"):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=openrouter_client.GenerateConfig(timeout=timeout_seconds),
                )
            )

        assert _require_mapping(seen["kwargs"])["timeout"] == 0.01

    def test_call_model_logs_completion_tool_calls_and_reasoning(self, monkeypatch, caplog):
        from nof1_causal_lab.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "final answer",
                        "reasoning": "thought process",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "validate_extractions",
                                    "arguments": {"ok": True},
                                },
                            }
                        ],
                    },
                }
            ],
        }

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda: _FakeOpenRouterClient(response, seen),
        )

        with caplog.at_level(logging.INFO):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=openrouter_client.GenerateConfig(),
                    log_label="extraction chunk=1",
                )
            )

        assert _require_mapping(seen["kwargs"])["model"] == "test-model"
        assert "call_model completion:\nfinal answer" in caplog.text
        assert '"name": "validate_extractions"' in caplog.text
        assert "call_model reasoning:\nthought process" in caplog.text

    def test_call_model_logs_completion_without_label(self, monkeypatch, caplog):
        from nof1_causal_lab.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "unlabeled completion",
                    },
                }
            ],
        }

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda: _FakeOpenRouterClient(response, seen),
        )

        with caplog.at_level(logging.INFO):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=openrouter_client.GenerateConfig(),
                )
            )

        assert _require_mapping(seen["kwargs"])["messages"] == [
            {"role": "user", "content": "hello"}
        ]
        assert "call_model completion:\nunlabeled completion" in caplog.text

    def test_call_model_strips_repo_openrouter_prefix(self, monkeypatch):
        from nof1_causal_lab.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "anthropic/claude-sonnet-4",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "ok",
                    },
                }
            ],
        }

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda: _FakeOpenRouterClient(response, seen),
        )

        _run(
            openrouter_client.call_model(
                "openrouter/anthropic/claude-sonnet-4",
                [{"role": "user", "content": "hello"}],
                config=openrouter_client.GenerateConfig(),
            )
        )

        assert _require_mapping(seen["kwargs"])["model"] == "anthropic/claude-sonnet-4"

    def test_call_model_uses_ambient_env_openrouter_key(self, monkeypatch):
        from nof1_causal_lab.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "ok",
                    },
                }
            ],
        }

        def fake_async_openai(*, base_url: str, api_key: str):
            del base_url
            seen["api_key"] = api_key
            return _FakeOpenRouterClient(response, seen)

        monkeypatch.setattr(openrouter_client, "_openrouter_client", None)
        monkeypatch.setattr(openrouter_client, "AsyncOpenAI", fake_async_openai)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

        _run(
            openrouter_client.call_model(
                "test-model",
                [{"role": "user", "content": "hello"}],
                config=openrouter_client.GenerateConfig(),
            )
        )

        assert seen["api_key"] == "env-key"
        assert _require_mapping(seen["kwargs"])["model"] == "test-model"

    def test_call_model_uses_reasoning_config_in_extra_body(self, monkeypatch):
        from nof1_causal_lab.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "ok",
                    },
                }
            ],
        }

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda: _FakeOpenRouterClient(response, seen),
        )

        _run(
            openrouter_client.call_model(
                "test-model",
                [{"role": "user", "content": "hello"}],
                config=openrouter_client.GenerateConfig(reasoning_effort="high"),
            )
        )

        assert _require_mapping(seen["kwargs"])["extra_body"] == {
            "provider": {"sort": "throughput"},
            "reasoning": {"effort": "high"},
        }
