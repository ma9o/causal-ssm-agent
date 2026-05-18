"""Harness-backed agent session primitives.

The :mod:`mcp_server` module exposes a :class:`~causal_ssm_agent.utils.openrouter_client.Tool`
list as an in-process MCP server that an external harness CLI (such as
``claude -p`` or ``codex exec``) can connect to. Backend-specific
``AgentSession`` implementations will land alongside.
"""
