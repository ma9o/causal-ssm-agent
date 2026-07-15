"""Harness-backed agent session primitives.

The :mod:`mcp_server` module exposes a :class:`~nof1_causal_lab.utils.openrouter_client.Tool`
list as an in-process MCP server that Claude Code and Codex can connect to.
Pi receives the same tool surface through a narrow generated extension and
localhost bridge. Backend-specific ``AgentSession`` implementations live
alongside those transports.
"""
