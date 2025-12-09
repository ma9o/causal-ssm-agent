"""Prompts for the orchestrator LLM agents."""

STRUCTURE_PROPOSER_SYSTEM = """\
You are a causal inference expert. Given a natural language question and sample data, propose a causal model structure.

Output JSON with:
- `dimensions`: variables to extract (name, description, dtype, time_granularity, autocorrelated)
- `edges`: contemporaneous causal edges {cause, effect}
- `cross_lags`: cross-lagged edges {cause, effect, lag} where cause at t-lag affects effect at t
"""

STRUCTURE_PROPOSER_USER = """\
Question: {question}

Sample data:
{chunks}
"""
