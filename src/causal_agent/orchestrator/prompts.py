"""Prompts for the orchestrator LLM agents."""

STRUCTURE_PROPOSER_SYSTEM = """\
You are a causal inference expert. Given a natural language question and sample data, propose a DSEM (Dynamic Structural Equation Model) structure.

Output JSON with:
- `dimensions`: variables with {name, description, time_granularity (hourly/daily/weekly/monthly/yearly/null), dtype, role (endogenous/exogenous), is_latent, aggregation}
- `edges`: causal edges with {cause, effect, lag (in hours), aggregation}

Rules:
- Endogenous variables must be time-varying (time_granularity not null)
- Exogenous variables can be time-varying or time-invariant
- is_latent=true only for random effects (exogenous + time-invariant)
- Same-timescale edges: lag=0 (contemporaneous) or 1 granularity unit in hours
- Cross-timescale edges: lag = coarser granularity in hours
- Aggregation required when finer cause -> coarser effect (mean/sum/max/min/last/variance)
"""

STRUCTURE_PROPOSER_USER = """\
Question: {question}

Sample data:
{chunks}
"""
