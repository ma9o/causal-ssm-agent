# Stage 1a: Latent Model Proposal

Translates a natural-language research question into a theoretical causal DAG over constructs.

## At a Glance

| Property | Value |
|---|---|
| Type | Semantic |
| Interactive | Yes |
| Gate | No |
| Produces | `LatentModel`, primary outcome, candidate treatments |

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | Pipeline request | User's research question in natural language |

## Process

1. Run a single LLM conversation with `validate_latent_model(structure_json)`.
2. Parse a valid latent model containing constructs, causal edges, an outcome, and candidate treatments.
3. Give the LLM a self-review pass to refine the proposal.

This stage is purely theoretical and does not inspect the dataset.

## Outputs

| Output | Type | Description |
|---|---|---|
| `latent_model` | `LatentModel` | DAG with constructs and edges |
| `outcome_name` | `str` | Primary outcome construct |
| `treatments` | `list[str]` | Candidate treatment variables |
| `llm_trace` | `LLMTrace?` | Conversation trace |

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `Construct` | `{name, description, role, is_outcome, temporal_status}` | Theoretical variable in the latent model |
| `CausalEdge` | `{cause, effect, description, lagged}` | `lagged=True` means the effect at time `t` depends on the cause at `t-1` |

Unobserved confounding is modeled as explicit latent nodes in the DAG. ADMGs are only used internally for the y0 identification algorithm.

## Related Docs

- [../concepts/scope-and-timescales.md](../concepts/scope-and-timescales.md)
- [../concepts/assumptions.md](../concepts/assumptions.md)
- [../concepts/artifact-glossary.md](../concepts/artifact-glossary.md)
