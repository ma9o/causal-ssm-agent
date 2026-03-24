# Stage 1a: Latent Model Proposal

| Type | Interactive | Gate | Produces |
|---|---|---|---|
| llm | Yes | No | [`Latent Model`](#latent-model) |

Maps a natural-language research question to a theoretical causal topological structure over latent constructs. See [../concepts/causal-modeling-terminology.md](../concepts/causal-modeling-terminology.md) for the distinction.

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | User | User's research question in natural language |

## Process

The proposal must satisfy the [construct ontology and edge rules](../primitives/latent-model/constructs-and-edges.md), the [temporal semantics](../primitives/latent-model/temporal-semantics.md), and the [latent-model assumptions](../primitives/latent-model/assumptions.md) that govern valid causal topological structure at this stage.

1. Run one LLM conversation with `validate_latent_model(structure_json)`.
2. Ask the model to propose a candidate latent model: a construct-level causal topological structure with directed edges and exactly one designated outcome.
3. Validate the candidate for schema correctness and causal-graph constraints; if validation fails, let the model revise it in the same conversation until a valid latent model is obtained.
4. Persist the validated latent model as the authoritative Stage 1a artifact. Downstream stages may derive the designated outcome and candidate intervention variables from that graph when needed.

Stage 1a is purely theoretical and does not inspect the dataset. It defines the construct-level causal topological structure, not indicators, observed columns, support-window semantics, identifiability findings, or functional specification.

## Outputs

| Output | Type | Description |
|---|---|---|
| `latent_model` | `LatentModel` | Theoretical causal topological structure over latent constructs |

The public stage payload exposes that artifact directly. It may also include `llm_trace` as runtime provenance for the UI.

## Definitions

### Latent Model

The `LatentModel` is the theoretical causal topological structure over latent constructs proposed before any measurement choices are made. It owns:

- the construct set
- the directed causal edges between constructs
- the outcome designation encoded on the outcome construct

Each construct carries `name`, `description`, `role`, `is_outcome`, and `temporal_status`. Each causal edge carries `cause`, `effect`, `description`, and `lagged`, where `lagged=true` means the effect at time `t` depends on the cause at `t-1`.

The designated outcome is encoded on the outcome construct via `is_outcome=true`. Candidate intervention variables are derived from the validated graph rather than stored as separate Stage 1a state.

Example: for a question about whether staffing pressure affects patient deterioration through care delays, Stage 1a may posit constructs such as `Staffing Pressure`, `Care Delay`, `Patient Severity`, and `Patient Deterioration`, plus directed edges between them and an explicit latent confounder node if an unobserved common cause is believed to exist.
