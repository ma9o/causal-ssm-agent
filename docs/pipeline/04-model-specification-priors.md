# Stage 4: Model Specification and Prior Elicitation

Runs a multi-turn agentic conversation to choose the statistical model and elicit priors grounded in data profiles and optional literature. This is the point where the pipeline enters the downstream model-runtime path: see [../model-runtime/functional-specification.md](../model-runtime/functional-specification.md) for the rules and [../model-runtime/compilation.md](../model-runtime/compilation.md) for how the result becomes executable.

## At a Glance

| Property | Value |
|---|---|
| Type | Semantic |
| Interactive | Yes |
| Gate | No |
| Produces | [`ModelSpec`](../concepts/artifact-glossary.md), priors, prior predictive samples |

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | Pipeline request | Research question |
| `stage1b.result` | Stage 1b | `CausalSpec` with latent and measurement models |
| `stage2.result` | Stage 2 | Model-ready observation data |
| `stage3.result` | Stage 3 | Indicator audits and empirical profiles |
| `enable_literature` | Pipeline config | Whether literature search is available |

## Process

1. Build decision cards from the `CausalSpec` and empirical profiles.
2. Run a multi-turn conversation with:
   - `validate_model(model_json)`
   - `search_literature(query, parameter_name)`
   - `elicit_prior_gmm(...)` when paraphrased elicitation is enabled
3. Validate schema shape, trial compilation, and prior predictive behavior before finalizing.

The full functional-specification deep dive lives in [../model-runtime/functional-specification.md](../model-runtime/functional-specification.md). The handoff from Stage 4 into compilation is summarized in [../model-runtime/handoff-map.md](../model-runtime/handoff-map.md).

## Outputs

| Output | Type | Description |
|---|---|---|
| `model_spec` | `ModelSpec` | Full functional specification |
| `priors` | `dict[str, PriorProposal]` | Prior distribution per parameter |
| `search_queries` | `dict[str, str]?` | Literature searches used during elicitation |
| `prior_predictive_samples` | `dict[str, list[float]]?` | Simulated samples from the prior |
| `llm_trace` | `LLMTrace?` | Conversation trace |

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `ModelSpec` | `{parameters: list[ParameterSpec], likelihoods: list[LikelihoodSpec]}` | Top-level Stage 4 output |
| `ParameterSpec` | `{name, role, constraint, description}` | Parameter definition in the compiled model |
| `LikelihoodSpec` | `{variable, distribution, link, reasoning, sources}` | Observation model choice per variable |
| `PriorProposal` | `{parameter, distribution, params, sources, reasoning, reference_interval_days, density_points}` | Supports DT-to-CT translation of prior information |
