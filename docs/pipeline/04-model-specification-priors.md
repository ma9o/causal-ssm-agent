# Stage 4: Model Specification and Prior Elicitation

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | Yes | [`ModelSpec`](#modelspec), [`PriorProposal`](#priorproposal) per parameter |

Translates the [Stage 1b `CausalSpec`](01b-measurement-identifiability.md#causalspec) into a fully specified statistical model by choosing observation-model distributions for ambiguous indicators and eliciting Bayesian priors for every parameter, validated against [prior predictive checks](#prior-predictive-validation). The resulting [`ModelSpec`](#modelspec) plus priors are then consumed by the [SSM compilation pipeline](../reference/compilation.md) to build an executable NumPyro model.

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | User | Original research question, used to justify prior reasoning |
| `stage1b.result` | [Stage 1b](01b-measurement-identifiability.md) | `CausalSpec` with latent model, measurement model, and identifiability status |
| `stage2.result` | [Stage 2](02-indicator-extraction.md) | Encoded long-format `ObservationRecord` table persisted from Stage 2 |
| `stage3.result` | [Stage 3](03-extraction-validation.md) | Per-indicator empirical profiles and validation audits |
| `enable_literature` | Pipeline config | Whether the `search_literature` tool is offered to the LLM |

Stage 4 is the first point where the pipeline reasons about statistical model form. Earlier stages defined what to measure and how; this stage decides what distributions and priors govern the generative model.

## Process

Stage 4 runs one multi-turn LLM conversation that bridges causal structure and statistical specification. The conversation has two phases, model specification decisions followed by prior elicitation, both grounded by a unified `validate_model` tool. An optional literature search tool provides empirical evidence for effect sizes.

**Deterministic skeleton pre-computation.** Before the LLM conversation begins, a deterministic engine derives everything that follows from the `CausalSpec` without statistical judgment:

- *Parameter enumeration*: one AR coefficient (`rho`) per endogenous time-varying construct, one fixed effect (`beta`) per causal edge, one residual SD (`sigma`) per construct, one loading (`lambda`) per non-reference indicator in multi-indicator constructs, one static-state SD (`tau`) per time-invariant endogenous construct when needed, and one correlation (`cor`) per pair of constructs whose shared confounder was marginalized at identifiability time.
- *Deterministic likelihoods*: where an indicator's `measurement_dtype` maps to exactly one valid distribution and link, the likelihood is locked without LLM input.
- *Ambiguous indicators*: where the dtype admits multiple valid distributions or links, a decision card is generated for the LLM.
- *Prompt context*: the engine assembles model-topology cards, distribution decision cards, construct-scale cards, and per-parameter prior cards. Likelihood-family names, link names, and prior-family names are exact canonical strings; aliases are rejected at validation.

**Phase 1: Model specification decisions.** The LLM reviews the decision cards and submits two sets of choices via a `validate_model` tool call:

- *Distribution and link* for each ambiguous indicator, selected from the valid options shown on the decision card and informed by the indicator's empirical profile and domain semantics
- *Loading constraint* for each loading parameter: `positive` for sign identification or `none` if negative loadings are theoretically plausible

The tool merges these decisions with the pre-computed skeleton to produce a complete `ModelSpec`, then runs schema validation and a trial compilation against the [SSM compiler](../reference/compilation.md). On failure the tool returns specific errors; the LLM revises and resubmits within the same conversation until the model spec is accepted.

**Phase 2: Prior elicitation.** With the model spec locked, the LLM proposes priors for every parameter in small batches. Each prior specifies a distribution family, its parameters, reasoning, literature sources, and optionally a `reference_interval_days` when the evidence comes from a study with a different observation interval than the model clock. The `distribution` field must use the exact canonical `PriorDistributionFamily` names documented in [Supported Prior Distribution Families](../reference/model-spec/prior-distribution-families.md). On each submission the tool validates prior schemas, performs a real compilation with the proposed priors, and runs [prior predictive checks](#prior-predictive-validation). If prior predictive simulation reveals implausible implied data, the tool returns per-parameter feedback with suggested adjustments; the LLM revises the flagged priors and resubmits.

The `validate_model` tool is stateful: it retains accepted model decisions and valid priors across calls. It enforces protocol constraints, including no mixed decision and prior submissions, no redundant resubmissions of already accepted state, and batch size limits on priors. After any rejection, the LLM resubmits only the changed fields.

**Literature search.** When `enable_literature` is true, the LLM has access to a `search_literature` tool that queries [Exa](https://exa.ai/) for empirical studies on effect sizes. The tool is used selectively for key causal effect parameters where domain knowledge is uncertain. Each search is captured as provenance on the stage output. Literature evidence anchors priors on meta-analyses or large longitudinal studies; heterogeneous evidence widens priors.

**Paraphrased elicitation (optional).** When configured, the LLM can call `elicit_prior_gmm` for a single parameter, which runs multiple paraphrased LLM calls and aggregates them through either simple pooling or a Gaussian mixture model. This follows the AutoElicit-style strategy from Capstick et al. (2024) and is intended to reduce brittle overconfidence from any one prompt wording. Default behavior keeps this disabled for cost reasons.

## Deterministic Guardrails

The following rules are stage-owned semantics of `ModelSpec`, not auxiliary reference material.

### Link Functions from Indicator Dtype

| `measurement_dtype` | Default distribution | Link | Alternatives |
|---|---|---|---|
| `continuous` | `gaussian` | `identity` | `student_t`, `gamma` (`log` or `inverse`), `beta` (`logit` or `probit`) |
| `binary` | `bernoulli` | `logit` | `bernoulli` with `probit` |
| `count` | `poisson` | `log` | `negative_binomial` (`log`) |
| `ordinal` | `ordered_logistic` | `cumulative_logit` | None |
| `categorical` | `categorical` | `softmax` | `ordered_logistic` (`cumulative_logit`) when categories are substantively ordered |

The default distribution is selected automatically from `measurement_dtype`. Alternative distributions for the same dtype can be specified explicitly through per-indicator `LikelihoodSpec` entries in `ModelSpec`.

### Temporal and Measurement Structure

- Endogenous time-varying constructs receive AR(1) under the Stage 1a Markov commitment.
- Single-indicator constructs fix `lambda = 1`; multi-indicator constructs use factor-analysis structure with the first or reference loading fixed for identification.
- When cause and effect operate at different granularities, finer-to-coarser effects are aggregated with the indicator's declared operator, while coarser-to-finer values are broadcast across the governed finer timepoints.

### Parameter Roles

| Role | Symbol | Meaning | Appears in |
|---|---|---|---|
| `ar_coefficient` | `rho` | Autoregressive persistence of a latent state | Drift diagonal |
| `fixed_effect` | `beta` | Cross-lag causal effect between constructs | Drift off-diagonal |
| `residual_sd` | `sigma` | Innovation process scale | Diffusion diagonal |
| `static_state_sd` | `tau` | Quasi-constant latent-state variation | Static-state block |
| `loading` | `lambda` | Factor loading mapping latent to observed | Measurement model |
| `correlation` | `cor` | Off-diagonal residual correlation between latent innovations | Diffusion covariance |

### Parameter Constraints

| Constraint | Domain | Typical owners |
|---|---|---|
| `none` | `(-inf, +inf)` | Fixed effects |
| `positive` | `(0, +inf)` | Residual SDs, static-state SDs, some loadings |
| `unit_interval` | `[0, 1]` | AR coefficients on the discrete-time persistence scale |
| `correlation` | `[-1, 1]` | Residual correlations |

### Role-to-Constraint Mapping

| Role | Default constraint | Rationale |
|---|---|---|
| `ar_coefficient` | `unit_interval` | Stage 4 elicits discrete-time persistence magnitude and compilation later converts it to continuous-time drift |
| `fixed_effect` | `none` | Causal effects can be positive or negative |
| `residual_sd` | `positive` | Standard deviations are non-negative |
| `static_state_sd` | `positive` | Static-state scales are non-negative |
| `loading` | `positive` or `none` | Stage 4 may enforce sign identification while allowing substantively negative loadings when justified |
| `correlation` | `correlation` | Correlations are bounded by definition |

Typical prior-family guidance by constraint lives in [Supported Prior Distribution Families](../reference/model-spec/prior-distribution-families.md).

### Prior Predictive Validation

Prior predictive checks run automatically as part of the `validate_model` tool whenever real priors and Stage 2 observation data are both available. The validation path:

1. Samples parameters from their proposed prior distributions
2. Builds and simulates the compiled generative model under those draws
3. Checks for compile failures, non-finite or unstable simulations, and broad scale mismatches against Stage 3 empirical profiles

Failures surface as per-parameter or global feedback identifying whether the issue is a model-spec problem, for example an incompatible likelihood or build failure, or a prior problem, for example implausible implied scale. The LLM iterates until the checks pass or the conversation exhausts its turn budget.

## Outputs

| Output | Type | Description |
|---|---|---|
| `model_spec` | [`ModelSpec`](#modelspec) | Complete statistical model specification |
| `authored_priors` | `dict[str, PriorProposal]` | LLM-authored prior per parameter, keyed by parameter name |
| `resolved_priors` | `list[PriorProposal]` | Canonical public prior rows after compiler resolution, including DT-to-CT adjustments and implicit defaults exposed by compilation |

The public stage payload also includes `search_queries` for literature provenance, `prior_predictive_samples` for the UI, and `llm_trace` as runtime provenance.

## Definitions

### ModelSpec

`ModelSpec` is the functional specification emitted by Stage 4. It contains:

- the [likelihood specifications](#likelihoodspec), one per observed indicator
- the [parameter specifications](#parameterspec), one per free parameter

The downstream [SSM compiler](../reference/compilation.md) consumes `ModelSpec` together with priors to produce an executable NumPyro model.

### LikelihoodSpec

`LikelihoodSpec` specifies the observation model for one indicator variable:

| Field | Type | Description |
|---|---|---|
| `variable` | `str` | Name of the observed indicator |
| `distribution` | `DistributionFamily` | Distribution family such as `gaussian`, `student_t`, `poisson`, `bernoulli`, or `ordered_logistic` |
| `link` | `LinkFunction` | Link function such as `identity`, `log`, `logit`, `probit`, `cumulative_logit`, or `softmax` |
| `reasoning` | `str` | Why this distribution and link were chosen |
| `sources` | `list[LikelihoodSource]` | Optional literature evidence |

### ParameterSpec

`ParameterSpec` defines one free parameter in the model:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Parameter name such as `beta_stress_anxiety`, `rho_mood`, or `sigma_sleep` |
| `role` | `ParameterRole` | Role in the model: `ar_coefficient`, `fixed_effect`, `residual_sd`, `loading`, `correlation`, or `static_state_sd` |
| `constraint` | `ParameterConstraint` | Domain constraint: `none`, `positive`, `unit_interval`, or `correlation` |
| `description` | `str` | Human-readable description |

Loading constraints are the one case where the LLM chooses between `positive` and `none`. All other constraints are determined mechanically by role.

### PriorProposal

`PriorProposal` is the prior distribution proposed for one parameter:

| Field | Type | Description |
|---|---|---|
| `parameter` | `str` | Name of the parameter this prior is for |
| `distribution` | `PriorDistributionFamily` | Prior family from [Supported Prior Distribution Families](../reference/model-spec/prior-distribution-families.md) |
| `params` | `dict[str, float]` | Distribution parameters such as `{"mu": 0.3, "sigma": 0.15}` |
| `sources` | `list[PriorSource]` | Literature evidence supporting this prior |
| `reasoning` | `str` | Justification for the prior |
| `reference_interval_days` | `float` \| `null` | Observation interval the prior is expressed in when it differs from the model clock |
| `density_points` | `list[{x, y}]` \| `null` | Pre-computed density curve for frontend visualization |

`PriorDistributionFamily` is a separate vocabulary from `DistributionFamily`: likelihood families describe observation noise, while prior families describe parameter uncertainty. The exact canonical prior names are owned by [Supported Prior Distribution Families](../reference/model-spec/prior-distribution-families.md).

Both fixed-effect and AR priors are specified on the discrete-time scale at the model clock interval. Compilation later converts them to continuous-time rates where needed.

### Prior-Elicitation References

- Capstick et al. (2024). *AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling.* arXiv: [2411.17284](https://arxiv.org/abs/2411.17284)
- Chen et al. (2025). *LLM-BI: Towards Fully Automated Bayesian Inference with Large Language Models.* arXiv: [2508.08300](https://arxiv.org/abs/2508.08300)
- Huang (2025). *LLM-Prior: A Framework for Knowledge-Driven Prior Elicitation and Aggregation.* arXiv: [2508.03766](https://arxiv.org/abs/2508.03766)
- Riegler et al. (2025). *Using large language models to suggest informative prior distributions in Bayesian regression analysis.* *Scientific Reports*. DOI: [10.1038/s41598-025-18425-9](https://www.nature.com/articles/s41598-025-18425-9)
- Selby et al. (2024). *Had Enough of Experts? Elicitation and Evaluation of Bayesian Priors from Large Language Models.* NeurIPS BDU Workshop

Example: for a study of classroom engagement and academic performance where Stage 1b posited constructs `Teacher Feedback Frequency`, `Student Engagement`, and `Test Scores` with model clock `1w`, Stage 4 might resolve `Test Scores` deterministically to `gaussian` with `identity`, choose `poisson` with `log` for `Teacher Feedback Frequency`, set `beta_teacher_feedback_engagement` prior to `Normal(0.2, 0.15)` based on an educational psychology meta-analysis, and set `rho_engagement` prior to `Beta(5, 2)` reflecting moderate weekly persistence of engagement.
