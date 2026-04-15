# Stage 4: Model Specification and Prior Elicitation

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | Yes | `ModelSpec`, `PriorProposal` per parameter |

Translates the [Stage 1b `CausalSpec`](01b-measurement-identifiability.md#causalspec) into a fully specified statistical model by choosing observation-model distributions for ambiguous indicators and eliciting Bayesian priors for every parameter, validated against prior predictive checks.

For the high-level reducer flow, see [Stage 4 State Machine](../reference/model-spec/state-machine.md). For the exact control semantics of the LLM-driven loop, see [LLM-Driven Stage 4 Specification](../reference/model-spec/llm-driven-specification.md).

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | User | Original research question, used to justify prior reasoning |
| `causal_spec` | [Stage 1b](01b-measurement-identifiability.md) | [`CausalSpec`](01b-measurement-identifiability.md#causalspec) with constructs, edges, indicators, and `model_clock` |
| `data_for_model` | [Stage 2](02-indicator-extraction.md) | Encoded long-format [`ObservationRecord`](02-indicator-extraction.md#observationrecord) table |
| `indicator_audits` | [Stage 3](03-extraction-validation.md) | Per-indicator [`EmpiricalProfile`](03-extraction-validation.md#empiricalprofile)s and validation summaries |
| `enable_literature` | Pipeline config | Whether the `search_literature` tool is offered to the LLM |

Stage 4 is the first point where the pipeline reasons about statistical model form. Earlier stages defined what to measure and how.

## Process

Stage 4 makes decisions incrementally: each block scopes the LLM to one choice, and accepted blocks stay frozen unless a validator reopens them.

```mermaid
flowchart LR
    S[Skeleton] --> BF[Frontier\nFormation] --> MD[Active Model-\nDecision Block] --> CV{Validation}
    CV -- "fail" --> MD
    CV -- "next" --> MD
    CV -- "spec\nlocked" --> PB[Active Prior\nElicitation Block]
    PB --> BV{Validation}
    BV -- "fail" --> PB
    BV -- "next" --> PB
    BV -- "priors\naccepted" --> GV{PPCs}
    GV -- ok --> F([ModelSpec + Priors])
    GV -- fail --> RR{Failure\nClassifier}
    RR -- "model issue" --> MD
    RR -- "prior issue" --> PB
```

**Skeleton:** Before any LLM judgment, a deterministic engine enumerates [parameters](../reference/model-spec/parameters.md), locks [likelihoods](../reference/model-spec/likelihoods.md#dtype-to-distribution-mapping) where the dtype maps to exactly one distribution, fixes loading orientations from Stage 1b indicator polarity, and fixes temporal structure (AR(1) dynamics, factor-analysis loadings with scale identification[^bollen1989], multi-resolution aggregation). Indicators where the dtype admits multiple distributions or links are deferred to the LLM.

**Frontier Formation:** The skeleton produces *model-decision blocks* (one per ambiguous indicator) and *prior blocks* in dependency order: measurement → dynamics → grouped causal-effect families (incoming effects per target construct) → confounding.

**Model-Decision Block:** Each block resolves:

- *Distribution and link* for one ambiguous indicator, informed by its [Stage 3](03-extraction-validation.md) empirical profile and domain semantics

Model-decision blocks are validated locally against the active frontier and accepted into reducer state one block at a time. Once all model-decision blocks are accepted, the stage materializes the full `ModelSpec` and runs a [compilation check](../reference/compilation.md) with PPCs disabled; compile failures reopen the smallest responsible model-decision block. Before prior elicitation, a compact global-review checkpoint can reopen the relevant model-decision blocks when those choices need to move together. Loading orientations remain visible at that checkpoint but are already fixed from Stage 1b indicator polarity rather than authored blockwise in Stage 4.

**Prior Elicitation Block:** Once the `ModelSpec` is locked, the LLM proposes a full prior specification for each block in dependency order: distribution family, hyperparameters, and reasoning. All priors are specified on the discrete-time scale at the model clock interval; [compilation](../reference/compilation.md) converts them to continuous-time rates where needed.

When enabled, the LLM can query [Exa](https://exa.ai/) for empirical studies to inform prior calibration, justifying narrower priors only when the estimand, population, and timescale align[^gelman2020] [^gelman2013]. Optionally, multiple paraphrased calls for one parameter can be aggregated via a Gaussian mixture model[^capstick2024] to reduce prompt-wording bias.

**Validation:** The stage validates in two layers. After the model-decision phase closes, the full locked `ModelSpec` is compiled once, enforcing distribution-link and dtype compatibility, loading-matrix rank, and successful SSM construction. During prior elicitation, each accepted prior block is merged into the accumulated authored priors, but real prior compilation and prior predictive checks only run once the full required prior set is present. At that point, a global prior predictive simulation checks:

- *Numerical health*: no NaN/Inf or extreme values (|value| > 10⁶)
- *Constraint satisfaction*: positive-constrained parameters must not violate their support
- *Dynamics stability*: the drift matrix must have strictly negative real eigenvalues under a majority of prior draws[^sarkka2019]
- *Scale plausibility*: the implied observation SD from the stationary covariance[^sarkka2019] must be within a reasonable ratio of the [Stage 3](03-extraction-validation.md) empirical SD[^gelman2020] [^riegler2025]

If validation fails, a deterministic classifier reopens the smallest responsible block.

### Example

For a study of classroom engagement and academic performance where Stage 1b posited constructs `Teacher Feedback Frequency`, `Student Engagement`, and `Test Scores` with model clock `1w`, Stage 4 might: resolve `Test Scores` deterministically to `gaussian`/`identity` in the skeleton; present one model-decision block for `Teacher Feedback Frequency` where the LLM chooses `poisson`/`log`; then process prior blocks in order — the dynamics block for `Student Engagement` yields `rho_engagement ~ Beta(5, 2)` reflecting moderate weekly persistence, and the causal-effect block for the feedback→engagement edge yields `beta_teacher_feedback_engagement ~ Normal(0.2, 0.15)` anchored by an educational psychology meta-analysis.

## Outputs

| Output | Type | Description |
|---|---|---|
| `model_spec` | `ModelSpec` | Complete statistical model specification |
| `_compiled_ssm` | [`CompiledSSMArtifact`](../reference/compilation.md) | Serializable compiled model consumed by [Stage 5a](05a-svi-preflight.md); contains the flat `SSMSpec`, `edge_lag_days`, compiled prior semantics, parameter bindings, and compile diagnostics |

### ModelSpec.LikelihoodSpec

| Field | Type | Description |
|---|---|---|
| `variable` | `str` | Name of the observed indicator |
| `distribution` | [`DistributionFamily`](../reference/model-spec/likelihoods.md#distribution-families) | Observation-model distribution family |
| `link` | [`LinkFunction`](../reference/model-spec/likelihoods.md#link-functions) | Link function mapping latent state to distribution parameter |
| `centered` | `bool` | Deterministic auto-centering flag for additive-location indicators that are centered before fitting |

### ModelSpec.ParameterSpec

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Parameter name such as `beta_stress_anxiety`, `rho_mood`, or `sigma_sleep` |
| `role` | [`ParameterRole`](../reference/model-spec/parameters.md#parameter-roles) | Role in the model |
| `constraint` | [`ParameterConstraint`](../reference/model-spec/parameters.md#parameter-roles) | Domain constraint |
| `description` | `str` | Human-readable description |

### ModelSpec

| Field | Type | Description |
|---|---|---|
| `likelihoods` | `list[LikelihoodSpec]` | One likelihood row per retained manifest indicator |
| `parameters` | `list[ParameterSpec]` | Compiler-authoritative semantic prior surfaces that remain active after model decisions are locked |
| `initialization_policy` | `\"stationary\" \| \"free\"` | Whether dynamic-state initial conditions are stationary-derived or exposed as free `t0_*` surfaces |
| `equilibrium_forcing` | `bool` | Whether eligible centered dynamic constructs may expose a continuous-time intercept `cint_*` |

[^gelman2020]: Gelman, A., Vehtari, A., Simpson, D., et al. (2020). Bayesian Workflow. arXiv:2011.01808. [Bibliography entry](../reference/bibliography.md)
[^gelman2013]: Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Bibliography entry](../reference/bibliography.md)
[^bollen1989]: Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley. [Bibliography entry](../reference/bibliography.md)
[^sarkka2019]: Särkkä, S., & Solin, A. (2019). *Applied Stochastic Differential Equations*. Cambridge University Press. [Bibliography entry](../reference/bibliography.md)
[^capstick2024]: Capstick, A., Krishnan, R. G., & Barnaghi, P. (2024). AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling. arXiv:2411.17284. [Bibliography entry](../reference/bibliography.md)
[^riegler2025]: Riegler, M. A., Hellton, K. H., Thambawita, V., & Hammer, H. L. (2025). Using Large Language Models to Suggest Informative Prior Distributions in Bayesian Regression Analysis. *Scientific Reports*, 15, 33386. [Bibliography entry](../reference/bibliography.md)
