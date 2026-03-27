# Stage 4: Model Specification and Prior Elicitation

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | Yes | `ModelSpec`, `PriorProposal` per parameter |

Translates the [Stage 1b `CausalSpec`](01b-measurement-identifiability.md#causalspec) into a fully specified statistical model by choosing observation-model distributions for ambiguous indicators and eliciting Bayesian priors for every parameter, validated against prior predictive checks.

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

A **frontier reducer** drives Stage 4 by processing a queue of decision blocks, each scoped to the minimum context the LLM needs for one incremental choice. Blocks are formed deterministically from the skeleton and processed one at a time, while accepted blocks stay frozen unless a downstream validator explicitly reopens them.

```mermaid
flowchart LR
    S[Skeleton] --> BF[Block\nFormation] --> MD[Active Model-\nDecision Block] --> CV{Compile\nCheck}
    CV -- "reopen block" --> MD
    CV -- "next model block" --> MD
    CV -- "model spec locked" --> PB[Active Prior\nBlock]
    PB --> BV{Block\nValidation}
    BV -- "reopen block" --> PB
    BV -- "next prior block" --> PB
    BV -- "all prior blocks accepted" --> GV{Global\nPPC}
    GV -- ok --> F([ModelSpec + Priors])
    GV -- fail --> RR{Reopen\nRouter}
    RR -- "model issue" --> MD
    RR -- "prior issue" --> PB
```

### Skeleton

Before any LLM judgment, a deterministic engine derives everything that follows mechanically from the `CausalSpec`:

- *Parameter enumeration*: one parameter per structural element in the `CausalSpec`; [roles, scoping rules, and constraints](../reference/model-spec/parameters.md) are defined in the reference
- *Deterministic likelihoods*: where an indicator's `measurement_dtype` maps to exactly one valid distribution and link per the [dtype-to-distribution mapping](../reference/model-spec/likelihoods.md#dtype-to-distribution-mapping), the likelihood is locked without LLM input
- *Ambiguous indicators*: where the dtype admits multiple valid distributions or links, the choice is deferred to the LLM

Temporal and measurement structure fixed by the skeleton:

- Endogenous time-varying constructs receive AR(1) dynamics under the [Stage 1a](01a-latent-model.md) Markov commitment
- Single-indicator constructs fix λ = 1; multi-indicator constructs use factor-analysis structure with the first or reference loading fixed for scale identification[^bollen1989]
- When cause and effect operate at different granularities, finer-to-coarser effects are aggregated with the indicator's declared operator; coarser-to-finer values are broadcast across governed finer timepoints

### Block formation

The skeleton produces two classes of decision blocks:

- *Model-decision blocks*: one per ambiguous indicator (distribution/link choice) and one per construct with loading parameters (constraint choice)
- *Prior blocks*, in dependency order:
  - *Measurement*: one per multi-indicator construct — loading priors, given indicator scales and the locked `ModelSpec`
  - *Dynamics*: one per construct — AR coefficient and residual SD priors, given construct scales and model clock
  - *Causal effects*: one per fixed effect or tightly coupled effect family — informed by the full model topology, construct scales, and accepted dynamics priors
  - *Confounding*: one per induced-dependency component — correlation priors, given accepted dynamics and effect priors

Each block carries only its local parameter cards, the relevant [Stage 3](03-extraction-validation.md) empirical profiles, and accepted upstream decisions needed for compatibility.

### Model-decision blocks

The frontier reducer presents one model-decision block at a time. A block resolves either:

- *Distribution and link* for one ambiguous indicator, informed by its [Stage 3](03-extraction-validation.md) empirical profile and domain semantics; or
- *Loading constraint* for a construct's loading parameters: `positive` for sign identification, or `none` if negative loadings are theoretically plausible

Accepted decisions are frozen and merged into the growing `ModelSpec`. The [compilation check](#compilation) gates each block with PPCs disabled; errors reopen only the failing block.

### Prior-block preparation

Once the `ModelSpec` is locked, deterministic code prepares the prior frontier without taking ownership of prior-family selection:

- Orders prior blocks by dependency
- Assembles block-local parameter cards and empirical-profile context
- Applies mechanical time-scale normalization and interval translation after the LLM proposes a prior
- Enforces schema and support constraints during validation

The LLM still proposes the full prior specification for each active block: distribution family, hyperparameters, and reasoning, optionally informed by literature evidence.

### Prior blocks

The frontier reducer presents one prior block at a time in dependency order. For each block the LLM sees only the local parameter cards, relevant empirical profiles, and accepted upstream decisions, and proposes full prior specifications only for that block's parameters.

*Literature search:* When enabled, the LLM can query [Exa](https://exa.ai/) for empirical studies that inform prior calibration. Evidence synthesis such as meta-analyses or closely matched longitudinal studies can justify narrower priors only when the estimand, population, and timescale align; otherwise the safer default is a weaker prior checked by prior predictive simulation[^gelman2020], following the standard Bayesian workflow[^gelman2013].

*Paraphrased elicitation (optional):* To reduce overconfidence from any single prompt wording, the pipeline can run multiple paraphrased LLM calls for a parameter and aggregate via a Gaussian mixture model, following the AutoElicit strategy[^capstick2024]. An alternative aggregation approach uses logarithmic opinion pooling across independently elicited priors[^huang2025]. Disabled by default for cost reasons.

All priors are specified on the discrete-time scale at the model clock interval; [compilation](../reference/compilation.md) converts them to continuous-time rates where needed.

### Validation

Two tiers of checks run at different points in the frontier.

#### Compilation

The [SSM compiler](../reference/compilation.md) runs after each model-decision block and after each prior block. It enforces:

- Distribution–link compatibility: the chosen link must be valid for the chosen distribution family
- Dtype–distribution compatibility: the chosen distribution must be valid for the indicator's [`measurement_dtype`](01b-measurement-identifiability.md)
- Loading-matrix rank: the number of observed indicators must be at least the number of latent constructs
- Full SSM construction: the complete state-space model must build without error

#### Global prior predictive simulation

After all prior blocks are accepted, the validator samples from the proposed priors, simulates from the compiled generative model, and checks:

- *Numerical health*: no NaN/Inf values in simulated sites; no extreme values (|value| > 10⁶)
- *Constraint satisfaction*: positive-constrained parameters (diffusion, observation variance, initial-state variance) must not violate their support
- *Dynamics stability*: the drift matrix must have strictly negative real eigenvalues under a majority of prior draws, ensuring stationary dynamics for the linear SDE[^sarkka2019]
- *Scale plausibility*: the implied observation standard deviation — derived analytically from the stationary covariance via the Lyapunov equation[^sarkka2019] — must be within a reasonable ratio of the empirical standard deviation from [Stage 3](03-extraction-validation.md) profiles. This is a prior-predictive scale-calibration check in the sense of Gelman et al. (2020)[^gelman2020], aimed at catching prior-predictive over-dispersion or under-dispersion; LLM-elicited priors appear especially vulnerable to width miscalibration, showing tendencies toward both overconfidence and underconfidence[^riegler2025].

#### Reopen router

If global validation fails, a deterministic classifier maps each failure to the smallest responsible block:

- Distribution–link or dtype mismatch → the indicator's model-decision block
- Measurement-scale issue → the construct's measurement prior block
- Drift stability or interval issue → the relevant dynamics or effect block
- Prior support violation → the affected parameter's block
- Diffuse global instability → the smallest connected dynamics component

The reopen router may target either a model-decision block or a prior block. The frontier reducer never falls back to showing the full parameter inventory unless diagnostics genuinely cannot be localized.

### Example

For a study of classroom engagement and academic performance where Stage 1b posited constructs `Teacher Feedback Frequency`, `Student Engagement`, and `Test Scores` with model clock `1w`, Stage 4 might: resolve `Test Scores` deterministically to `gaussian`/`identity` in the skeleton; present one model-decision block for `Teacher Feedback Frequency` where the LLM chooses `poisson`/`log`; then process prior blocks in order — the dynamics block for `Student Engagement` yields `rho_engagement ~ Beta(5, 2)` reflecting moderate weekly persistence, and the causal-effect block for the feedback→engagement edge yields `beta_teacher_feedback_engagement ~ Normal(0.2, 0.15)` anchored by an educational psychology meta-analysis.

## Outputs

| Output | Type | Description |
|---|---|---|
| `model_spec` | `ModelSpec` | Complete statistical model specification |
| `_compiled_ssm` | [`CompiledSSMArtifact`](../reference/compilation.md) | Serializable compiled model consumed by [Stage 5a](05a-svi-preflight.md); contains the `SSMSpec`, compiled prior semantics, and parameter bindings |

### ModelSpec.LikelihoodSpec

| Field | Type | Description |
|---|---|---|
| `variable` | `str` | Name of the observed indicator |
| `distribution` | [`DistributionFamily`](../reference/model-spec/likelihoods.md#distribution-families) | Observation-model distribution family |
| `link` | [`LinkFunction`](../reference/model-spec/likelihoods.md#link-functions) | Link function mapping latent state to distribution parameter |

### ModelSpec.ParameterSpec

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Parameter name such as `beta_stress_anxiety`, `rho_mood`, or `sigma_sleep` |
| `role` | [`ParameterRole`](../reference/model-spec/parameters.md#parameter-roles) | Role in the model |
| `constraint` | [`ParameterConstraint`](../reference/model-spec/parameters.md#parameter-roles) | Domain constraint |
| `description` | `str` | Human-readable description |

[^gelman2020]: Gelman, A., Vehtari, A., Simpson, D., et al. (2020). Bayesian Workflow. arXiv:2011.01808. [Bibliography entry](../reference/bibliography.md)
[^gelman2013]: Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Bibliography entry](../reference/bibliography.md)
[^bollen1989]: Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley. [Bibliography entry](../reference/bibliography.md)
[^sarkka2019]: Särkkä, S., & Solin, A. (2019). *Applied Stochastic Differential Equations*. Cambridge University Press. [Bibliography entry](../reference/bibliography.md)
[^huang2025]: Huang, Y. (2025). LLM-Prior: A Framework for Knowledge-Driven Prior Elicitation and Aggregation. arXiv:2508.03766. [Bibliography entry](../reference/bibliography.md)
[^capstick2024]: Capstick, A., Krishnan, R. G., & Barnaghi, P. (2024). AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling. arXiv:2411.17284. [Bibliography entry](../reference/bibliography.md)
[^riegler2025]: Riegler, M. A., Hellton, K. H., Thambawita, V., & Hammer, H. L. (2025). Using Large Language Models to Suggest Informative Prior Distributions in Bayesian Regression Analysis. *Scientific Reports*, 15, 33386. [Bibliography entry](../reference/bibliography.md)
