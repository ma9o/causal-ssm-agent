# Stage 4: Model Specification and Prior Elicitation

| Modality | Interactive | Gate | Produces |
|---|---|---|---|
| Semantic | Yes | No | [`ModelSpec`](#modelspec), [`PriorProposal`](#priorproposal) per parameter |

Translates the [Stage 1b `CausalSpec`](01b-measurement-identifiability.md#causalspec) into a fully specified statistical model by choosing observation-model distributions for ambiguous indicators and eliciting Bayesian priors for every parameter, validated against [prior predictive checks](#prior-predictive-validation). The resulting [`ModelSpec`](#modelspec) plus priors are then consumed by the [SSM compilation pipeline](../reference/compilation.md) to build an executable NumPyro model.

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | User | Original research question—anchors prior reasoning |
| `stage1b.result` | [Stage 1b](01b-measurement-identifiability.md) | `CausalSpec` with latent model, measurement model, and identifiability status |
| `stage2.result` | [Stage 2](02-indicator-extraction.md) | Model-ready long-format observation table persisted from Stage 2 |
| `stage3.result` | [Stage 3](03-extraction-validation.md) | Per-indicator empirical profiles and validation audits |
| `enable_literature` | Pipeline config | Whether the `search_literature` tool is offered to the LLM |

Stage 4 is the first point where the pipeline reasons about statistical model form. Earlier stages defined *what* to measure and *how*; this stage decides *what distributions and priors* govern the generative model.

## Process

Stage 4 runs a single multi-turn LLM conversation that bridges causal structure and statistical specification. The conversation has two phases—model specification decisions followed by prior elicitation—both grounded by a unified `validate_model` tool. An optional literature search tool provides empirical evidence for effect sizes.

**Deterministic skeleton pre-computation.** Before the LLM conversation begins, a deterministic engine derives everything that follows from the `CausalSpec` without statistical judgment:

- *Parameter enumeration*: one AR coefficient (`rho`) per endogenous time-varying construct, one fixed effect (`beta`) per causal edge, one residual SD (`sigma`) per construct, one loading (`lambda`) per non-reference indicator in multi-indicator constructs, and one correlation (`cor`) per pair of constructs whose shared confounder was [marginalized at identifiability time](01b-measurement-identifiability.md#identifiabilitystatus).
- *Deterministic likelihoods*: where an indicator's `measurement_dtype` maps to exactly one valid distribution and link (e.g. `binary` → `bernoulli`/`logit`), the likelihood is locked without LLM input.
- *Ambiguous indicators*: where the dtype admits multiple valid distributions or links (e.g. `continuous` can be `gaussian`, `student_t`, `gamma`, or `beta`), a decision card is generated for the LLM.
- *Prompt context*: the engine assembles model-topology cards, distribution decision cards, construct scale cards (with empirical profiles from Stage 3), and per-parameter prior cards. These give the LLM the fixed structural context and the exact decision surface. Likelihood-family names, link names, and prior-family names are exact canonical strings; aliases are rejected at validation.

The deterministic rules for parameter roles, constraints, and likelihood mappings are defined in [parameters-likelihoods-and-priors.md](../reference/model-spec/parameters-likelihoods-and-priors.md).

**Phase 1: Model specification decisions.** The LLM reviews the decision cards and submits two sets of choices via a `validate_model` tool call:

- *Distribution + link* for each ambiguous indicator, selected from the valid options shown on the decision card and informed by the indicator's empirical profile and domain semantics.
- *Loading constraint* for each loading parameter: `positive` for sign identification (the default when the reference indicator and the non-reference indicator should co-vary positively) or `none` if negative loadings are theoretically plausible.

The tool merges these decisions with the pre-computed skeleton to produce a complete `ModelSpec`, then runs schema validation and a trial compilation against the [SSM compiler](../reference/compilation.md). On failure the tool returns specific errors; the LLM revises and resubmits within the same conversation until the model spec is accepted.

**Phase 2: Prior elicitation.** With the model spec locked, the LLM proposes priors for every parameter in small batches (at most 8 per `validate_model` call). Each prior specifies a [distribution family](#priorproposal), its parameters, reasoning, literature sources, and optionally a `reference_interval_days` when the evidence comes from a study with a different observation interval than the model clock. The `distribution` field must use the exact canonical `PriorDistributionFamily` names documented in [Supported Prior Distribution Families](../reference/model-spec/prior-distribution-families.md). On each submission the tool validates prior schemas, performs a real compilation with the proposed priors, and runs [prior predictive checks](#prior-predictive-validation). If prior predictive simulation reveals implausible implied data, the tool returns per-parameter feedback with suggested adjustments; the LLM revises the flagged priors and resubmits.

The `validate_model` tool is **stateful**: it retains accepted model decisions and valid priors across calls. It enforces protocol constraints—no mixed decision+prior submissions, no redundant resubmissions of already-accepted state, and batch size limits on priors. After any rejection, the LLM resubmits only the changed fields.

**Literature search.** When `enable_literature` is true, the LLM has access to a `search_literature` tool that queries [Exa](https://exa.ai/) for empirical studies on effect sizes. The tool is used selectively for key causal effect parameters where domain knowledge is uncertain. Each search is captured as provenance on the stage output. Literature evidence anchors priors on meta-analyses or large longitudinal studies; heterogeneous evidence widens priors.

**Paraphrased elicitation (optional).** When configured, the LLM can call `elicit_prior_gmm` for a single parameter, which runs *N* paraphrased LLM calls and aggregates them via a Gaussian mixture model following [Capstick et al. (2024)](../reference/model-spec/parameters-likelihoods-and-priors.md#references). This reduces brittle overconfidence from any one prompt wording. Default behavior keeps this disabled for cost reasons.

### Prior Predictive Validation

Prior predictive checks run automatically as part of the `validate_model` tool whenever real priors and model-ready data are both available. The validation path:

1. Samples parameters from their proposed prior distributions
2. Builds and simulates the compiled generative model under those draws
3. Checks for compile failures, non-finite or unstable simulations, and broad scale mismatches against Stage 3 empirical profiles

Failures surface as per-parameter or global feedback identifying whether the issue is a model-spec problem (for example, an incompatible likelihood or build failure) or a prior problem (for example, implausible implied scale). The LLM iterates until the checks pass or the conversation exhausts its turn budget.

## Outputs

| Output | Type | Description |
|---|---|---|
| `model_spec` | [`ModelSpec`](#modelspec) | Complete statistical model specification |
| `authored_priors` | `dict[str, PriorProposal]` | LLM-authored prior per parameter, keyed by parameter name |
| `resolved_priors` | `list[PriorProposal]` | Canonical public prior rows after compiler resolution, including DT→CT adjustments and implicit defaults exposed by compilation |

The public stage payload also includes `search_queries` (literature provenance), `prior_predictive_samples` (per-variable simulated samples for the UI), and `llm_trace` as runtime provenance.

## Definitions

### ModelSpec

`ModelSpec` is the functional specification of the statistical model proposed by Stage 4. It owns:

- the [likelihood specifications](#likelihoodspec)—one per observed indicator, each binding a distribution family, link function, and reasoning
- the [parameter specifications](#parameterspec)—one per free parameter, each carrying a name, role, constraint, and description

Later stages should treat `ModelSpec` as the authoritative answer to "what is the statistical model we are fitting?" The downstream [SSM compiler](../reference/compilation.md) consumes it together with priors to produce an executable NumPyro model. The [compilation pipeline](../reference/compilation.md) describes how each field maps to compilation inputs.

### LikelihoodSpec

`LikelihoodSpec` specifies the observation model for one indicator variable:

| Field | Type | Description |
|---|---|---|
| `variable` | `str` | Name of the observed indicator |
| `distribution` | `DistributionFamily` | Distribution family (`gaussian`, `student_t`, `gamma`, `beta`, `poisson`, `negative_binomial`, `bernoulli`, `ordered_logistic`, `categorical`) |
| `link` | `LinkFunction` | Link function mapping the linear predictor to the distribution mean (`identity`, `log`, `inverse`, `logit`, `probit`, `cumulative_logit`, `softmax`) |
| `reasoning` | `str` | Why this distribution/link was chosen |
| `sources` | `list[LikelihoodSource]` | Optional literature evidence |

### ParameterSpec

`ParameterSpec` defines one free parameter in the model:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Parameter name (e.g. `beta_stress_anxiety`, `rho_mood`, `sigma_sleep`) |
| `role` | `ParameterRole` | Role in the model: `ar_coefficient`, `fixed_effect`, `residual_sd`, `loading`, `correlation`, or `static_state_sd` |
| `constraint` | `ParameterConstraint` | Domain constraint: `none` (unconstrained), `positive`, `unit_interval`, or `correlation` ([-1, 1]) |
| `description` | `str` | Human-readable description |

Constraints are determined by role: AR coefficients are `unit_interval`, residual SDs are `positive`, fixed effects are `none`, correlations are `correlation`. Loading constraints are the one case where the LLM decides between `positive` and `none`.

### PriorProposal

`PriorProposal` is the prior distribution proposed for one parameter. It owns:

| Field | Type | Description |
|---|---|---|
| `parameter` | `str` | Name of the parameter this prior is for |
| `distribution` | `PriorDistributionFamily` | Prior family from [Supported Prior Distribution Families](../reference/model-spec/prior-distribution-families.md) |
| `params` | `dict[str, float]` | Distribution parameters (e.g. `{"mu": 0.3, "sigma": 0.15}`) |
| `sources` | `list[PriorSource]` | Literature evidence supporting this prior |
| `reasoning` | `str` | Justification for the prior |
| `reference_interval_days` | `float?` | Observation interval (in days) the prior is expressed in, when it differs from the model clock. Used for DT→CT conversion of dynamic parameters |
| `density_points` | `list[{x, y}]?` | Pre-computed density curve for frontend visualization |

`PriorDistributionFamily` is a separate vocabulary from [`DistributionFamily`](#likelihoodspec): likelihood families describe observation noise, while prior families describe parameter uncertainty. The supported prior set is defined in [Supported Prior Distribution Families](../reference/model-spec/prior-distribution-families.md).

Both fixed-effect and AR priors are specified on the **discrete-time scale** at the model clock interval. The pipeline automatically converts them to continuous-time rates during compilation.

Example: for a study of classroom engagement and academic performance where Stage 1b posited constructs `Teacher Feedback Frequency`, `Student Engagement`, and `Test Scores` with model clock `1w`, Stage 4 might resolve `Test Scores` (continuous) deterministically to `gaussian`/`identity`, choose `poisson`/`log` for `Teacher Feedback Frequency` (count, low variance-to-mean ratio), set `beta_teacher_feedback_engagement` prior to `Normal(0.2, 0.15)` based on an educational psychology meta-analysis (with `reference_interval_days: 30` because the cited study measured monthly), and set `rho_engagement` prior to `Beta(5, 2)` reflecting moderate weekly persistence of engagement.
