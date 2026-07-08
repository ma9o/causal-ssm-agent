# LLM-Driven Model-Spec Specification

This page is an exact start-to-finish walkthrough of [`statistical_model_spec` transition](../../pipeline/statistical-model-spec.md) as it is currently implemented. The goal is to make one run traceable in order: what enters the transition, what the deterministic code computes before the LLM speaks, what each promptable block can and cannot do, when validation runs, how repairs are localized, and what completion means.

For the `statistical_model_spec` transition artifact contract, see [`statistical_model_spec` transition](../../pipeline/statistical-model-spec.md). For the high-level reducer mental model, see [`statistical_model_spec` transition State Machine](state-machine.md). For the allowed observation-model vocabulary, see [likelihoods](likelihoods.md). For the parameter roles and prior-family vocabulary, see [parameters and priors](parameters.md).

## 1. Entry Conditions

`statistical_model_spec` transition starts after the causal graph, measurement structure, extracted observations, and indicator audits already exist.

| Input | Source | Exact role in `statistical_model_spec` transition |
|---|---|---|
| `question` | User | Grounds distribution and prior reasoning in the actual substantive question. |
| `causal_design` | [`measurement_structure` transition](../../pipeline/measurement-structure.md#causaldesign) | Supplies the retained constructs, indicators, estimation edges, explicit latent confounders, and `model_clock`. |
| `data_for_model` | [`measurements` transition](../../pipeline/extraction.md) | Supplies the realized observation table used by prior-predictive validation. |
| `indicator_audits` | [`validation_report` derivation](../../pipeline/extraction-validation.md) | Supplies empirical profile and validation summaries used in distribution cards, construct-scale cards, and scale plausibility checks. |

By the time `statistical_model_spec` transition begins, the pipeline is no longer deciding the causal DAG or the indicator set. `statistical_model_spec` transition receives that structure as fixed input and translates it into an executable statistical specification plus authored priors.

## 2. Deterministic Initialization Happens Before Any LLM Turn

The first thing `statistical_model_spec` transition does is build a deterministic skeleton from the `causal_design`.

The skeleton has four exact outputs.

| Skeleton output | Meaning |
|---|---|
| `resolved_likelihoods` | Indicators whose dtype admits exactly one valid distribution and exactly one valid link. These are locked before the LLM speaks. |
| `ambiguous_indicators` | Indicators whose dtype admits multiple valid likelihood/link choices, or one fixed family with multiple valid links. These become `indicator_decision` blocks. |
| `parameters` | Non-loading semantic parameters exposed by the compiler-backed `statistical_model_spec` transition inventory. |
| `loading_params` | Loading parameters for non-reference indicators in multi-indicator constructs, with polarity-derived constraints already fixed. |

The skeleton is exact, not heuristic, in the following sense.

| What is fixed deterministically | How it is fixed |
|---|---|
| Unique likelihood choices | By the [dtype-to-distribution mapping](likelihoods.md#dtype-to-distribution-mapping) plus the valid links for each distribution. |
| Loading orientations | From `measurement_structure` transition indicator polarity. `statistical_model_spec` transition does not open a separate loading-orientation decision surface. |
| Parameter inventory | By combining seeded semantic metadata with the compiler-authoritative prior inventory. `statistical_model_spec` transition errors if the public inventory drifts from compiler-exposed parameters. |
| Structural ordering | By the retained estimation-state order, retained directed edges, and induced dependencies from explicit latent confounders. |

In the current implementation, the compiler-authoritative `statistical_model_spec` transition parameter inventory can include these roles when applicable:

| Role |
|---|
| `ar_coefficient` |
| `fixed_effect` |
| `residual_sd` |
| `state_intercept` |
| `observation_intercept` |
| `initial_state_mean` |
| `initial_state_sd` |
| `static_state_sd` |
| `loading` |
| `measurement_error_sd` |
| `observation_hyperparameter` |
| `observation_hyperparameter_positive` |
| `correlation` |

That is the first key boundary of `statistical_model_spec` transition: the LLM does not invent the parameter set. The code does.

## 3. The Skeleton Is Converted into a Fixed Execution Plan

After the skeleton is built, `statistical_model_spec` transition converts it into a fixed execution plan. That plan fixes the exact promptable blocks and their deterministic order.

The planned blocks are:

| Planned block | When it is created |
|---|---|
| `model:configuration` | Always present as the first model-decision block. |
| `indicator:{variable}` | Once per ambiguous indicator in the skeleton. |
| `review:statistical_model_spec` | Always present as the compact model-form review checkpoint. |
| `measurement:{construct}` | Once per construct with one or more non-reference loadings. |
| `observation:{parameter}` | Once per active observation intercept or observation-family hyperparameter surface. |
| `dynamics:{scc}` | Once per strongly connected latent subsystem that owns dynamics-role parameters. |
| `effects:{target}` | Once per target construct with one or more fixed-effect parameters. |
| `correlation:{parameter}` | Once per innovation-correlation or compiled baseline-factor scale surface. |
| `review:prior_system` | Always planned, but initialized as inactive and used only if repair routing escalates to whole-system prior review. |

The exact nominal order is:

1. `model:configuration`,
2. all `indicator_decision` blocks,
3. `review:statistical_model_spec`,
4. all `measurement_prior` blocks in construct order,
5. all `observation_prior` blocks in deterministic parameter order,
6. all `dynamics_prior` blocks in strongly connected component order,
7. all `effect_prior` blocks in target-construct order,
8. all `correlation_prior` blocks,
9. `review:prior_system` only if a repair route activates it.

Two implementation details matter here.

| Detail | Exact behavior |
|---|---|
| `review:prior_system` on the happy path | It is in the plan, but `statistical_model_spec` transition does not visit it on the straight-through path. If all prior blocks validate cleanly, the transition goes directly to `done`. |
| No ambiguous indicators | `statistical_model_spec` transition still visits `model:configuration`; after that block is accepted, it can proceed straight to `StatisticalModelSpec` locking without any `indicator_decision` turns. |

## 4. Runtime State Is Explicit and Finite

`statistical_model_spec` transition is not an open-ended conversation. The reducer moves through a small explicit runtime state.

### 4.1 Cursor Kinds

| Cursor kind | Meaning |
|---|---|
| `block` | The transition is waiting for an LLM submission for one concrete prompt block. |
| `statistical_model_spec_lock` | All model-decision blocks are accepted and the reducer is about to materialize or validate the locked `StatisticalModelSpec`. |
| `repair_barrier` | A multi-block repair campaign has finished its block edits and now requires joint validation of the repaired scope. |
| `done` | `statistical_model_spec` transition has a completed accepted result. |

### 4.2 Block Status Values

| Status | Meaning |
|---|---|
| `pending` | Planned but not yet accepted. |
| `accepted` | Accepted and currently frozen. |
| `reopened` | Previously accepted or pending, then reopened by deterministic repair routing. |
| `inactive` | Planned but not currently reachable on the nominal path, used initially for `review:prior_system`. |

### 4.3 Accepted State

The reducer stores accepted state separately from the current cursor.

| Accepted-state field | Meaning |
|---|---|
| `statistical_model_spec` | The locked accepted model form, once materialized. |
| `authored_priors` | The merged accepted prior proposals accumulated across prior blocks. |
| `validation` | The latest accepted validation payload. |
| `distribution_choices` | Pre-lock accepted ambiguous-indicator likelihood choices. |

This separation is why `statistical_model_spec` transition can reopen one scope without regenerating the whole transition.

## 5. Every Outer Turn Has One Required Contract

For every promptable block, `statistical_model_spec` transition renders a prompt and then requires the model turn to end with that block's primary submit tool. If the turn ends without the required submit tool, the transition errors.

Only some tools are allowed in each block family.

| Block family | Tools allowed in that turn |
|---|---|
| `model_configuration` | `submit_model_configuration` |
| `indicator_decision` | `submit_indicator_choice` |
| `global_review` | `submit_model_review` |
| `measurement_prior` | `submit_prior_block`, `elicit_prior_gmm` when paraphrasing is enabled |
| `observation_prior` | `submit_prior_block`, `elicit_prior_gmm` when paraphrasing is enabled |
| `dynamics_prior` | `submit_prior_block`, `elicit_prior_gmm` when paraphrasing is enabled |
| `effect_prior` | `submit_prior_block`, `search_literature` when enabled, `elicit_prior_gmm` when paraphrasing is enabled |
| `correlation_prior` | `submit_prior_block`, `elicit_prior_gmm` when paraphrasing is enabled |
| `global_prior_review` | `submit_prior_block` |

Only the block's primary submit tool advances reducer state. Auxiliary tools can help the LLM think, but they do not themselves accept or reopen anything.

## 6. Model-Decision Phase: Exact Semantics

The first promptable phase is the model-decision phase.

In the current implementation, the LLM-owned model-form decision surface is exactly this:

| Open model-form decision | Owned by |
|---|---|
| `initialization_policy` | LLM |
| `observation_intercept_policy` | LLM |
| `equilibrium_forcing` | LLM |
| Ambiguous indicator distribution and link | LLM |
| Auto-centering eligibility and `centered` tags | Deterministic skeleton plus observation semantics |
| Loading orientations | Deterministic skeleton |
| Parameter inventory | Deterministic skeleton |

The `model_configuration` block is always first. It owns only the three model-level decisions:

| Model-configuration decision | Meaning |
|---|---|
| `initialization_policy="stationary"` | Dynamic-state initial conditions are derived from the stationary residual process; only retained time-invariant states can keep free `t0_*` surfaces. |
| `initialization_policy="free"` | Active `t0_mean_*` and `t0_sd_*` surfaces remain free and must be prior-authored. |
| `observation_intercept_policy="free"` | Eligible manifest intercepts remain free `manifest_mean_*` surfaces and must be prior-authored. |
| `observation_intercept_policy="fixed"` | Eligible manifest intercepts are fixed rather than exposed as free `manifest_mean_*` surfaces. |
| `equilibrium_forcing=false` | No `cint_*` surfaces remain active. |
| `equilibrium_forcing=true` | `cint_*` surfaces can remain active only for dynamic constructs identified by centered additive-location indicators. |

For an `indicator_decision` block, the prompt is restricted to the active indicator and includes:

| Prompt-local section | Exact content |
|---|---|
| Filtered model topology | Only the constructs and latent edges relevant to the active block. |
| Distribution cards | The active indicator's admissible family/link options, `how_to_measure`, aggregation, effective window, empirical profile, and validation issues. |
| Construct-scale cards | The relevant construct-local scale context enriched with any already accepted likelihood choices. |
| Frontier status | Counts of accepted model and prior blocks, model-lock status, repair-campaign status, and block-local scope names. |
| Latest feedback | The last validator feedback for the current frontier. |

If the submission is accepted, `statistical_model_spec` transition stores the chosen distribution and link in `distribution_choices` and advances to the next pending `indicator_decision` block.

If the submission is rejected, the accepted state does not roll back globally. Only the active block remains unresolved or reopens.

## 7. Locking the `StatisticalModelSpec` Is a Separate Reducer Step

Once `model:configuration` and all `indicator_decision` blocks are accepted, `statistical_model_spec` transition does not immediately start authoring priors. It first enters the `statistical_model_spec_lock` step.

The lock step does exactly two things.

1. It builds a full `StatisticalModelSpec` by combining:
   the accepted `initialization_policy`, `observation_intercept_policy`, and `equilibrium_forcing`,
   the skeleton's deterministic likelihoods,
   the accepted ambiguous-indicator choices,
   deterministic `centered` tags derived from observation semantics,
   and the compiler-authoritative parameter inventory after conditional activation.
2. It validates that `StatisticalModelSpec` with a compile-only `statistical_model_spec` transition assembly check.

At this point `statistical_model_spec` transition is asking a narrow question: "Is the full model form executable?" It is not yet asking whether the prior system is plausible.

If the lock succeeds:

- the accepted `statistical_model_spec` is stored,
- the accepted validation payload is stored,
- and `statistical_model_spec` transition advances to `review:statistical_model_spec`.

If the lock fails:

- the reducer classifies the compile failure,
- reopens the smallest matching model-form scope,
- and returns to the corresponding `model_configuration` or `indicator_decision` block.

Compile routing is exact:

| Compile-failure case | Reopened scope |
|---|---|
| Compile feedback names a concrete parameter or indicator owner block | The matching local block |
| Active block is `global_prior_review` | `review:prior_system` |
| No narrower identifier match exists | The active block |

## 8. `global_review` Is a Real Checkpoint, Not a Summary Prompt

After the `StatisticalModelSpec` locks successfully, `statistical_model_spec` transition runs `review:statistical_model_spec`.

This block has only two legal outcomes.

| Review decision | Exact effect |
|---|---|
| `approve` | Mark `review:statistical_model_spec` accepted and advance to the first pending prior block. |
| `reopen` with `reopen_block_ids` | Reopen exactly those named `indicator_decision` blocks and route back to the first reopened block. |

This checkpoint exists because some indicator-level likelihood choices only become problematic when considered jointly. It does not create a new free-form decision surface. It can reopen named model-decision blocks, but it cannot invent new block types or rewrite unrelated accepted state.

## 9. Prior Phase: Exact Block Ownership and Order

If `global_review` approves the locked model form, `statistical_model_spec` transition enters the prior phase.

The exact prior-block families and what each owns are:

| Prior block family | Exact ownership |
|---|---|
| `measurement_prior` | The loading parameters for one construct. |
| `observation_prior` | One active observation intercept or observation-family auxiliary parameter surface, such as `manifest_mean_*`, `obs_df`, `obs_shape`, `obs_r`, or ordered/categorical auxiliary sites. |
| `dynamics_prior` | The subsystem's `ar_coefficient`, `residual_sd`, any active `cint_*`, and any exposed `initial_state_mean` and `initial_state_sd` parameters for the constructs in that subsystem. |
| `effect_prior` | All fixed-effect parameters whose target construct is the active target. |
| `correlation_prior` | One `correlation` or `static_state_sd` parameter. |

The current implementation orders them as follows:

1. all measurement blocks in retained construct order,
2. all observation blocks in deterministic parameter order,
3. all dynamics blocks in strongly connected component order,
4. all effect blocks in retained target-construct order,
5. all correlation blocks in deterministic parameter order.

That ordering is not cosmetic. It means the transition sees measurement scale before observation intercepts and family extras, then dynamics before incoming effect rows, and only then confounding covariance terms.

## 10. Prior Submission Semantics Are Incremental and Exact

A prior submission is always block-local. The submission may contain only priors whose parameter names belong to the active block.

When a prior submission arrives, `statistical_model_spec` transition performs these exact steps.

1. Schema-validate each submitted prior proposal.
2. Merge schema-valid priors into the current `authored_priors`.
3. Compute the required prior set from the locked `StatisticalModelSpec`.
4. Decide whether the transition is still in partial-prior mode or has enough priors for full validation.

The required prior set is exact:

| Parameter roles required to close `statistical_model_spec` transition | Parameter roles that are currently optional for closure |
|---|---|
| Everything except the roles in the right column | `initial_state_mean`, `initial_state_sd` |

This means `statistical_model_spec` transition can finish without authored priors for `initial_state_mean` and `initial_state_sd` even when those parameters are present in prompt cards. The prompts may still discuss them, but the transition does not require them to satisfy the "all required priors present" check.

## 11. Full Prior Compilation Does Not Start Immediately

One subtle but important implementation detail is that partial prior authoring does not immediately trigger full prior compilation.

While required priors are still missing, `statistical_model_spec` transition does this:

| Condition | Exact behavior |
|---|---|
| Some required priors are still missing | Validate only prior schema plus compilability of the locked `StatisticalModelSpec`; return feedback listing the remaining missing priors. |
| The active block is `dynamics_prior` or `effect_prior` and compile-only validation passed | Run a block-local partial drift guard even before the full prior set exists. |

So there are two distinct pre-completion modes:

| Mode | What is being checked |
|---|---|
| Partial-prior accumulation | "Are these priors schema-valid, and does the locked model form still exist?" |
| Partial drift guard for dynamics/effects | "Even before the full prior system is present, is this local dynamics or effect bundle already obviously unstable or budget-exhausting?" |

This is why `statistical_model_spec` transition can reject a dynamics or effect block before the whole prior table has been written, while still deferring full prior-predictive simulation until the required prior set exists.

## 12. Full Validation Starts Only After the Required Prior Set Exists

Once all required priors are present, `statistical_model_spec` transition switches from partial-prior mode to full `statistical_model_spec` transition assembly validation.

The validation sequence is exact:

1. compile the `StatisticalModelSpec` together with the authored priors,
2. collect compile diagnostics,
3. if `skip_ppc` is false and observation data are available, run prior-predictive validation using:
   the compiled model,
   `data_for_model`,
   and per-indicator scale summaries extracted from `indicator_audits`.

The full prior-predictive layer checks at least these failure families:

| Validation family | Exact purpose |
|---|---|
| Numerical pathologies | Reject NaN, Inf, and extreme simulated values. |
| Support compliance | Reject priors whose implied draws fall outside required support. |
| Dynamical stability | Reject prior systems whose implied dynamics are unstable. |
| Observation-scale plausibility | Compare implied observation scale against `validation_report` derivation empirical scale summaries. |

If validation succeeds, the accepted state is updated.

If validation fails:

- the failing submission does not overwrite the last accepted valid state,
- the validator feedback is preserved,
- and repair routing decides what must reopen.

This non-overwrite rule is exact and important: rejected compile attempts and rejected prior-predictive attempts do not become the new accepted state.

## 13. Repair Routing Uses a Deterministic Scope Ladder

When validation fails, `statistical_model_spec` transition does not let the LLM decide what to revisit. It computes repair scopes deterministically from the diagnostics.

### 13.1 Support Mismatches

If the diagnostics indicate likelihood support mismatch, `statistical_model_spec` transition immediately routes to a `likelihood_support` scope and reopens the responsible `indicator_decision` block when it can identify it.

### 13.2 Drift-Related Prior Failures

If the failure is drift-related, `statistical_model_spec` transition constructs an ordered candidate ladder:

| Scope kind | Scope rank | Meaning |
|---|---|---|
| `local_drift_motif` | 0 | The smallest local set of direct writer blocks and associated dynamics blocks suggested by the failing parameters. |
| `reciprocal_pair` | 1 | Expand the local motif to include reciprocal feedback edges when present. |
| `scc_drift_subsystem` | 2 | Expand to the full strongly connected subsystem. |

If the validator itself emitted a repair scope, `statistical_model_spec` transition also considers:

| Scope kind | Scope rank | Meaning |
|---|---|---|
| `validator_scope` | 2 | A validator-owned structural scope, currently used for dynamics-SCC repairs. |

### 13.3 Non-Drift Local Failures

If the failure is not drift-related but `statistical_model_spec` transition can identify the directly responsible parameters, it uses:

| Scope kind | Scope rank | Meaning |
|---|---|---|
| `direct_writer_blocks` | 0 | Reopen the prompt blocks that directly own the failing parameters. |

### 13.4 Global Failures

If the diagnostics indicate a global failure, `statistical_model_spec` transition adds:

| Scope kind | Scope rank | Meaning |
|---|---|---|
| `global_prior_review` | 3 | Reopen the whole prior system through `review:prior_system`. |

The ladder is monotone. `statistical_model_spec` transition will retry the same scope only up to two attempts and only while the pathology certificate is improving or unavailable. Otherwise it escalates to the next wider scope.

## 14. Multi-Block Repairs Become Repair Campaigns

Some repair scopes map to more than one prompt block. When that happens, `statistical_model_spec` transition opens a repair campaign.

The exact campaign rule is:

| Repair plan size | Barrier behavior |
|---|---|
| One prompt block | No barrier is required. |
| More than one prompt block | `requires_barrier_validation=True`, so the transition must stop at a repair barrier and revalidate the repaired scope jointly. |

The runtime behavior is:

1. mark the relevant blocks `reopened`,
2. preserve accepted state outside the reopened scope,
3. repair the campaign blocks in deterministic plan order,
4. when all blocks in the campaign are accepted, move to the `repair_barrier` cursor,
5. re-run joint validation over the repaired scope,
6. if the barrier passes, clear the campaign and continue,
7. if the barrier fails, classify again and possibly escalate to a wider scope.

This is the exact mechanism that prevents `statistical_model_spec` transition from accepting a set of locally plausible edits that only fail when assembled jointly.

## 15. `review:prior_system` Is an Escalation Endpoint

`review:prior_system` is not a nominal final step. It is an escalation endpoint for global prior-system repair.

When `statistical_model_spec` transition activates it:

| Property | Exact behavior |
|---|---|
| Scope | It may revise priors across the whole system. |
| Model-form authority | It may not change locked likelihood choices or loading orientations. |
| Tools | Only `submit_prior_block` is allowed. |
| Compile failure route | If compile fails while this block is active, compile repair routes back to `review:prior_system` itself. |

So `review:prior_system` is the widest prior-authoring scope, but it is still prior-only. It does not reopen the model-form surface directly.

## 16. Exact Completion Criteria

`statistical_model_spec` transition is complete only when the reducer reaches `done` and has a valid accepted result.

The reducer-level completion condition is:

| Condition | Must hold |
|---|---|
| Cursor | `done` |
| Accepted `statistical_model_spec` | Present |
| Accepted `authored_priors` | Non-empty |

Operationally, a clean completion also implies:

- the required prior set is complete,
- no repair campaign remains active,
- and the last accepted validation is not a compile failure or prior-predictive failure.

## 17. Outputs at Completion

There are two useful views of completion.

### 17.1 Reducer Result

When the agentic `statistical_model_spec` transition session completes, its direct result contains:

| Field | Meaning |
|---|---|
| `statistical_model_spec` | Final accepted model form. |
| `authored_priors` | Final accepted authored prior proposals. |
| `search_queries` | Any recorded literature-search queries used during the transition. |
| `validation` | Final accepted `statistical_model_spec` transition validation payload. |

### 17.2 Materialized Transition Output

When the Prefect wrapper materializes the final `statistical_model_spec` transition artifact, it adds derived outputs:

| Field | Meaning |
|---|---|
| `statistical_model_spec` | Final accepted model form. |
| `authored_priors` | Final accepted authored prior proposals. |
| `resolved_priors` | Compiler-resolved prior semantics for downstream runtime use. |
| `search_queries` | Recorded literature-search queries, if any. |
| `validation_warnings` | Any non-fatal compile or prior-predictive warnings. |
| `prior_predictive_samples` | Per-manifest prior-predictive samples for the web payload when available. |
| `_compiled_ssm` | Executable compiled state-space artifact for downstream transitions. |

That is the end of the `statistical_model_spec` transition. The transition has converted upstream causal and measurement structure into an accepted executable model form plus an accepted authored prior system.

## 18. What `statistical_model_spec` transition Does Not Decide

The boundary around `statistical_model_spec` transition is exact and deliberate.

| Not decided in `statistical_model_spec` transition | Where it is decided |
|---|---|
| Which constructs, indicators, and causal edges belong in the model | [`latent_structure` transition](../../pipeline/latent-structure.md) and [`measurement_structure` transition](../../pipeline/measurement-structure.md) |
| Whether a causal estimand is identified from the graph | [`measurement_structure` transition](../../pipeline/measurement-structure.md) |
| Posterior fitting and post-fit diagnostics | [`posterior` transition](../../pipeline/inference.md) |
| Intervention ranking and interactive causal analysis | [`baseline_report` transition](../../pipeline/analysis.md) |

The exact role of `statistical_model_spec` transition is narrower: it is the controlled functional-specification transition that turns a fixed upstream causal and measurement problem into a locked `StatisticalModelSpec`, an accepted prior system, and a compiled executable artifact for downstream fitting.
