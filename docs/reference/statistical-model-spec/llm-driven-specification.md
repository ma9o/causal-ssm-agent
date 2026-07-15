# LLM-Driven Model-Spec Specification

This page defines the exact LLM-facing control semantics of the [`statistical_model_spec` transition](../../pipeline/statistical-model-spec.md). For the reducer and recovery overview, see the [construct-admission state machine](state-machine.md). For allowed observation models, see [likelihoods](likelihoods.md). For parameter roles and prior families, see [parameters and priors](parameters.md).

## Entry Conditions

The transition consumes exact current versions of these artifacts.

| Input | Role |
|---|---|
| `question` | Substantive estimand context for prior reasoning |
| `causal_design` | Retained constructs, indicators, edges, confounders, and model clock |
| `identification_report` | Identification findings already derived from the causal design |
| `panel` | Encoded longitudinal observations used by admission checks |
| `validation_report` | Indicator empirical profiles and data-quality findings |

The selected versions are recorded as input pins in every checkpoint and in the final artifact provenance.

## Deterministic Planning

Before the first LLM turn, code:

1. derives the compiler-authoritative parameter catalog;
2. condenses estimation-graph strongly connected components into a deterministic admission DAG;
3. determines the parameters and indicators owned by each construct;
4. detects cycle-closing edge parameters that must be authored with the closing construct; and
5. creates or resumes an immutable checkpoint lineage.

The LLM cannot add constructs, indicators, causal edges, or arbitrary parameter names.

## One Construct per Concurrent LLM Subroutine

Every ready singleton unit opens a fresh LLM subroutine concurrently. Feedback-component members open one at a time in retained-state order. Each attempt still contains exactly one user message for one construct. The prompt contains:

| Prompt section | Content |
|---|---|
| Active construct | Construct identity, causal parents, and position in its admission unit |
| Indicators | Allowed families and links plus empirical profiles |
| Parameter surface | Canonical parameters that may receive priors in this submission |
| Accepted context | Previously admitted constructs relevant to the cumulative model |
| Validation feedback | Findings from the preceding attempt, when present |
| Guidance | Prior shapes, scales, dynamics semantics, and soft-check acceptance rules |

The active subroutine exposes `submit_construct` and, when configured with an Exa key, `search_literature`.

## `search_literature`

`search_literature` accepts a query and the exact parameter name it informs. Results are cached by query for the current checkpoint lineage. Search does not advance the branch.

## `submit_construct`

Every attempt must call `submit_construct` with this conceptual payload.

| Field | Meaning |
|---|---|
| `construct` | Branch construct; submissions outside the ready frontier are rejected |
| `indicators` | Distribution, link, and reasoning for each active indicator |
| `priors` | Canonical parameter name to distribution, parameters, and reasoning |
| `accept` | Optional list of `{check, target, rationale}` objects naming current soft failures exactly |

The tool call is terminal for that attempt. A non-admitted result causes the workflow to open another fresh attempt, up to the configured four-attempt limit.

## Structural Validation Before Simulation

The submission is rejected before prior-predictive simulation when:

- its construct is not the branch's assigned ready construct;
- it names a parameter outside the active compiler-authoritative surface;
- a cycle-closing edge has no linear or Hill prior in the closing submission;
- pooled parameters assign different prior families to one compiler sample site;
- an observation family or link is incompatible with the data; or
- the cumulative partial model does not compile.

Linear edges are declared by `beta_{parent}_{construct}` priors. Saturating edges are declared by the corresponding `hill_emax`, `hill_ec50`, and `hill_n` priors. A self-limiting well is declared by `self_limit_{construct}`.

## Exact Admission Battery

The proposed contribution is appended temporarily to its causal-ancestor closure from the immutable master snapshot on which the branch started. Code restricts the causal design to that cumulative partial model, compiles it, and simulates prior-predictive trajectories through the true nonlinear drift and observation densities. Unrelated accepted constructs are excluded, so branch completion timing cannot change the model being validated.

The battery evaluates applicable checks including:

| Check family | Question |
|---|---|
| Numerical health and confinement | Does the exact solve remain finite, and do trajectories avoid sustained growth? |
| Marginal latent scale | Does the across-draw latent distribution respect the standardized-scale convention? |
| Replicated-data location and dispersion | Do family-appropriate summaries of the observed data lie inside prior-replicate envelopes? |
| Resolvability | Does sufficient prior timescale mass remain visible through this construct's actual irregular gaps and span? |
| Per-edge influence | Does one parent dominate the child's temporal variation when disabled under the same Brownian path? |
| Hill activation | Do paired draws spend meaningful time on the bend of the actual nonnegative Hill response? |
| Transmission | Does the noise-free expected response move meaningfully relative to the sampled predictive response? |

Hard failures cannot be overridden. Monte Carlo prior-data discrepancies are soft. A soft failure can be accepted only by naming both its check and target and providing a rationale; one indicator or edge cannot implicitly accept another. Accepted rationales become annotations in the accumulated model state.

## Success Transaction

An admitted submission advances through one idempotent success transaction:

1. Build the accepted contribution and admission report.
2. Write an immutable child checkpoint containing the branch's accepted addition.
3. Write the submission result that references that checkpoint.
4. Return the result to the LLM child workflow.
5. Let the parent Stage 4 workflow gather the next completed branch batch.
6. Merge those child checkpoints once, in deterministic construct order, into a new master checkpoint and schedule newly ready descendants immediately.

The tool-request identifier determines each branch checkpoint filename. If execution stops after step 2, an activity retry finds that checkpoint and reconstructs the same result. It does not rerun the state mutation. A completed child may come from an earlier master snapshot; the merge requires its inherited state to be an unchanged subset of the current master and requires exactly one new construct. Search-cache conflicts fail instead of choosing a racing writer.

Rejected submissions create attempt results and traces but do not create success checkpoints.

## Checkpoint Contract

| Field | Meaning |
|---|---|
| `schema_version` | Exact checkpoint schema; unsupported schemas fail explicitly |
| `workspace_id` | Owning workspace |
| `run_id`, `seq` | Stage 4 run identity and episode sequence |
| `checkpoint_index` | Monotonic accepted-success index within the run |
| `parent_ref` | Previous checkpoint, including cross-run resume ancestry |
| `input_pins` | Exact artifact versions against which this state was accepted |
| `accepted_constructs` | Ordered semantic submissions, outcomes, and annotations |
| `search_queries`, `search_cache` | Literature-search state |
| `repair_feedback` | Target-scoped feedback from the full-model barrier |
| `full_model_validated` | Whether the complete shared-model barrier passed |
| `rebase` | Resume source and retained/reopened summary for a new run |

Checkpoint JSON lives in the run sidecar tree. The checkpoint excludes the panel, compiler runtime, and other reconstructible objects.

## Failure Contract

On terminal failure, Stage 4 emits a failed progress event and raises a typed Temporal application failure. The episode workflow records it as a `raised` transition with diagnostics containing:

| Diagnostic | Meaning |
|---|---|
| `transition_id` | `statistical_model_spec` |
| `construct` | Construct active when the failure occurred, if any |
| `checkpoint_ref` | Latest immutable accepted checkpoint |
| Original diagnostics | Compile or validation details supplied by the failing activity |

The public artifact state does not advance. The outer orchestrator can inspect the timeline and progress events, revise upstream artifacts through normal moves, and then propose `run(statistical_model_spec)` again.

## Resume with Unchanged Inputs

When the new input pins equal the source checkpoint pins, code reconstructs `AdmissionState` directly from accepted semantic contributions. Previously accepted batteries are not rerun. The next attempts start from every currently ready unit.

## Rebase with Changed Inputs

When an upstream artifact changed, saved work becomes a proposal to revalidate rather than trusted accepted state.

Code rebuilds the catalog and replays saved contributions when their predecessor units remain valid. Each contribution passes through the same compiler and exact admission battery. Rebase invalidates:

- a construct missing from its new admission unit;
- a contribution that no longer compiles;
- a contribution with a hard admission failure; or
- a contribution with an unresolved soft finding.

Each invalid unit and its descendants reopen. The new run writes checkpoint zero with the remaining dependency-closed set and a `rebase` summary. Independent saved branches remain accepted because they did not depend on the invalidated state.

## Finalization

After all constructs are admitted, a deterministic barrier compiles the complete model once, draws one exact prior-predictive sample set, and runs every construct's battery against that shared state. Any failed unit and its descendants reopen for another admission cycle. Finalization runs only after the barrier passes:

1. reconstructs the complete accepted state from the latest checkpoint;
2. materializes `StatisticalModelSpec` and authored priors;
3. merges per-attempt LLM traces;
4. writes the versioned `statistical_model_spec` artifact with the run's input pins; and
5. invokes the normal derivation cascade that produces `compiled_ssm`.

No checkpoint is treated as a public model artifact, and no downstream inference can start from partial accepted state.
