/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python Pydantic models via:
 *   cd apps/data-pipeline && uv run python scripts/export_schemas.py
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/causal_ssm_agent/flows/stages/contracts.py
 */

/**
 * Whether a variable is modeled or treated as given.
 */
export type Role = "endogenous" | "exogenous";
/**
 * Whether a construct changes within-person over time.
 */
export type TemporalStatus = "time_varying" | "time_invariant";
/**
 * Direction of an indicator relative to its construct.
 */
export type IndicatorPolarity = "positive" | "negative";
/**
 * Whether the measurement equation is point-local or interval-summary.
 */
export type SupportKind = "point" | "interval";
/**
 * Supported summary operators for indicator observations.
 */
export type SummaryOperator = "first" | "last" | "sum" | "count" | "mean" | "std";
/**
 * Which support boundary receives the observation anchor.
 */
export type AnchorPolicy = "support_start" | "support_end";
/**
 * Distribution families for observation and process noise.
 */
export type DistributionFamily =
  | "gaussian"
  | "student_t"
  | "poisson"
  | "gamma"
  | "bernoulli"
  | "negative_binomial"
  | "beta"
  | "ordered_logistic"
  | "categorical";
/**
 * Link functions mapping linear predictor to distribution mean.
 */
export type LinkFunction = "identity" | "log" | "inverse" | "logit" | "probit" | "cumulative_logit" | "softmax";
/**
 * Role of a parameter in the model.
 */
export type ParameterRole =
  | "fixed_effect"
  | "ar_coefficient"
  | "residual_sd"
  | "state_intercept"
  | "observation_intercept"
  | "initial_state_mean"
  | "initial_state_sd"
  | "static_state_sd"
  | "correlation"
  | "initial_state_correlation"
  | "loading"
  | "measurement_error_sd"
  | "observation_hyperparameter"
  | "observation_hyperparameter_positive";
/**
 * Constraints on parameter values.
 */
export type ParameterConstraint = "none" | "positive" | "negative" | "unit_interval" | "correlation";
/**
 * Global initial-state policy for retained dynamic states.
 */
export type InitializationPolicy = "stationary" | "free";
/**
 * Global policy for whether eligible manifest intercepts remain free.
 */
export type ObservationInterceptPolicy = "fixed" | "free";
/**
 * Distribution families allowed in Stage 4 prior proposals.
 */
export type PriorDistributionFamily =
  | "Normal"
  | "HalfNormal"
  | "Beta"
  | "Uniform"
  | "TruncatedNormal"
  | "Gamma"
  | "LogNormal"
  | "Exponential"
  | "Delta";

/**
 * Combined JSON Schema for all stage contracts. Generated from Python Pydantic models.
 */
export interface CausalSSMContracts {
  "stage-0": Stage0Contract;
  "stage-1a": Stage1AContract;
  "stage-1b": Stage1BContract;
  "stage-2": Stage2Contract;
  "stage-3": Stage3Contract;
  "stage-4": Stage4Contract;
  "stage-4b": Stage4BContract;
  "stage-5a": Stage5AContract;
  "stage-5b": Stage5BContract;
  "stage-6": Stage6Contract;
}
export interface Stage0Contract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  llm_trace?: LLMTrace | null;
  column_descriptions: Stage0ColumnDescriptionContract[];
}
/**
 * Full trace of an LLM multi-turn conversation.
 */
export interface LLMTrace {
  messages: TraceMessage[];
  model: string;
  total_time_seconds: number;
  usage: TraceUsage;
}
/**
 * A single message in an LLM trace.
 */
export interface TraceMessage {
  role: string;
  content: string;
  reasoning?: string | null;
  tool_calls?:
    | {
        [k: string]: any | undefined;
      }[]
    | null;
  tool_call_id?: string | null;
  tool_name?: string | null;
  tool_result?: string | null;
  tool_is_error: boolean;
}
/**
 * Token usage for an LLM trace.
 */
export interface TraceUsage {
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens?: number | null;
}
export interface Stage0ColumnDescriptionContract {
  name: string;
  description: string;
}
export interface Stage1AContract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  llm_trace?: LLMTrace | null;
  latent_model: LatentModel;
}
/**
 * Theoretical causal structure over constructs.
 */
export interface LatentModel {
  /**
   * Theoretical constructs in the model
   */
  constructs: Construct[];
  /**
   * Causal edges between constructs
   */
  edges: CausalEdge[];
}
/**
 * A theoretical entity in the latent causal model.
 */
export interface Construct {
  /**
   * Construct name (e.g., 'stress', 'sleep_quality')
   */
  name: string;
  /**
   * What this theoretical construct represents
   */
  description: string;
  role: Role;
  /**
   * True if this is the primary outcome variable Y implied by the question
   */
  is_outcome: boolean;
  temporal_status: TemporalStatus;
}
/**
 * A directed causal relationship between constructs.
 */
export interface CausalEdge {
  /**
   * Name of cause construct
   */
  cause: string;
  /**
   * Name of effect construct
   */
  effect: string;
  /**
   * Theoretical justification for this causal link
   */
  description: string;
  /**
   * If True, effect at t is caused by cause at t-1 (one model_clock tick delay). If False (contemporaneous), effect at t is caused by cause at t.
   */
  lagged: boolean;
}
export interface Stage1BContract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  llm_trace?: LLMTrace | null;
  causal_spec: CausalSpec;
}
/**
 * Complete causal specification combining latent and measurement models.
 */
export interface CausalSpec {
  latent: LatentModel;
  measurement: MeasurementModel;
  /**
   * Identifiability status of target causal effects
   */
  identifiability?: IdentifiabilityStatus | null;
  /**
   * Deterministic estimation-time projection consumed by downstream fitting
   */
  estimation?: EstimationSpec | null;
}
/**
 * Operationalization of constructs into observed indicators.
 */
export interface MeasurementModel {
  /**
   * Observed indicators, each measuring a construct
   */
  indicators: Indicator[];
  /**
   * Observation window width for extraction and SSM discretization. Any Polars-compatible duration string (e.g. '1h', '4h', '1d', '1w'). Choose based on data density: need enough events per support window.
   */
  model_clock: string;
}
/**
 * An observed variable that reflects a construct.
 */
export interface Indicator {
  /**
   * Indicator name (e.g., 'hrv', 'self_reported_stress')
   */
  name: string;
  /**
   * Which construct this indicator measures
   */
  construct_name: string;
  /**
   * Instructions for workers on how to extract this from data
   */
  how_to_measure: string;
  construct_polarity: IndicatorPolarity;
  /**
   * 'continuous', 'binary', 'count', 'ordinal', 'categorical'
   */
  measurement_dtype: string;
  /**
   * Aggregation function applied when bucketing raw extractions within the indicator support window. Measurement-model support is currently limited to: first, last, sum, count, mean, std. Available parser operators: count, cv, entropy, first, instability, iqr, kurtosis, last, max, mean, median, min, n_unique, p10, p25, p75, p90, p99, range, skew, std, sum, trend, var
   */
  aggregation: string;
  /**
   * Optional duration string describing the support window summarized by this indicator (for example '1mo' for a monthly average on a daily model clock). If omitted, the support window defaults to the global model_clock.
   */
  observation_window?: string | null;
  /**
   * Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
   */
  ordinal_levels?: string[] | null;
  /**
   * Raw data column names referenced by how_to_measure. Used to project chunks to only relevant columns before extraction.
   */
  source_columns: string[];
  /**
   * Optional deterministic support-window expression for extraction_mode='computed'. Use this when a computed indicator needs formulas, thresholds, or multiple source columns instead of a direct single-column aggregation. The expression must return one scalar per support window.
   */
  computed_rule?: ComputedRule | null;
  /**
   * 'computed' (deterministic pipeline extraction) or 'semantic' (LLM extraction). Use 'computed' when the indicator can be derived deterministically either from a direct source-column aggregation or from a computed_rule support-window expression over the declared source_columns.
   */
  extraction_mode: string;
  support_kind: SupportKind;
  summary_operator: SummaryOperator;
  anchor_policy: AnchorPolicy;
}
/**
 * Deterministic per-window expression for computed indicators.
 */
export interface ComputedRule {
  /**
   * Deterministic support-window expression that returns one scalar per window. Use Python-like syntax over source_columns with arithmetic, comparisons, if/else, and helper functions such as any(), sum(), mean(), std(), first(), last(), count_true(), count_non_null(), lower(), contains(), and contains_any(). Use None for missing values.
   */
  window_expr: string;
}
/**
 * Status of causal effect identifiability.
 */
export interface IdentifiabilityStatus {
  /**
   * Treatments with identifiable effects and how to estimate them
   */
  identifiable_treatments: {
    [k: string]: IdentifiedTreatmentStatus | undefined;
  };
  /**
   * Treatments whose effects are currently not identifiable
   */
  non_identifiable_treatments: {
    [k: string]: NonIdentifiableTreatmentStatus | undefined;
  };
}
/**
 * Details on how a treatment effect is identified.
 */
export interface IdentifiedTreatmentStatus {
  /**
   * Identification strategy (e.g., do_calculus, instrumental_variable)
   */
  method: string;
  /**
   * Closed-form estimand or IV placeholder
   */
  estimand: string;
  /**
   * Unobserved confounders the estimand integrates out
   */
  marginalized_confounders: string[];
  /**
   * Instrumental variables used (if method=instrumental_variable)
   */
  instruments: string[];
}
/**
 * Context on why a treatment effect is not identifiable.
 */
export interface NonIdentifiableTreatmentStatus {
  /**
   * Unobserved constructs blocking identification
   */
  confounders: string[];
  /**
   * Optional explanation if confounders cannot be enumerated
   */
  notes?: string | null;
}
/**
 * Deterministic estimation-time projection of the user-facing latent DAG.
 */
export interface EstimationSpec {
  /**
   * Retained latent states in canonical array order for compilation
   */
  state_order: string[];
  /**
   * Directed estimation graph over retained states
   */
  edges: CausalEdge[];
  /**
   * Dependencies induced after marginalizing latent root confounders
   */
  induced_dependencies: InducedDependency[];
  /**
   * Observed construct trajectories compiled as B u(t) transition inputs
   */
  known_inputs: KnownInput[];
}
/**
 * Dependence induced among retained states after marginalizing latent roots.
 */
export interface InducedDependency {
  /**
   * Pair of retained states whose joint dependence is induced
   *
   * @minItems 2
   * @maxItems 2
   */
  between: [any, any];
  /**
   * Which covariance block the induced dependence belongs to
   */
  kind: "innovation_correlation" | "initial_state_correlation";
  /**
   * Marginalized source constructs that induce this dependence
   */
  source_confounders: string[];
}
/**
 * Observed input trajectory used as a deterministic transition driver.
 */
export interface KnownInput {
  /**
   * Construct removed from the latent state vector
   */
  construct: string;
  /**
   * Measurement indicator column supplying u(t)
   */
  source_indicator: string;
  /**
   * Positive divisor applied to the source indicator before inference
   */
  scale: number;
  /**
   * How to fill missing input values on the model time grid
   */
  missing_policy: "zero" | "forward_fill";
}
export interface Stage2Contract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  llm_trace?: LLMTrace | null;
  workers: WorkerStatusContract[];
}
export interface WorkerStatusContract {
  worker_id: number;
  status: "pending" | "running" | "completed" | "failed";
  n_extractions: number;
  n_windows: number;
  error?: string | null;
}
export interface Stage3Contract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  is_valid: boolean;
  indicators: {
    [k: string]: IndicatorAuditContract | undefined;
  };
  dataset_issues: ValidationIssueContract[];
}
export interface IndicatorAuditContract {
  profile?: IndicatorEmpiricalProfileContract | null;
  validation: IndicatorValidationContract;
}
export interface IndicatorEmpiricalProfileContract {
  measurement_dtype?: string | null;
  n_obs: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  q25?: number | null;
  q50?: number | null;
  q75?: number | null;
  variance: number | null;
  time_coverage_ratio: number | null;
  max_gap_ratio: number | null;
  dtype_violations?: number | null;
  duplicate_pct?: number | null;
  arithmetic_sequence_detected: boolean;
  n_unparseable_timestamps?: number | null;
  zero_fraction?: number | null;
  is_nonnegative?: boolean | null;
  is_unit_interval?: boolean | null;
  looks_integer_valued?: boolean | null;
  variance_to_mean_ratio?: number | null;
}
export interface IndicatorValidationContract {
  issues: ValidationIssueContract[];
  checks: {
    [k: string]: ("ok" | "warning" | "error") | undefined;
  };
}
export interface ValidationIssueContract {
  indicator?: string | null;
  issue_type: string;
  severity: "error" | "warning" | "info";
  message: string;
}
export interface Stage4Contract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  llm_trace?: LLMTrace | null;
  model_spec: ModelSpec;
  authored_priors: {
    [k: string]: PriorProposal | undefined;
  };
  resolved_priors: PriorProposal | undefined[];
  search_queries?: {
    [k: string]: string | undefined;
  } | null;
  validation_warnings?: string[] | null;
  prior_predictive_samples?: {
    [k: string]: number[] | undefined;
  } | null;
}
/**
 * Complete statistical model specification.
 */
export interface ModelSpec {
  /**
   * Likelihood specifications for each observed indicator
   */
  likelihoods: LikelihoodSpec[];
  /**
   * All parameters requiring priors
   */
  parameters: ParameterSpec[];
  initialization_policy: InitializationPolicy;
  observation_intercept_policy: ObservationInterceptPolicy;
  /**
   * Whether eligible dynamic states may carry a continuous-time intercept term
   */
  equilibrium_forcing: boolean;
}
/**
 * Specification for a likelihood (observed variable distribution).
 */
export interface LikelihoodSpec {
  /**
   * Name of the observed indicator variable
   */
  variable: string;
  distribution: DistributionFamily;
  link: LinkFunction;
  /**
   * Whether deterministic additive centering is applied before fitting
   */
  centered: boolean;
  /**
   * Why this distribution/link was chosen for this variable
   */
  reasoning: string;
  /**
   * Literature sources supporting this likelihood choice
   */
  sources: LikelihoodSource[];
}
/**
 * A source of evidence for a likelihood distribution choice.
 */
export interface LikelihoodSource {
  /**
   * Title of the source (paper, textbook, etc.)
   */
  title: string;
  /**
   * URL of the source if available
   */
  url?: string | null;
  /**
   * Relevant excerpt from the source
   */
  snippet: string;
}
/**
 * Specification for a parameter requiring a prior.
 */
export interface ParameterSpec {
  /**
   * Parameter name
   */
  name: string;
  role: ParameterRole;
  constraint: ParameterConstraint;
  /**
   * Human-readable description of what this parameter represents
   */
  description: string;
}
/**
 * A proposed prior distribution for a parameter.
 */
export interface PriorProposal {
  /**
   * Name of the parameter this prior is for
   */
  parameter: string;
  distribution: PriorDistributionFamily;
  /**
   * Distribution parameters (e.g., {'mu': 0.3, 'sigma': 0.1})
   */
  params: {
    [k: string]: number | undefined;
  };
  /**
   * Literature sources supporting this prior
   */
  sources: PriorSource[];
  /**
   * Justification for the chosen prior distribution and parameters
   */
  reasoning: string;
  /**
   * Observation interval (in days) that the DT prior is expressed in. Sourced from the study's measurement schedule (e.g., 7 for a weekly study). Used for DT→CT conversion of dynamic priors (e.g. beta/dt for cross-lags, -log(rho)/dt for baseline persistence).
   */
  reference_interval_days?: number | null;
  /**
   * Pre-computed density curve points [{x, y}, ...] for frontend visualization. Computed by the pipeline before persistence so the frontend doesn't need to approximate the PDF client-side.
   */
  density_points?:
    | {
        [k: string]: number | undefined;
      }[]
    | null;
}
/**
 * A source of evidence for a prior distribution.
 */
export interface PriorSource {
  /**
   * Title of the source (paper, meta-analysis, etc.)
   */
  title: string;
  /**
   * URL of the source if available
   */
  url?: string | null;
  /**
   * Relevant excerpt from the source
   */
  snippet: string;
  /**
   * Reported effect size if available (e.g., 'r=0.3', 'β=0.2')
   */
  effect_size?: string | null;
  /**
   * Observation/measurement interval of this study in days (daily=1, weekly=7, monthly=30)
   */
  study_interval_days?: number | null;
}
export interface Stage4BContract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  parametric_id: ParametricIdResult;
  inference_structure?: InferenceStructureResult | null;
}
/**
 * Full parametric identifiability result (Stage 4b payload).
 */
export interface ParametricIdResult {
  checked: boolean;
  sensitivity_analysis?: SensitivityAnalysisResult | null;
  map_geometry?: MAPGeometryResult | null;
  summary?: ParametricIdSummary | null;
  per_param_classification?: ParameterIdentification[] | null;
  threshold?: number | null;
  error?: string | null;
}
/**
 * Output sensitivity analysis result (pre-inference identifiability).
 *
 * Structural identifiability check via the Jacobian of the forward model's
 * emitted-observation moment summary. Near-zero singular values indicate
 * parameter combinations that observations cannot distinguish.
 */
export interface SensitivityAnalysisResult {
  singular_values: number[];
  normalized_singular_values: number[];
  deficiency_count: number;
  weak_directions: SensitivityDirection[];
  per_parameter: SensitivityEntry[];
  n_draws: number;
  n_observations: number;
  n_parameters: number;
}
/**
 * A direction in parameter space from the normalized sensitivity SVD.
 */
export interface SensitivityDirection {
  index: number;
  singular_value: number;
  normalized_singular_value: number;
  status: "pass" | "warn" | "fail";
  top_loadings: SensitivityDirectionLoading[];
}
/**
 * One parameter's loading within a weak local sensitivity direction.
 */
export interface SensitivityDirectionLoading {
  parameter: string;
  interpretable_parameter: string;
  loading: number;
  abs_loading: number;
}
/**
 * Per-parameter output sensitivity analysis entry.
 */
export interface SensitivityEntry {
  parameter: string;
  interpretable_parameter: string;
  sensitivity_norm: number;
  effective_sv: number;
  sv_status: "pass" | "warn" | "fail";
  normalized_effective_sv: number;
  normalized_sv_status: "pass" | "warn" | "fail";
  identifiable: boolean;
}
/**
 * Dataset-conditioned MAP search plus H_lik / H_post local geometry.
 */
export interface MAPGeometryResult {
  n_starts: number;
  n_successful_starts: number;
  best_start_index: number;
  map_log_posterior: number;
  map_log_likelihood: number;
  map_log_prior: number;
  final_grad_norm: number;
  runner_up_objective_gap?: number | null;
  starts: MAPOptimizationRun[];
  likelihood_curvature: MAPCurvatureResult;
  posterior_curvature: MAPCurvatureResult;
  prior_rescued_parameters: string[];
  boundary_parameters: string[];
}
/**
 * One start in the multi-start MAP search.
 */
export interface MAPOptimizationRun {
  index: number;
  start_kind: string;
  start_log_posterior: number;
  log_posterior: number;
  log_likelihood: number;
  log_prior: number;
  objective: number;
  success: boolean;
  status: number;
  message: string;
  n_iters: number;
  n_function_evals: number;
  grad_norm: number;
  distance_to_best: number;
}
/**
 * One Hessian family's local geometry at the selected MAP.
 */
export interface MAPCurvatureResult {
  eigenvalues: number[];
  normalized_eigenvalues: number[];
  negative_direction_count: number;
  deficiency_count: number;
  positive_definite: boolean;
  condition_number?: number | null;
  normalized_condition_number?: number | null;
  weak_directions: CurvatureDirection[];
  per_parameter: CurvatureParameterEntry[];
}
/**
 * A weak Hessian eigen-direction within the MAP neighborhood.
 */
export interface CurvatureDirection {
  index: number;
  eigenvalue: number;
  normalized_eigenvalue: number;
  status: "pass" | "warn" | "fail";
  top_loadings: CurvatureDirectionLoading[];
}
/**
 * One parameter's loading within a weak local-curvature eigen-direction.
 */
export interface CurvatureDirectionLoading {
  parameter: string;
  interpretable_parameter: string;
  loading: number;
  abs_loading: number;
}
/**
 * Per-parameter local-curvature summary at the selected MAP.
 */
export interface CurvatureParameterEntry {
  parameter: string;
  interpretable_parameter: string;
  diagonal_curvature: number;
  effective_eigenvalue: number;
  status: "pass" | "warn" | "fail";
  normalized_effective_eigenvalue: number;
  normalized_status: "pass" | "warn" | "fail";
}
/**
 * Summary of parametric identifiability issues.
 */
export interface ParametricIdSummary {
  structural_issues: string[];
  boundary_issues: string[];
  weak_params: string[];
}
/**
 * Per-parameter identifiability classification.
 */
export interface ParameterIdentification {
  name: string;
  classification: "identified" | "practically_unidentifiable" | "structurally_unidentifiable";
  contraction_ratio?: number | null;
  profile_x?: number[] | null;
  profile_ll?: number[] | null;
}
/**
 * Canonical inference-structure plan shared by pipeline and web.
 */
export interface InferenceStructureResult {
  likelihood_path: "kalman" | "composed" | "particle";
  auto_method: "aux_gibbs";
  first_pass_rb: FirstPassRBResult;
}
/**
 * Active first-pass Rao-Blackwellization plan for the prepared runtime.
 */
export interface FirstPassRBResult {
  status: "active" | "inactive";
  latent_variables: InferenceStructureVariable[];
  obs_variables: InferenceStructureVariable[];
}
/**
 * A single latent or observed channel assignment in the active split.
 */
export interface InferenceStructureVariable {
  name: string;
  method: "kalman" | "particle";
}
/**
 * SVI preflight: fast approximate fit before expensive inference.
 */
export interface Stage5AContract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  inference_metadata: InferenceMetadataContract;
  svi_diagnostics?: SVIDiagnostics | null;
  posterior_marginals?: PosteriorMarginal[] | null;
  posterior_pairs?: PosteriorPair[] | null;
}
export interface InferenceMetadataContract {
  method: string;
  n_samples: number;
  duration_seconds: number;
}
/**
 * SVI (variational inference) diagnostics.
 */
export interface SVIDiagnostics {
  elbo_losses: number[];
}
/**
 * Marginal posterior density for a single scalar parameter.
 */
export interface PosteriorMarginal {
  parameter: string;
  x_values: number[];
  density: number[];
  mean: number;
  sd: number;
  hdi_3: number;
  hdi_97: number;
}
/**
 * Pairwise posterior scatter data for joint visualization.
 */
export interface PosteriorPair {
  param_x: string;
  param_y: string;
  x_values: number[];
  y_values: number[];
  divergent?: boolean[] | null;
}
export interface Stage5BContract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  power_scaling: PowerScalingResultContract[];
  ppc: PPCResultContract;
  inference_metadata: InferenceMetadataContract;
  mcmc_diagnostics?: MCMCDiagnostics | null;
  svi_diagnostics?: SVIDiagnostics | null;
  smc_diagnostics?: SMCDiagnostics | null;
  loo_diagnostics?: LOODiagnostics | null;
  posterior_marginals?: PosteriorMarginal[] | null;
  posterior_pairs?: PosteriorPair[] | null;
}
export interface PowerScalingResultContract {
  parameter: string;
  diagnosis: "prior_dominated" | "well_identified" | "prior_data_conflict";
  prior_sensitivity: number;
  likelihood_sensitivity: number;
  psis_k_hat?: number | null;
}
export interface PPCResultContract {
  per_variable_warnings: PPCWarning[];
  checked?: boolean | null;
  n_subsample?: number | null;
  overlays: PPCOverlay[];
  test_stats: PPCTestStat[];
}
/**
 * A single diagnostic warning for one manifest variable.
 */
export interface PPCWarning {
  variable: string;
  check_type: "calibration" | "autocorrelation" | "variance";
  message: string;
  value: number;
  passed: boolean;
}
/**
 * Per-variable quantile bands for PPC ribbon/density overlay plots.
 *
 * Provides the data for Gabry's ppc_dens_overlay / ppc_ribbon plots:
 * observed time series vs posterior predictive quantile bands.
 * Optionally includes individual y_rep draw lines for spaghetti plots.
 */
export interface PPCOverlay {
  variable: string;
  observed: (number | null)[];
  q025: number[];
  q25: number[];
  median: number[];
  q75: number[];
  q975: number[];
  spaghetti_draws: number[][];
}
/**
 * Distribution of a test statistic across y_rep draws vs observed.
 *
 * Provides the data for Gabry's ppc_stat plots: histogram of T(y_rep)
 * with a vertical line at T(y_observed).
 */
export interface PPCTestStat {
  variable: string;
  stat_name: "mean" | "sd" | "min" | "max";
  observed_value: number;
  rep_values: number[];
}
/**
 * Top-level MCMC diagnostics container.
 */
export interface MCMCDiagnostics {
  per_parameter: MCMCParamDiagnostic[];
  num_divergences: number;
  divergence_rate: number;
  tree_depth_mean: number;
  tree_depth_max: number;
  accept_prob_mean: number;
  num_chains?: number | null;
  num_samples?: number | null;
  trace_data?: TraceData[] | null;
  rank_histograms?: RankHistogram[] | null;
  energy?: EnergyDiagnostics | null;
}
/**
 * Per-parameter MCMC convergence diagnostics.
 */
export interface MCMCParamDiagnostic {
  parameter: string;
  r_hat: number | number[];
  ess_bulk: number | number[];
  ess_tail?: number | number[] | null;
  mcse_mean?: number | number[] | null;
}
/**
 * Per-parameter trace data across chains.
 */
export interface TraceData {
  parameter: string;
  chains: TraceChain[];
}
/**
 * Thinned trace values for a single chain.
 */
export interface TraceChain {
  chain: number;
  values: number[];
}
/**
 * Per-parameter rank histogram for chain mixing assessment.
 */
export interface RankHistogram {
  parameter: string;
  n_bins: number;
  expected_per_bin: number;
  chains: RankHistogramChain[];
}
/**
 * Rank histogram bin counts for a single chain.
 */
export interface RankHistogramChain {
  chain: number;
  counts: number[];
}
/**
 * Hamiltonian energy diagnostics.
 */
export interface EnergyDiagnostics {
  energy_hist: EnergyHistogram;
  energy_transition_hist: EnergyHistogram;
  bfmi: number[];
}
/**
 * Histogram of energy values (bin centers + density).
 */
export interface EnergyHistogram {
  bin_centers: number[];
  density: number[];
}
/**
 * Sequential Monte Carlo diagnostics.
 */
export interface SMCDiagnostics {
  beta_schedule: number[];
  ess_history: number[];
  accept_rates: number[];
  n_levels: number;
  n_particles: number;
}
/**
 * Leave-one-out cross-validation diagnostics (ArviZ).
 *
 * Uses one-step-ahead predictive log-likelihoods from the filter's
 * innovation decomposition. Each LOO "observation" is one complete
 * timestep (all manifest variables at time t), not individual cells.
 */
export interface LOODiagnostics {
  elpd_loo: number;
  p_loo: number;
  se: number;
  n_data_points: number;
  observation_unit: string;
  pareto_k?: number[] | null;
  n_bad_k?: number | null;
  loo_pit?: number[] | null;
}
export interface Stage6Contract {
  outcome: "success" | "warn" | "fail";
  fail_reason?: string | null;
  llm_trace?: LLMTrace | null;
  intervention_results: TreatmentEffectContract[];
  saved_scenarios?: SavedScenarioContract[] | null;
  final_summary?: string | null;
}
export interface TreatmentEffectContract {
  treatment: string;
  posterior_draws?: number[] | null;
  temporal?: TemporalEffect | null;
  manifest_effects?: {
    [k: string]: number | undefined;
  } | null;
}
/**
 * Temporal decomposition of a treatment effect.
 */
export interface TemporalEffect {
  effect_1d: number;
  effect_7d: number;
  effect_30d: number;
  peak_effect: number;
  time_to_peak_days: number;
}
export interface SavedScenarioContract {
  label: string;
  query: string;
  summary?: string | null;
}
