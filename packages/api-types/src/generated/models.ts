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
 * Whether a variable is modeled (endogenous) or given (exogenous).
 */
export type Role = "endogenous" | "exogenous";
/**
 * Whether a variable changes over time.
 */
export type TemporalStatus = "time_varying" | "time_invariant";
/**
 * Distribution families for observation and process noise.
 *
 * Used throughout the SSM pipeline: from LLM-proposed likelihoods to
 * emission function dispatch.  Values are lowercase so they can be
 * passed directly as strings to likelihood backends.
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
  | "random_intercept_sd"
  | "correlation"
  | "loading";
/**
 * Constraints on parameter values.
 */
export type ParameterConstraint = "none" | "positive" | "unit_interval" | "correlation";

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
  "stage-5": Stage5Contract;
  "stage-6": Stage6Contract;
  _partial: PartialStageResult;
}
export interface Stage0Contract {
  outcome: "success" | "warn" | "fail";
  source_type: string;
  source_label: string;
  n_records: number;
  date_range: DateRangeContract;
  sample: {
    [k: string]: (string | null) | undefined;
  }[];
}
export interface DateRangeContract {
  start: string;
  end: string;
}
export interface Stage1AContract {
  outcome: "success" | "warn" | "fail";
  latent_model: LatentModel;
  outcome_name: string;
  treatments: string[];
  llm_trace?: LLMTrace | null;
}
/**
 * Theoretical causal structure over constructs (the latent model).
 *
 * This is the output of Stage 1a - proposed based on domain knowledge alone,
 * without seeing data. Defines the topological structure among latent constructs.
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
 * A theoretical entity in the causal model.
 *
 * Constructs are conceptually 'latent' - they represent theoretical entities
 * that may be measured by one or more observed indicators.
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
  /**
   * 'hourly', 'daily', 'weekly', 'monthly', 'yearly'. Required for time-varying constructs. The timescale at which causal dynamics operate.
   */
  temporal_scale?: string | null;
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
   * If True, effect at t is caused by cause at t-1. If False (contemporaneous), effect at t is caused by cause at t. Cross-timescale edges are always lagged.
   */
  lagged: boolean;
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
export interface Stage1BContract {
  outcome: "success" | "warn" | "fail";
  causal_spec: CausalSpec;
  llm_trace?: LLMTrace | null;
  gate_overridden?: GateOverrideContract | null;
}
/**
 * Complete causal specification combining latent and measurement models.
 *
 * This is the full model after both Stage 1a (latent) and Stage 1b (measurement).
 * Includes identifiability status for target causal effects.
 */
export interface CausalSpec {
  latent: LatentModel;
  measurement: MeasurementModel;
  /**
   * Identifiability status of target causal effects
   */
  identifiability?: IdentifiabilityStatus | null;
}
/**
 * Operationalization of constructs into observed indicators.
 *
 * This is the output of Stage 1b - proposed after seeing data sample,
 * given the latent model from Stage 1a.
 *
 * Each construct from the latent model must have at least one indicator.
 */
export interface MeasurementModel {
  /**
   * Observed indicators, each measuring a construct
   */
  indicators: Indicator[];
}
/**
 * An observed variable that reflects a construct.
 *
 * Following the reflective measurement model (A1), causality flows from
 * construct to indicator: the latent construct causes the observed values.
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
  /**
   * 'continuous', 'binary', 'count', 'ordinal', 'categorical'
   */
  measurement_dtype: string;
  /**
   * Aggregation function applied when bucketing raw extractions within aggregation window. Available: count, cv, entropy, first, instability, iqr, kurtosis, last, max, mean, median, min, n_unique, p10, p25, p75, p90, p99, range, skew, std, sum, trend, var
   */
  aggregation: string;
  /**
   * Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
   */
  ordinal_levels?: string[] | null;
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
export interface GateOverrideContract {
  reason: string;
}
export interface Stage2Contract {
  outcome: "success" | "warn" | "fail";
  workers: WorkerStatusContract[];
  combined_extractions_sample: ExtractionContract[];
  per_indicator_counts: {
    [k: string]: number | undefined;
  };
}
export interface WorkerStatusContract {
  worker_id: number;
  status: "pending" | "running" | "completed" | "failed";
  n_extractions: number;
  chunk_size: number;
  error?: string | null;
}
export interface ExtractionContract {
  indicator: string;
  value: number | boolean | string | null;
  timestamp: string | null;
}
export interface Stage3Contract {
  outcome: "success" | "warn" | "fail";
  validation_report: ValidationReportContract;
}
export interface ValidationReportContract {
  is_valid: boolean;
  issues: ValidationIssueContract[];
  per_indicator_health: IndicatorHealthContract[];
}
export interface ValidationIssueContract {
  indicator: string;
  issue_type: string;
  severity: "error" | "warning" | "info";
  message: string;
}
export interface IndicatorHealthContract {
  indicator: string;
  n_obs: number;
  variance: number | null;
  time_coverage_ratio: number | null;
  max_gap_ratio: number | null;
  dtype_violations: number;
  duplicate_pct: number;
  arithmetic_sequence_detected: boolean;
  cell_statuses: {
    [k: string]: ("ok" | "warning" | "error") | undefined;
  };
}
export interface Stage4Contract {
  outcome: "success" | "warn" | "fail";
  model_spec: ModelSpec;
  priors: PriorProposal[];
  validation_retries?: ValidationRetryContract[] | null;
  llm_trace?: LLMTrace | null;
  prior_predictive_samples?: {
    [k: string]: number[] | undefined;
  } | null;
}
/**
 * Complete model specification from orchestrator.
 *
 * This is what the orchestrator proposes based on the CausalSpec structure.
 * It enumerates all parameters needing priors and specifies the statistical model.
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
  /**
   * Overall reasoning for the model specification choices
   */
  reasoning: string;
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
   * Why this distribution/link was chosen for this variable
   */
  reasoning: string;
}
/**
 * Specification for a parameter requiring a prior.
 */
export interface ParameterSpec {
  /**
   * Parameter name (e.g., 'beta_stress_anxiety', 'rho_mood')
   */
  name: string;
  role: ParameterRole;
  constraint: ParameterConstraint;
  /**
   * Human-readable description of what this parameter represents
   */
  description: string;
  /**
   * Context for Exa literature search to find relevant effect sizes
   */
  search_context: string;
}
/**
 * A proposed prior distribution for a parameter.
 */
export interface PriorProposal {
  /**
   * Name of the parameter this prior is for
   */
  parameter: string;
  /**
   * Distribution name (e.g., 'Normal', 'HalfNormal', 'Beta', 'Uniform')
   */
  distribution: string;
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
   * Observation interval (in days) that the DT prior is expressed in. Sourced from the study's measurement schedule (e.g., 7 for a weekly study). Used for DT→CT conversion: drift = beta / reference_interval_days.
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
export interface ValidationRetryContract {
  attempt: number;
  failed_params: string[];
  feedback: string;
}
export interface Stage4BContract {
  outcome: "success" | "warn" | "fail";
  parametric_id: ParametricIdResult;
  rb_partition?: RBPartitionResult | null;
  gate_overridden?: GateOverrideContract | null;
}
/**
 * Full parametric identifiability result (Stage 4b payload).
 */
export interface ParametricIdResult {
  checked: boolean;
  t_rule?: TRuleResult | null;
  sensitivity_analysis?: SensitivityAnalysisResult | null;
  summary?: ParametricIdSummary | null;
  per_param_classification?: ParameterIdentification[] | null;
  threshold?: number | null;
  error?: string | null;
}
/**
 * Result of the t-rule (counting condition) check.
 *
 * The t-rule is a necessary condition for identification: if the number
 * of free parameters exceeds the number of available moment conditions,
 * the model is provably non-identified.
 *
 * For cross-sectional SEMs the constraint is n_params <= p(p+1)/2.
 * For time series (SSMs), autocovariance at each lag provides p^2
 * additional moment conditions, so the constraint is much weaker.
 */
export interface TRuleResult {
  n_free_params: number;
  n_manifest: number;
  n_timepoints: number | null;
  n_moments: number;
  satisfies: boolean;
  param_counts: {
    [k: string]: number | undefined;
  };
}
/**
 * Output sensitivity analysis result (pre-inference identifiability).
 *
 * Structural identifiability check via the Jacobian of the forward model's
 * predicted observation means and variances. Near-zero singular values
 * indicate parameter combinations that observations cannot distinguish.
 */
export interface SensitivityAnalysisResult {
  singular_values: number[];
  condition_number: number;
  per_parameter: SensitivityEntry[];
  n_draws: number;
  n_observations: number;
  n_parameters: number;
}
/**
 * Per-parameter output sensitivity analysis entry.
 */
export interface SensitivityEntry {
  parameter: string;
  sensitivity_norm: number;
  effective_sv: number;
  identifiable: boolean;
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
 * First-pass Rao-Blackwellization partition for frontend display.
 */
export interface RBPartitionResult {
  latent_variables: RBVariable[];
  obs_variables: RBVariable[];
}
/**
 * A single variable's Rao-Blackwellization assignment.
 */
export interface RBVariable {
  name: string;
  method: "kalman" | "particle";
}
export interface Stage5Contract {
  outcome: "success" | "warn" | "fail";
  power_scaling: PowerScalingResultContract[];
  ppc: PPCResultContract;
  inference_metadata: InferenceMetadataContract;
  mcmc_diagnostics?: MCMCDiagnostics | null;
  svi_diagnostics?: SVIDiagnostics | null;
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
export interface InferenceMetadataContract {
  method: string;
  n_samples: number;
  duration_seconds: number;
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
 * NUTS energy diagnostics (Betancourt 2017).
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
 * SVI (variational inference) diagnostics.
 */
export interface SVIDiagnostics {
  elbo_losses: number[];
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
export interface Stage6Contract {
  outcome: "success" | "warn" | "fail";
  intervention_results: TreatmentEffectContract[];
  inference_metadata: InferenceMetadataContract;
}
export interface TreatmentEffectContract {
  treatment: string;
  effect_size: number | null;
  posterior_draws?: number[] | null;
  prob_positive?: number | null;
  identifiable: boolean;
  warning?: string | null;
  ppc_warnings?: string[] | null;
  prior_sensitivity_warning?: string | null;
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
/**
 * Partial stage result written to disk during LLM generation.
 *
 * A subset of the full stage contract: only the ``llm_trace`` field (the part
 * available mid-run) plus ``_live`` metadata so the frontend can distinguish
 * in-progress from completed results.  Overwritten by ``persist_web_result``
 * when the stage completes.
 */
export interface PartialStageResult {
  llm_trace: LLMTrace;
  _live: LiveMetadata;
}
/**
 * Metadata attached to partial stage results while an LLM stage is running.
 */
export interface LiveMetadata {
  status: "running";
  label: string;
  turn: number;
  elapsed_seconds: number;
}
