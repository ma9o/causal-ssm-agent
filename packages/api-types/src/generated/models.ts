/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python Pydantic models via:
 *   cd apps/data-pipeline && uv run python scripts/export_schemas.py
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/nof1_causal_lab/flows/artifact_contracts.py
 * plus facade API models exported from apps/data-pipeline/src/nof1_causal_lab/episode_api.py
 */

/**
 * Whether a variable is modeled or treated as given.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Role".
 */
export type Role = "endogenous" | "exogenous";
/**
 * Whether a construct changes within-person over time.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TemporalStatus".
 */
export type TemporalStatus = "time_varying" | "time_invariant";
/**
 * Direction of an indicator relative to its construct.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorPolarity".
 */
export type IndicatorPolarity = "positive" | "negative";
/**
 * Whether the measurement equation is point-local or interval-summary.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SupportKind".
 */
export type SupportKind = "point" | "interval";
/**
 * Supported summary operators for indicator observations.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SummaryOperator".
 */
export type SummaryOperator = "first" | "last" | "sum" | "count" | "mean" | "std";
/**
 * Which support boundary receives the observation anchor.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "AnchorPolicy".
 */
export type AnchorPolicy = "support_start" | "support_end";
/**
 * Distribution families for observation and process noise.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DistributionFamily".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LinkFunction".
 */
export type LinkFunction = "identity" | "log" | "inverse" | "logit" | "probit" | "cumulative_logit" | "softmax";
/**
 * Role of a parameter in the model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterRole".
 */
export type ParameterRole =
  | "fixed_effect"
  | "ar_coefficient"
  | "dynamics_parameter"
  | "dynamics_parameter_positive"
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterConstraint".
 */
export type ParameterConstraint = "none" | "positive" | "negative" | "unit_interval" | "correlation";
/**
 * Global initial-state policy for retained dynamic states.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InitializationPolicy".
 */
export type InitializationPolicy = "stationary" | "free";
/**
 * Global policy for whether eligible manifest intercepts remain free.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ObservationInterceptPolicy".
 */
export type ObservationInterceptPolicy = "fixed" | "free";
/**
 * Distribution families allowed in model-spec prior proposals.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorDistributionFamily".
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
 * Combined JSON Schema for exported artifact contracts and facade API models. Generated from Python Pydantic models.
 */
export interface CausalSSMContracts {
  raw_data: RawDataContract;
  latent_structure: LatentStructureContract;
  measurement_structure: MeasurementStructureContract;
  measurements: MeasurementsContract;
  validation_report: ValidationReportContract;
  statistical_model_spec: StatisticalModelSpecContract;
  posterior: PosteriorContract;
  baseline_report: BaselineReportContract;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataContract".
 */
export interface RawDataContract {
  llm_trace_ref?: string | null;
  column_descriptions: RawDataColumnDescriptionContract[];
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataColumnDescriptionContract".
 */
export interface RawDataColumnDescriptionContract {
  name: string;
  description: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LatentStructureContract".
 */
export interface LatentStructureContract {
  llm_trace_ref?: string | null;
  latent_structure: LatentStructure;
}
/**
 * Theoretical causal structure over constructs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LatentStructure".
 */
export interface LatentStructure {
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Construct".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CausalEdge".
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
  /**
   * Literature sources supporting this causal link
   */
  sources: EdgeSource[];
}
/**
 * A source of evidence supporting a causal edge.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EdgeSource".
 */
export interface EdgeSource {
  /**
   * Title of the source (paper, meta-analysis, textbook, etc.)
   */
  title: string;
  /**
   * URL of the source if available
   */
  url?: string | null;
  /**
   * Relevant excerpt or paraphrase from the source
   */
  snippet: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementStructureContract".
 */
export interface MeasurementStructureContract {
  llm_trace_ref?: string | null;
  measurement_structure: MeasurementStructure;
}
/**
 * Operationalization of constructs into observed indicators.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementStructure".
 */
export interface MeasurementStructure {
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Indicator".
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
   * Aggregation function applied when bucketing raw extractions within the indicator support window. Measurement-structure support is currently limited to: first, last, sum, count, mean, std. Available parser operators: count, cv, entropy, first, instability, iqr, kurtosis, last, max, mean, median, min, n_unique, p10, p25, p75, p90, p99, range, skew, std, sum, trend, var
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ComputedRule".
 */
export interface ComputedRule {
  /**
   * Deterministic support-window expression that returns one scalar per window. Use Python-like syntax over source_columns with arithmetic, comparisons, if/else, and helper functions such as any(), sum(), mean(), std(), first(), last(), count_true(), count_non_null(), lower(), contains(), and contains_any(). Use None for missing values.
   */
  window_expr: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementsContract".
 */
export interface MeasurementsContract {
  llm_trace_ref?: string | null;
  workers: WorkerStatusContract[];
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WorkerStatusContract".
 */
export interface WorkerStatusContract {
  worker_id: number;
  status: "pending" | "running" | "completed" | "failed";
  n_extractions: number;
  n_windows: number;
  error?: string | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ValidationReportContract".
 */
export interface ValidationReportContract {
  is_valid: boolean;
  indicators: {
    [k: string]: IndicatorAuditContract | undefined;
  };
  dataset_issues: ValidationIssueContract[];
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorAuditContract".
 */
export interface IndicatorAuditContract {
  profile?: IndicatorEmpiricalProfileContract | null;
  validation: IndicatorValidationContract;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorEmpiricalProfileContract".
 */
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
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorValidationContract".
 */
export interface IndicatorValidationContract {
  issues: ValidationIssueContract[];
  checks: {
    [k: string]: ("ok" | "warning" | "error") | undefined;
  };
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ValidationIssueContract".
 */
export interface ValidationIssueContract {
  indicator?: string | null;
  issue_type: string;
  severity: "error" | "warning" | "info";
  message: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StatisticalModelSpecContract".
 */
export interface StatisticalModelSpecContract {
  llm_trace_ref?: string | null;
  statistical_model_spec: StatisticalModelSpec;
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StatisticalModelSpec".
 */
export interface StatisticalModelSpec {
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LikelihoodSpec".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LikelihoodSource".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterSpec".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorProposal".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorSource".
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
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorContract".
 */
export interface PosteriorContract {
  ppc: PPCResultContract;
  inference_metadata: InferenceMetadataContract;
  mcmc_diagnostics?: MCMCDiagnostics | null;
  smc_diagnostics?: SMCDiagnostics | null;
  loo_diagnostics?: LOODiagnostics | null;
  posterior_marginals?: PosteriorMarginal[] | null;
  posterior_pairs?: PosteriorPair[] | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PPCResultContract".
 */
export interface PPCResultContract {
  per_variable_warnings: PPCWarning[];
  checked?: boolean | null;
  n_subsample?: number | null;
  overlays: PPCOverlay[];
  test_stats: PPCTestStat[];
}
/**
 * A single diagnostic warning for one manifest variable.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PPCWarning".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PPCOverlay".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PPCTestStat".
 */
export interface PPCTestStat {
  variable: string;
  stat_name: "mean" | "sd" | "min" | "max";
  observed_value: number;
  rep_values: number[];
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InferenceMetadataContract".
 */
export interface InferenceMetadataContract {
  method: string;
  n_samples: number;
  duration_seconds: number;
}
/**
 * Top-level MCMC diagnostics container.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MCMCDiagnostics".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MCMCParamDiagnostic".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceData".
 */
export interface TraceData {
  parameter: string;
  chains: TraceChain[];
}
/**
 * Thinned trace values for a single chain.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceChain".
 */
export interface TraceChain {
  chain: number;
  values: number[];
}
/**
 * Per-parameter rank histogram for chain mixing assessment.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RankHistogram".
 */
export interface RankHistogram {
  parameter: string;
  n_bins: number;
  expected_per_bin: number;
  chains: RankHistogramChain[];
}
/**
 * Rank histogram bin counts for a single chain.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RankHistogramChain".
 */
export interface RankHistogramChain {
  chain: number;
  counts: number[];
}
/**
 * Hamiltonian energy diagnostics.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EnergyDiagnostics".
 */
export interface EnergyDiagnostics {
  energy_hist: EnergyHistogram;
  energy_transition_hist: EnergyHistogram;
  bfmi: number[];
}
/**
 * Histogram of energy values (bin centers + density).
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EnergyHistogram".
 */
export interface EnergyHistogram {
  bin_centers: number[];
  density: number[];
}
/**
 * Sequential Monte Carlo diagnostics.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SMCDiagnostics".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LOODiagnostics".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorMarginal".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorPair".
 */
export interface PosteriorPair {
  param_x: string;
  param_y: string;
  x_values: number[];
  y_values: number[];
  divergent?: boolean[] | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BaselineReportContract".
 */
export interface BaselineReportContract {
  llm_trace_ref?: string | null;
  intervention_results: TreatmentEffectContract[];
  saved_scenarios?: SavedScenarioContract[] | null;
  final_summary?: string | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TreatmentEffectContract".
 */
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TemporalEffect".
 */
export interface TemporalEffect {
  effect_1d: number;
  effect_7d: number;
  effect_30d: number;
  peak_effect: number;
  time_to_peak_days: number;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SavedScenarioContract".
 */
export interface SavedScenarioContract {
  label: string;
  query: string;
  summary?: string | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactEnvelope".
 */
export interface ArtifactEnvelope {
  workspace_id: string;
  artifact_id:
    | "question"
    | "raw_data"
    | "latent_structure"
    | "measurement_structure"
    | "causal_design"
    | "identification_report"
    | "measurements"
    | "panel"
    | "validation_report"
    | "statistical_model_spec"
    | "compiled_ssm"
    | "posterior"
    | "baseline_report"
    | "saved_scenarios";
  version: number;
  meta: ArtifactVersionInfo;
  payload: {
    [k: string]: any | undefined;
  };
  binary_files: string[];
}
/**
 * Immutable metadata for one artifact version (payload lives in the store).
 *
 * ``derived_from`` pins the exact input versions the payload was computed
 * from. For root artifacts (user writes) it is empty. ``created_at`` is
 * stamped by the activity that produced the version — never inside workflow
 * code, where wall-clock time is non-deterministic.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactVersionInfo".
 */
export interface ArtifactVersionInfo {
  artifact_id:
    | "question"
    | "raw_data"
    | "latent_structure"
    | "measurement_structure"
    | "causal_design"
    | "identification_report"
    | "measurements"
    | "panel"
    | "validation_report"
    | "statistical_model_spec"
    | "compiled_ssm"
    | "posterior"
    | "baseline_report"
    | "saved_scenarios";
  version: number;
  provenance: "computed" | "human" | "llm";
  derived_from: {
    [k: string]: number | undefined;
  };
  produced_by?: string | null;
  created_at: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CapabilitiesResponse".
 */
export interface CapabilitiesResponse {
  moves_enabled: boolean;
}
/**
 * Full trace of an LLM multi-turn conversation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LLMTrace".
 */
export interface LLMTrace {
  messages: TraceMessage[];
  model: string;
  total_time_seconds: number;
  usage: TraceUsage;
}
/**
 * A single message in an LLM trace.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceMessage".
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
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceUsage".
 */
export interface TraceUsage {
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens?: number | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "UploadResponse".
 */
export interface UploadResponse {
  path: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WorkspaceEntry".
 */
export interface WorkspaceEntry {
  href: string;
  question?: string | null;
  workspaceId: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WorkspaceList".
 */
export interface WorkspaceList {
  workspaces: WorkspaceEntry[];
}
