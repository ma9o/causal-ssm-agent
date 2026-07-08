// ---------------------------------------------------------------------------
// Hand-written (frontend-only) — not generated from Python
// ---------------------------------------------------------------------------

export type { ArtifactStatus, ArtifactViewState, PipelineRun, RunStatus } from "./run";
export type {
  ArtifactId,
  ArtifactViewId,
  TransitionId,
  TransitionLogScopePolicy,
  TransitionMeta,
} from "./transitions";
export { ARTIFACT_IDS, ARTIFACT_VIEW_IDS, TRANSITION_META, TRANSITIONS } from "./transitions";
export type {
  CausalDesign,
  IdentifiabilityStatus,
  IdentifiedTreatmentStatus,
  NonIdentifiableTreatmentStatus,
} from "./causal-design";

// ---------------------------------------------------------------------------
// Generated from Python contracts
// Re-exported with aliases where the generated name differs from frontend usage
// ---------------------------------------------------------------------------

// Artifact contracts re-exported as view data aliases (frontend convention)
// Latent structure types
// Measurement structure types
// Causal design types
// Worker / extraction types
// Validation types
// Statistical model spec types
// Prior types
// LLM trace types
// Inference diagnostic types
export type {
  CausalEdge,
  Construct,
  DistributionFamily,
  EdgeSource,
  EnergyDiagnostics,
  EnergyHistogram,
  Indicator,
  IndicatorAuditContract as IndicatorAudit,
  IndicatorEmpiricalProfileContract as IndicatorEmpiricalProfile,
  IndicatorValidationContract as IndicatorValidation,
  InferenceMetadataContract as InferenceMetadata,
  LatentStructure,
  LikelihoodSource,
  LikelihoodSpec,
  LinkFunction,
  LLMTrace,
  LOODiagnostics,
  MCMCDiagnostics,
  MCMCParamDiagnostic,
  MeasurementStructure,
  ParameterConstraint,
  ParameterRole,
  ParameterSpec,
  PosteriorMarginal,
  PosteriorPair,
  PosteriorContract as PosteriorData,
  PPCOverlay,
  PPCResultContract as PPCResult,
  PPCTestStat,
  PPCWarning,
  PriorDistributionFamily,
  PriorProposal,
  PriorSource,
  RankHistogram,
  RankHistogramChain,
  RawDataContract as RawDataPersistedData,
  Role,
  SMCDiagnostics,
  BaselineReportContract as BaselineReportData,
  LatentStructureContract as LatentStructureData,
  MeasurementStructureContract as MeasurementStructureData,
  MeasurementsContract as MeasurementsPersistedData,
  ValidationReportContract as ValidationReportData,
  StatisticalModelSpec,
  StatisticalModelSpecContract as StatisticalModelSpecPersistedData,
  TemporalStatus,
  TraceChain,
  TraceData,
  TraceMessage,
  TraceUsage,
  TreatmentEffectContract as TreatmentEffect,
  ValidationIssueContract as ValidationIssue,
  WorkerStatusContract as WorkerStatus,
} from "./generated/models";

export type {
  EffectSummaryContract as EffectSummary,
  EffectTrajectoryPointContract as EffectTrajectoryPoint,
  LatentClampInput,
  ScenarioStartResultContract as ScenarioStartResult,
  SimulateScenarioResultContract as SimulateScenarioResult,
  SimulateScenarioToolResultContract as SimulateScenarioToolResult,
  BaselineReportVisualizationContract as BaselineReportVisualization,
  ToolErrorContract as ToolError,
} from "./generated/tool-results";

export type StatisticalModelSpecPersistedViewData = Omit<
  import("./generated/models").StatisticalModelSpecContract,
  "resolved_priors"
> & {
  resolved_priors: import("./generated/models").PriorProposal[];
};
export interface HistogramBin {
  binCenter: number;
  count: number;
}

export interface ModelSpecLikelihoodDiagnostics {
  variable: string;
  profile: import("./generated/models").IndicatorEmpiricalProfileContract | null;
  histogram: HistogramBin[];
}

export type StatisticalModelSpecData = StatisticalModelSpecPersistedViewData & {
  likelihood_diagnostics: {
    [k: string]: ModelSpecLikelihoodDiagnostics | undefined;
  };
};

export type MeasurementStructureViewData = import("./generated/models").MeasurementStructureContract & {
  causal_design: import("./causal-design").CausalDesign;
};

// Distribution catalog metadata (codegen'd from Python)
export type { ObservationHyperparameter } from "./generated/metadata";
export { OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION } from "./generated/metadata";
// Tool definitions (codegen'd from Python ToolContract)
export type { ToolDefinition } from "./generated/tools";
export { CONTEXT_TOOLS, INTERACTIVE_CONTEXTS } from "./generated/tools";

export interface ArtifactData<T = unknown> {
  artifactId: string;
  data: T;
  context: string;
}

// Named type aliases inlined in generated types but needed as standalone exports
export type ValidationSeverity = "error" | "warning" | "info";
export type CellStatus = "ok" | "warning" | "error";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";
export type MeasurementDtype = "continuous" | "binary" | "count" | "ordinal" | "categorical";

export interface RawDataDateRange {
  start: string;
  end: string;
}

export interface RawDataColumnDescription {
  name: string;
  dtype: string;
  description: string;
}

export interface RawDataData {
  llm_trace?: import("./generated/models").LLMTrace | null;
  n_records: number;
  n_columns: number;
  date_range: RawDataDateRange;
  sample: {
    [k: string]: (string | null) | undefined;
  }[];
  column_descriptions: RawDataColumnDescription[];
}

export interface ObservationRecord {
  indicator: string;
  value: number | boolean | string | null;
  anchor_time: string | null;
  support_kind?: string | null;
  summary_operator?: string | null;
  anchor_policy?: string | null;
  observation_window?: string | null;
  support_start?: string | null;
  support_end?: string | null;
}

export interface MeasurementsData {
  llm_trace?: import("./generated/models").LLMTrace | null;
  workers: import("./generated/models").WorkerStatusContract[];
  per_indicator_counts: {
    [k: string]: number | undefined;
  };
  combined_extractions_sample: ObservationRecord[];
}

export type AggregationFunction =
  | "mean"
  | "sum"
  | "min"
  | "max"
  | "std"
  | "var"
  | "last"
  | "first"
  | "count"
  | "median"
  | "p10"
  | "p25"
  | "p75"
  | "p90"
  | "p99"
  | "skew"
  | "kurtosis"
  | "iqr"
  | "range"
  | "cv"
  | "entropy"
  | "instability"
  | "trend"
  | "n_unique";
