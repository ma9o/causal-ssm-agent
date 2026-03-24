// ---------------------------------------------------------------------------
// Hand-written (frontend-only) — not generated from Python
// ---------------------------------------------------------------------------

export type { PipelineRun, RunStatus, StageState, StageStatus } from "./run";
export type { StageId, StageMeta } from "./stages";
export { STAGE_IDS, STAGES } from "./stages";

// ---------------------------------------------------------------------------
// Generated from Python contracts
// Re-exported with aliases where the generated name differs from frontend usage
// ---------------------------------------------------------------------------

// Stage contracts as Stage*Data aliases (frontend convention)
// Latent model types
// Measurement model types
// Causal spec types
// Worker / extraction types
// Validation types
// Model spec types
// Prior types
// Parametric ID types
// Inference structure types
// LLM trace types
// Inference diagnostic types
export type {
  CausalEdge,
  CausalSpec,
  Construct,
  DistributionFamily,
  EnergyDiagnostics,
  EnergyHistogram,
  ExtractionContract as Extraction,
  IdentifiabilityStatus,
  IdentifiedTreatmentStatus,
  Indicator,
  IndicatorAuditContract as IndicatorAudit,
  IndicatorEmpiricalProfileContract as IndicatorEmpiricalProfile,
  IndicatorValidationContract as IndicatorValidation,
  InferenceMetadataContract as InferenceMetadata,
  LatentModel,
  LikelihoodSource,
  LikelihoodSpec,
  LinkFunction,
  LLMTrace,
  LOODiagnostics,
  MCMCDiagnostics,
  MCMCParamDiagnostic,
  MeasurementModel,
  ModelSpec,
  NonIdentifiableTreatmentStatus,
  ParameterConstraint,
  ParameterIdentification,
  ParameterRole,
  ParameterSpec,
  ParametricIdResult,
  InferenceStructureResult,
  InferenceStructureVariable,
  FirstPassRBResult,
  PosteriorMarginal,
  PosteriorPair,
  PowerScalingResultContract as PowerScalingResult,
  PPCOverlay,
  PPCResultContract as PPCResult,
  PPCTestStat,
  PPCWarning,
  PriorProposal,
  PriorSource,
  RankHistogram,
  RankHistogramChain,
  Role,
  SensitivityAnalysisResult,
  SensitivityEntry,
  SMCDiagnostics,
  Stage0Contract as Stage0PersistedData,
  Stage1AContract as Stage1aData,
  Stage1BContract as Stage1bData,
  Stage2Contract as Stage2Data,
  Stage3Contract as Stage3Data,
  Stage4BContract as Stage4bData,
  Stage4Contract as Stage4Data,
  Stage5AContract as Stage5aData,
  Stage5BContract as Stage5bData,
  Stage6Contract as Stage6Data,
  SVIDiagnostics,
  TemporalStatus,
  TRuleResult,
  TraceChain,
  TraceData,
  TraceMessage,
  TraceUsage,
  TreatmentEffectContract as TreatmentEffect,
  ValidationIssueContract as ValidationIssue,
  WorkerStatusContract as WorkerStatus,
} from "./generated/models";
// Tool definitions (codegen'd from Python ToolContract)
export type { ToolDefinition } from "./generated/tools";
export { INTERACTIVE_STAGES, STAGE_TOOLS } from "./generated/tools";

// ---------------------------------------------------------------------------
// Hand-written types — not in Python contracts but used by frontend
// ---------------------------------------------------------------------------

export interface GateOverride {
  reason: string;
}

export interface StageData<T = unknown> {
  stage: string;
  data: T;
  context: string;
}

// Named type aliases inlined in generated types but needed as standalone exports
export type ParameterClassification =
  | "identified"
  | "practically_unidentifiable"
  | "structurally_unidentifiable";
export type ValidationSeverity = "error" | "warning" | "info";
export type CellStatus = "ok" | "warning" | "error";
export type PowerScalingDiagnosis = "prior_dominated" | "well_identified" | "prior_data_conflict";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";
export type StageOutcome = "success" | "warn" | "fail";
export type MeasurementDtype = "continuous" | "binary" | "count" | "ordinal" | "categorical";

export interface Stage0DateRange {
  start: string;
  end: string;
}

export interface Stage0ColumnDescription {
  name: string;
  dtype: string;
  description: string;
}

export interface Stage0Data {
  outcome: StageOutcome;
  llm_trace?: import("./generated/models").LLMTrace | null;
  n_records: number;
  n_columns: number;
  date_range: Stage0DateRange;
  sample: {
    [k: string]: (string | null) | undefined;
  }[];
  column_descriptions: Stage0ColumnDescription[];
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
