// ---------------------------------------------------------------------------
// Hand-written (frontend-only) — not generated from Python
// ---------------------------------------------------------------------------

export type { PipelineRun, RunStatus, StageState, StageStatus } from "./run";
export type { StageId, StageMeta } from "./stages";
export { STAGES, STAGE_IDS } from "./stages";

// ---------------------------------------------------------------------------
// Generated from Python contracts
// Re-exported with aliases where the generated name differs from frontend usage
// ---------------------------------------------------------------------------

// Stage contracts as Stage*Data aliases (frontend convention)
export type { Stage0Contract as Stage0Data } from "./generated/models";
export type { Stage1AContract as Stage1aData } from "./generated/models";
export type { Stage1BContract as Stage1bData } from "./generated/models";
export type { Stage2Contract as Stage2Data } from "./generated/models";
export type { Stage3Contract as Stage3Data } from "./generated/models";
export type { Stage4Contract as Stage4Data } from "./generated/models";
export type { Stage4BContract as Stage4bData } from "./generated/models";
export type { Stage5Contract as Stage5Data } from "./generated/models";
export type { Stage6Contract as Stage6Data } from "./generated/models";

// Latent model types
export type {
  Construct,
  CausalEdge,
  LatentModel,
  Role,
  TemporalStatus,
} from "./generated/models";

// Measurement model types
export type {
  Indicator,
  MeasurementModel,
} from "./generated/models";

// Causal spec types
export type {
  CausalSpec,
  IdentifiabilityStatus,
  IdentifiedTreatmentStatus,
  NonIdentifiableTreatmentStatus,
} from "./generated/models";

// Worker / extraction types
export type { ExtractionContract as Extraction } from "./generated/models";
export type { WorkerStatusContract as WorkerStatus } from "./generated/models";

// Validation types
export type { ValidationReportContract as ValidationReport } from "./generated/models";
export type { ValidationIssueContract as ValidationIssue } from "./generated/models";
export type { IndicatorHealthContract as IndicatorHealth } from "./generated/models";

// Model spec types
export type {
  ModelSpec,
  LikelihoodSpec,
  LikelihoodSource,
  ParameterSpec,
  DistributionFamily,
  LinkFunction,
  ParameterRole,
  ParameterConstraint,
} from "./generated/models";

// Prior types
export type { PriorProposal, PriorSource } from "./generated/models";

// Parametric ID types
export type {
  ParametricIdResult,
  TRuleResult,
  ParameterIdentification,
  SensitivityAnalysisResult,
  SensitivityEntry,
} from "./generated/models";

// Rao-Blackwellization partition types
export type { RBPartitionResult, RBVariable } from "./generated/models";

// LLM trace types
export type { LLMTrace, TraceMessage, TraceUsage } from "./generated/models";

// Live trace (partial stage result written mid-run)
export type { PartialStageResult, LiveMetadata } from "./generated/models";

// Tool definitions (codegen'd from Python ToolContract)
export type { ToolDefinition } from "./generated/tools";
export { STAGE_TOOLS, INTERACTIVE_STAGES } from "./generated/tools";

// Inference diagnostic types
export type { TreatmentEffectContract as TreatmentEffect } from "./generated/models";
export type { PowerScalingResultContract as PowerScalingResult } from "./generated/models";
export type { PPCWarning, PPCOverlay, PPCTestStat } from "./generated/models";
export type { PPCResultContract as PPCResult } from "./generated/models";
export type { InferenceMetadataContract as InferenceMetadata } from "./generated/models";
export type {
  MCMCParamDiagnostic,
  MCMCDiagnostics,
  SVIDiagnostics,
  LOODiagnostics,
} from "./generated/models";
export type { TraceData, TraceChain } from "./generated/models";
export type { RankHistogram, RankHistogramChain } from "./generated/models";
export type { EnergyHistogram, EnergyDiagnostics } from "./generated/models";
export type { PosteriorMarginal, PosteriorPair } from "./generated/models";

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
export type ParameterClassification = "identified" | "practically_unidentifiable" | "structurally_unidentifiable";
export type ValidationSeverity = "error" | "warning" | "info";
export type CellStatus = "ok" | "warning" | "error";
export type PowerScalingDiagnosis = "prior_dominated" | "well_identified" | "prior_data_conflict";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";
export type StageOutcome = "success" | "warn" | "fail";
export type MeasurementDtype = "continuous" | "binary" | "count" | "ordinal" | "categorical";
export type AggregationFunction =
  | "mean" | "sum" | "min" | "max" | "std" | "var" | "last" | "first"
  | "count" | "median" | "p10" | "p25" | "p75" | "p90" | "p99"
  | "skew" | "kurtosis" | "iqr" | "range" | "cv" | "entropy"
  | "instability" | "trend" | "n_unique";
