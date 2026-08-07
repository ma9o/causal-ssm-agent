import type {
  BaselineReportData,
  CausalEdge,
  Construct,
  Indicator,
  IndicatorAudit,
  MeasurementsData,
  PPCOverlay,
  PPCTestStat,
  PPCWarning,
  RawDataData,
  ValidationReportData,
} from "@nof1-causal-lab/api-types";
import {
  buildDevMockMessages,
  makeMockSimulate,
  synthesizeMockScenarios,
} from "@/components/dag/interactive/dev-mock-scenario";
import type { ConstructStatus } from "@/components/dag/structure-dag";
import { buildBaselineReportScenarios } from "@/components/pipeline/output-views/baseline-report-scenarios";

export type WorkspaceLens = "data" | "model" | "split";

export type ArtifactMaterialization =
  | "structure"
  | "measurement"
  | "identified"
  | "fitted"
  | "simulated";

export const MODEL_NODE_IDS = [
  "chronotype",
  "work_demands",
  "screen_time",
  "arousal",
  "sleep_quality",
] as const;

export type ModelNodeId = (typeof MODEL_NODE_IDS)[number];

export type WorkspaceLayerId =
  | "model.structure"
  | "model.measurement"
  | "model.identification"
  | "model.dynamics"
  | "model.posterior"
  | "model.simulation"
  | "data.source"
  | "data.mapping"
  | "data.observations"
  | "data.quality"
  | "data.fit";

export interface WorkspaceLayer {
  id: WorkspaceLayerId;
  label: string;
  artifact: string;
  minimum: ArtifactMaterialization;
  description: string;
}

export interface PrototypeNodeMeta {
  label: string;
  eyebrow: string;
  description: string;
  kind: "confounder" | "known_input" | "treatment" | "mediator" | "outcome";
  observationModel?: string;
  posterior?: string;
}

export const MATERIALIZATION_ORDER: ArtifactMaterialization[] = [
  "structure",
  "measurement",
  "identified",
  "fitted",
  "simulated",
];

export const MATERIALIZATION_META: Record<
  ArtifactMaterialization,
  { label: string; artifacts: number; summary: string }
> = {
  structure: {
    label: "Structure available",
    artifacts: 3,
    summary: "Raw source and theoretical model",
  },
  measurement: {
    label: "Measurement mapped",
    artifacts: 5,
    summary: "Indicators attached to constructs",
  },
  identified: {
    label: "Data validated",
    artifacts: 9,
    summary: "Identification and indicator panel",
  },
  fitted: {
    label: "Model fitted",
    artifacts: 12,
    summary: "Posterior and predictive checks",
  },
  simulated: {
    label: "Analysis current",
    artifacts: 14,
    summary: "Interventions and saved scenarios",
  },
};

export const MODEL_LAYERS: WorkspaceLayer[] = [
  {
    id: "model.structure",
    label: "Causal structure",
    artifact: "latent_structure",
    minimum: "structure",
    description: "The persistent theory DAG",
  },
  {
    id: "model.measurement",
    label: "Measurement",
    artifact: "measurement_structure",
    minimum: "measurement",
    description: "Indicators attached to latent constructs",
  },
  {
    id: "model.identification",
    label: "Identification",
    artifact: "identification_report",
    minimum: "identified",
    description: "Estimand support and marginalized variables",
  },
  {
    id: "model.dynamics",
    label: "Dynamics",
    artifact: "statistical_model_spec",
    minimum: "fitted",
    description: "Nonlinear drift and observation semantics",
  },
  {
    id: "model.posterior",
    label: "Posterior",
    artifact: "posterior",
    minimum: "fitted",
    description: "Estimated trajectories and uncertainty",
  },
  {
    id: "model.simulation",
    label: "Simulation",
    artifact: "baseline_report",
    minimum: "simulated",
    description: "Intervention trajectories and controls",
  },
];

export const DATA_LAYERS: WorkspaceLayer[] = [
  {
    id: "data.source",
    label: "Source records",
    artifact: "raw_data",
    minimum: "structure",
    description: "Original events and files",
  },
  {
    id: "data.mapping",
    label: "Indicator mapping",
    artifact: "measurement_structure",
    minimum: "measurement",
    description: "Planned indicators and observation windows",
  },
  {
    id: "data.observations",
    label: "Observations",
    artifact: "panel",
    minimum: "identified",
    description: "Canonical irregular indicator panel",
  },
  {
    id: "data.quality",
    label: "Data quality",
    artifact: "validation_report",
    minimum: "identified",
    description: "Missingness, support, and validation findings",
  },
  {
    id: "data.fit",
    label: "Model fit",
    artifact: "posterior",
    minimum: "fitted",
    description: "Posterior predictive envelopes",
  },
];

function construct(
  name: ModelNodeId,
  description: string,
  role: Construct["role"],
  options: { outcome?: boolean; invariant?: boolean } = {},
): Construct {
  return {
    name,
    description,
    role,
    is_outcome: options.outcome ?? false,
    temporal_status: options.invariant ? "time_invariant" : "time_varying",
  };
}

function edge(cause: ModelNodeId, effect: ModelNodeId): CausalEdge {
  return {
    cause,
    effect,
    description: `${cause.replaceAll("_", " ")} influences ${effect.replaceAll("_", " ")}`,
    lagged: false,
    sources: [],
  };
}

function indicator(
  name: string,
  constructName: ModelNodeId,
  dtype: Indicator["measurement_dtype"],
  howToMeasure: string,
  aggregation: "mean" | "sum",
  observationWindow: string,
): Indicator {
  return {
    name,
    construct_name: constructName,
    how_to_measure: howToMeasure,
    construct_polarity: "positive",
    measurement_dtype: dtype,
    aggregation,
    source_columns: [],
    extraction_mode: "computed",
    support_kind: "interval",
    summary_operator: aggregation,
    anchor_policy: "support_end",
    observation_window: observationWindow,
  };
}

export const PROTOTYPE_CONSTRUCTS: Construct[] = [
  construct(
    "chronotype",
    "Stable sleep timing preference and household routines that influence both device use and sleep quality.",
    "exogenous",
    { invariant: true },
  ),
  construct(
    "work_demands",
    "Observed work intensity and late meetings that affect evening exposure and arousal.",
    "exogenous",
  ),
  construct(
    "screen_time",
    "Phone and tablet exposure during the three hours before intended sleep onset.",
    "endogenous",
  ),
  construct("arousal", "Cognitive and physiological activation close to bedtime.", "endogenous"),
  construct(
    "sleep_quality",
    "Nightly latent sleep quality reflected by wearable efficiency and morning self-report.",
    "endogenous",
    { outcome: true },
  ),
];

export const PROTOTYPE_EDGES: CausalEdge[] = [
  edge("chronotype", "screen_time"),
  edge("chronotype", "sleep_quality"),
  edge("work_demands", "screen_time"),
  edge("work_demands", "arousal"),
  edge("screen_time", "arousal"),
  edge("screen_time", "sleep_quality"),
  edge("arousal", "sleep_quality"),
];

export const PROTOTYPE_INDICATORS: Indicator[] = [
  indicator(
    "calendar_load",
    "work_demands",
    "continuous",
    "Hours marked busy on the participant's calendar.",
    "sum",
    "local calendar day",
  ),
  indicator(
    "screen_minutes",
    "screen_time",
    "continuous",
    "Foreground device minutes in the three hours before intended bedtime.",
    "sum",
    "trailing 3 hours",
  ),
  indicator(
    "bedtime_unlocks",
    "screen_time",
    "count",
    "Device unlocks in the three hours before intended bedtime.",
    "sum",
    "trailing 3 hours",
  ),
  indicator(
    "arousal_score",
    "arousal",
    "ordinal",
    "Evening EMA response to difficulty winding down, scored 0–4.",
    "mean",
    "final evening prompt",
  ),
  indicator(
    "sleep_efficiency",
    "sleep_quality",
    "continuous",
    "Wearable minutes asleep divided by minutes in bed.",
    "mean",
    "overnight sleep episode",
  ),
  indicator(
    "subjective_sleep",
    "sleep_quality",
    "ordinal",
    "Morning rating of overall sleep quality, scored 1–5.",
    "mean",
    "first morning prompt",
  ),
];

export const PROTOTYPE_NODE_STATUSES: Record<string, ConstructStatus> = {
  chronotype: "marginalized",
  work_demands: "observed",
  screen_time: "observed",
  arousal: "observed",
  sleep_quality: "observed",
};

export const PROTOTYPE_NODE_META: Record<ModelNodeId, PrototypeNodeMeta> = {
  chronotype: {
    label: "Chronotype & routines",
    eyebrow: "Latent confounder · invariant",
    description: PROTOTYPE_CONSTRUCTS[0].description,
    kind: "confounder",
  },
  work_demands: {
    label: "Work demands",
    eyebrow: "Known input · time-varying",
    description: PROTOTYPE_CONSTRUCTS[1].description,
    kind: "known_input",
    observationModel: "Observed exogenous input · held during forward simulation",
  },
  screen_time: {
    label: "Evening screen time",
    eyebrow: "Treatment · time-varying",
    description: PROTOTYPE_CONSTRUCTS[2].description,
    kind: "treatment",
    observationModel: "Gamma observation model · log link",
    posterior: "Identified intervention target",
  },
  arousal: {
    label: "Pre-sleep arousal",
    eyebrow: "Latent mediator · time-varying",
    description: PROTOTYPE_CONSTRUCTS[3].description,
    kind: "mediator",
    observationModel: "Ordered-logistic observation model",
    posterior: "Positive screen-time contribution",
  },
  sleep_quality: {
    label: "Sleep quality",
    eyebrow: "Outcome · time-varying",
    description: PROTOTYPE_CONSTRUCTS[4].description,
    kind: "outcome",
    observationModel: "Beta and ordered-logistic observation models",
    posterior: "Intervention trajectory shown with uncertainty",
  },
};

export const PROTOTYPE_RAW_DATA: RawDataData = {
  n_records: 1922,
  n_columns: 5,
  date_range: { start: "2026-04-08T00:00:00Z", end: "2026-05-17T23:59:59Z" },
  sample: [
    {
      timestamp: "2026-04-08 18:00",
      source: "calendar",
      event: "busy_hours",
      value: "5.0",
      unit: "hours",
    },
    {
      timestamp: "2026-04-08 21:14",
      source: "screen-events",
      event: "app_foreground",
      value: "14.4",
      unit: "minutes",
    },
    {
      timestamp: "2026-04-08 22:48",
      source: "ema-responses",
      event: "arousal",
      value: "1",
      unit: "0–4",
    },
    {
      timestamp: "2026-04-08 23:31",
      source: "wearable-sleep",
      event: "sleep_start",
      value: "23:31",
      unit: null,
    },
    {
      timestamp: "2026-04-09 07:42",
      source: "wearable-sleep",
      event: "sleep_end",
      value: "07:42",
      unit: null,
    },
    {
      timestamp: "2026-04-09 08:03",
      source: "ema-responses",
      event: "subjective_sleep",
      value: "4",
      unit: "1–5",
    },
  ],
  column_descriptions: [
    { name: "timestamp", dtype: "datetime", description: "Source-local event timestamp" },
    { name: "source", dtype: "string", description: "Originating source stream" },
    { name: "event", dtype: "string", description: "Source-native event name" },
    { name: "value", dtype: "string", description: "Unnormalized source value" },
    { name: "unit", dtype: "string", description: "Source-native unit when present" },
  ],
};

export const PROTOTYPE_MEASUREMENTS: MeasurementsData = {
  workers: [
    { worker_id: 1, status: "completed", n_extractions: 40, n_windows: 40 },
    { worker_id: 2, status: "completed", n_extractions: 78, n_windows: 80 },
    { worker_id: 3, status: "completed", n_extractions: 37, n_windows: 40 },
  ],
  per_indicator_counts: {
    calendar_load: 40,
    screen_minutes: 40,
    bedtime_unlocks: 38,
    arousal_score: 37,
    sleep_efficiency: 40,
    subjective_sleep: 38,
  },
  combined_extractions_sample: [
    {
      indicator: "calendar_load",
      value: 5,
      anchor_time: "2026-04-08T23:59:59Z",
      support_kind: "interval",
      summary_operator: "sum",
      anchor_policy: "support_end",
      observation_window: "local calendar day",
      support_start: "2026-04-08T00:00:00Z",
      support_end: "2026-04-08T23:59:59Z",
    },
    {
      indicator: "screen_minutes",
      value: 42,
      anchor_time: "2026-04-08T23:00:00Z",
      support_kind: "interval",
      summary_operator: "sum",
      anchor_policy: "support_end",
      observation_window: "trailing 3 hours",
      support_start: "2026-04-08T20:00:00Z",
      support_end: "2026-04-08T23:00:00Z",
    },
    {
      indicator: "bedtime_unlocks",
      value: 8,
      anchor_time: "2026-04-08T23:00:00Z",
      support_kind: "interval",
      summary_operator: "sum",
      anchor_policy: "support_end",
      observation_window: "trailing 3 hours",
      support_start: "2026-04-08T20:00:00Z",
      support_end: "2026-04-08T23:00:00Z",
    },
    {
      indicator: "arousal_score",
      value: 1,
      anchor_time: "2026-04-08T22:48:00Z",
      support_kind: "point",
      summary_operator: "mean",
      anchor_policy: "support_end",
      observation_window: "final evening prompt",
    },
    {
      indicator: "sleep_efficiency",
      value: 0.88,
      anchor_time: "2026-04-09T07:42:00Z",
      support_kind: "interval",
      summary_operator: "mean",
      anchor_policy: "support_end",
      observation_window: "overnight sleep episode",
      support_start: "2026-04-08T23:31:00Z",
      support_end: "2026-04-09T07:42:00Z",
    },
    {
      indicator: "subjective_sleep",
      value: 4,
      anchor_time: "2026-04-09T08:03:00Z",
      support_kind: "point",
      summary_operator: "mean",
      anchor_policy: "support_end",
      observation_window: "first morning prompt",
    },
  ],
};

function audit(
  nObs: number,
  mean: number,
  variance: number,
  coverage: number,
  maxGap: number,
  issue?: { issue_type: string; message: string },
): IndicatorAudit {
  return {
    profile: {
      measurement_dtype: null,
      n_obs: nObs,
      mean,
      std: null,
      min: null,
      max: null,
      q25: null,
      q50: null,
      q75: null,
      variance,
      time_coverage_ratio: coverage,
      max_gap_ratio: maxGap,
      dtype_violations: 0,
      duplicate_pct: 0,
      arithmetic_sequence_detected: false,
      n_unparseable_timestamps: 0,
      zero_fraction: null,
      is_nonnegative: true,
      is_unit_interval: null,
      looks_integer_valued: null,
      variance_to_mean_ratio: null,
    },
    validation: {
      issues: issue ? [{ ...issue, severity: "warning" }] : [],
      checks: {
        n_obs: "ok",
        variance: "ok",
        n_unparseable_timestamps: "ok",
        time_coverage_ratio: "ok",
        max_gap_ratio: issue ? "warning" : "ok",
        dtype_violations: "ok",
        duplicate_pct: "ok",
        arithmetic_sequence_detected: "ok",
      },
    },
  };
}

export const PROTOTYPE_VALIDATION_REPORT: ValidationReportData = {
  is_valid: true,
  dataset_issues: [],
  indicators: {
    calendar_load: audit(40, 5.95, 5.1, 1, 0.03),
    screen_minutes: audit(40, 59.3, 164.2, 1, 0.03),
    bedtime_unlocks: audit(38, 8.2, 9.7, 0.95, 0.08),
    arousal_score: audit(37, 2.08, 0.64, 0.93, 0.15, {
      issue_type: "support_gap",
      message: "Three consecutive evening prompts are missing.",
    }),
    sleep_efficiency: audit(40, 0.84, 0.003, 1, 0.03),
    subjective_sleep: audit(38, 3.42, 0.71, 0.95, 0.08),
  },
};

export const PROTOTYPE_PPC_WARNINGS: PPCWarning[] = [
  {
    variable: "screen_minutes",
    check_type: "calibration",
    message: "Observed coverage is consistent with the posterior predictive interval.",
    value: 0.94,
    passed: true,
  },
  {
    variable: "screen_minutes",
    check_type: "autocorrelation",
    message: "Residual lag-1 autocorrelation is small.",
    value: 0.08,
    passed: true,
  },
  {
    variable: "screen_minutes",
    check_type: "variance",
    message: "Replicated and observed scales agree.",
    value: 1.05,
    passed: true,
  },
  {
    variable: "sleep_efficiency",
    check_type: "calibration",
    message: "Observed coverage is consistent with the posterior predictive interval.",
    value: 0.92,
    passed: true,
  },
  {
    variable: "sleep_efficiency",
    check_type: "autocorrelation",
    message: "A small amount of residual temporal structure remains.",
    value: 0.21,
    passed: true,
  },
  {
    variable: "sleep_efficiency",
    check_type: "variance",
    message: "Replicated and observed scales agree.",
    value: 0.97,
    passed: true,
  },
];

export const PROTOTYPE_PPC_TEST_STATS: PPCTestStat[] = [
  {
    variable: "screen_minutes",
    stat_name: "mean",
    observed_value: 59.3,
    rep_values: [53, 55, 56, 57, 58, 59, 59.5, 60, 61, 62, 64, 66],
  },
  {
    variable: "screen_minutes",
    stat_name: "sd",
    observed_value: 12.8,
    rep_values: [10.4, 11.1, 11.7, 12, 12.3, 12.7, 13, 13.2, 13.6, 14, 14.4, 15],
  },
  {
    variable: "sleep_efficiency",
    stat_name: "mean",
    observed_value: 0.84,
    rep_values: [0.8, 0.81, 0.82, 0.83, 0.835, 0.84, 0.845, 0.85, 0.86, 0.87],
  },
  {
    variable: "sleep_efficiency",
    stat_name: "sd",
    observed_value: 0.055,
    rep_values: [0.04, 0.045, 0.048, 0.05, 0.053, 0.056, 0.059, 0.062, 0.066, 0.07],
  },
];

export const PROTOTYPE_PPC_OVERLAYS: PPCOverlay[] = [
  {
    variable: "screen_minutes",
    observed: [42, 51, 38, 66, 73, 58, 62, 49, 81, 76, 54, 47],
    q025: [28, 31, 33, 36, 39, 40, 39, 37, 40, 42, 39, 37],
    q25: [39, 41, 43, 47, 50, 51, 50, 48, 52, 54, 51, 49],
    median: [46, 48, 50, 55, 60, 61, 59, 58, 64, 66, 62, 60],
    q75: [54, 56, 58, 63, 68, 69, 67, 66, 72, 74, 70, 68],
    q975: [67, 70, 72, 77, 82, 84, 81, 80, 86, 89, 84, 82],
    spaghetti_draws: [],
  },
  {
    variable: "sleep_efficiency",
    observed: [0.88, 0.85, 0.9, 0.82, 0.79, 0.84, 0.83, 0.89, 0.76, 0.78, 0.85, 0.87],
    q025: [0.76, 0.75, 0.76, 0.74, 0.72, 0.72, 0.73, 0.74, 0.71, 0.7, 0.72, 0.73],
    q25: [0.82, 0.81, 0.81, 0.79, 0.77, 0.77, 0.78, 0.79, 0.76, 0.75, 0.77, 0.78],
    median: [0.87, 0.86, 0.86, 0.84, 0.82, 0.82, 0.83, 0.84, 0.81, 0.8, 0.82, 0.83],
    q75: [0.91, 0.9, 0.9, 0.88, 0.86, 0.86, 0.87, 0.88, 0.85, 0.84, 0.86, 0.87],
    q975: [0.95, 0.94, 0.94, 0.92, 0.9, 0.9, 0.91, 0.92, 0.89, 0.88, 0.9, 0.91],
    spaghetti_draws: [],
  },
];

type PrototypeSavedScenario = NonNullable<BaselineReportData["saved_scenarios"]>[number] & {
  clamps: Array<{ variable: string; value: number; from_day: number }>;
};

const PROTOTYPE_SAVED_SCENARIOS: PrototypeSavedScenario[] = [
  {
    label: "Less evening screen time",
    query: "What if I reduce evening screen time from day 14?",
    summary:
      "Clamp evening screen exposure below its fitted reference from day 14 and follow the change through arousal to sleep quality.",
    clamps: [{ variable: "screen_time", value: 0.25, from_day: 14 }],
  },
];

const synthesizedScenarios = synthesizeMockScenarios(
  PROTOTYPE_CONSTRUCTS,
  PROTOTYPE_EDGES,
  PROTOTYPE_INDICATORS,
  "sleep_quality",
  PROTOTYPE_SAVED_SCENARIOS,
);

export const PROTOTYPE_MOCK_SCENARIOS = {
  ...synthesizedScenarios,
  baseline: {
    ...synthesizedScenarios.baseline,
    blurb:
      "**No intervention — fitted reference.** The model evolves from the same fitted initial state without a clamp. Saved interventions are displayed as deviations from this reference world.",
  },
};

export const PROTOTYPE_SCENARIOS = buildBaselineReportScenarios({
  extraMessages: buildDevMockMessages(PROTOTYPE_MOCK_SCENARIOS),
});

export const PROTOTYPE_SIMULATE = makeMockSimulate(PROTOTYPE_MOCK_SCENARIOS.baseline.result);

export function isModelNodeId(value: string): value is ModelNodeId {
  return MODEL_NODE_IDS.includes(value as ModelNodeId);
}

export function isMaterialized(
  materialization: ArtifactMaterialization,
  minimum: ArtifactMaterialization,
): boolean {
  return MATERIALIZATION_ORDER.indexOf(materialization) >= MATERIALIZATION_ORDER.indexOf(minimum);
}
