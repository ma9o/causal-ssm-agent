import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  type PrefectStage4EventRecord,
  type Stage4BlockLastState,
  type Stage4ReplayState,
  EMPTY_STAGE4_REPLAY_STATE,
  STAGE4_EVENT_PREFIX,
  applyStage4Event,
  parseStage4Event,
} from "@/lib/stage4-runtime";
import type { Stage4Graph, Stage4Snapshot } from "@/lib/hooks/use-stage4-graph";
import { STAGES } from "@causal-ssm/api-types";
import { Stage4RunningView } from "./stage-4-running-content";
import { stageStoryDecorators } from "../stage-story-helpers";
import { StageStoryTemplate } from "../stage-story-template";
import { StoryStageLogView } from "../stage-story-log-stream";
import { useEffect, useState } from "react";

const stage = STAGES.find((s) => s.id === "stage-4")!;

const meta = {
  title: "Pipeline/Stages/4 – Model Specification",
  component: Stage4RunningView,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage4RunningView>;

export default meta;

// ---------------------------------------------------------------------------
// Mock Prefect event records — same wire format as production
// ---------------------------------------------------------------------------

/** Build a raw Prefect event record matching what `emit_stage4_graph_event` emits. */
function graphEvent(graph: Stage4Graph): PrefectStage4EventRecord {
  return {
    event: `${STAGE4_EVENT_PREFIX}graph`,
    occurred: new Date().toISOString(),
    payload: { stage_id: "stage-4", type: "graph", ...graph },
  };
}

/** Build a raw Prefect event record matching what `emit_stage4_snapshot_event` emits. */
function snapshotEvent(snapshot: Stage4Snapshot): PrefectStage4EventRecord {
  return {
    event: `${STAGE4_EVENT_PREFIX}snapshot`,
    occurred: new Date().toISOString(),
    payload: { stage_id: "stage-4", type: "snapshot", ...snapshot },
  };
}

/** Build a raw Prefect event record matching what `emit_stage4_block_transition_event` emits. */
function transitionEvent(transition: Stage4BlockLastState): PrefectStage4EventRecord {
  return {
    event: `${STAGE4_EVENT_PREFIX}block_transition`,
    occurred: new Date().toISOString(),
    payload: { stage_id: "stage-4", type: "block_transition", ...transition },
  };
}

// ---------------------------------------------------------------------------
// Static graph topology — scaled to the SMALLGOLDEN fixture
// ---------------------------------------------------------------------------

function titleize(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function chain(ids: string[], kind: "forward" | "phase_advance" | "repair_transition") {
  return ids.slice(0, -1).map((id, index) => ({
    from: id,
    to: ids[index + 1]!,
    kind,
  }));
}

const SMALLGOLDEN_INDICATORS = [
  "daily_search_count",
  "evening_search_count",
  "late_night_search_flag",
  "sleep_problem_search_count",
  "pre_sleep_stimulating_search_flag",
  "work_school_search_count",
  "stress_anxiety_search_count",
  "negative_mood_search_flag",
  "exercise_search_count",
  "caffeine_search_flag",
  "alcohol_search_flag",
  "fatigue_search_flag",
  "sleep_hygiene_search_flag",
  "social_media_minutes",
  "morning_alertness_rating",
  "heart_rate_variability",
  "meditation_minutes",
  "water_intake_glasses",
  "screen_brightness_flag",
  "nap_duration_minutes",
  "pain_severity_rating",
  "appetite_rating",
  "social_interaction_count",
  "outdoor_time_minutes",
] as const;

const SMALLGOLDEN_DYNAMICS = [
  "screen_time",
  "sleep_quality",
  "pre_sleep_arousal",
  "circadian_disruption",
  "daily_stress",
  "mood",
  "physical_activity",
  "caffeine_consumption",
  "alcohol_consumption",
  "daytime_fatigue",
  "sleep_hygiene",
] as const;

const SMALLGOLDEN_EFFECT_TARGETS = [
  "screen_time",
  "daily_stress",
  "physical_activity",
  "caffeine_consumption",
  "sleep_quality",
  "sleep_hygiene",
  "pre_sleep_arousal",
  "circadian_disruption",
  "mood",
  "alcohol_consumption",
  "daytime_fatigue",
] as const;

const SMALLGOLDEN_CORRELATIONS = [
  "cor0_alcohol_consumption_caffeine_consumption",
  "cor0_alcohol_consumption_screen_time",
  "cor0_alcohol_consumption_sleep_hygiene",
  "cor0_caffeine_consumption_physical_activity",
  "cor0_caffeine_consumption_screen_time",
  "cor0_caffeine_consumption_sleep_hygiene",
  "cor0_caffeine_consumption_sleep_quality",
  "cor0_physical_activity_screen_time",
  "cor0_physical_activity_sleep_quality",
  "cor0_pre_sleep_arousal_screen_time",
  "cor0_pre_sleep_arousal_sleep_hygiene",
  "cor0_pre_sleep_arousal_sleep_quality",
  "cor0_screen_time_sleep_hygiene",
  "cor0_screen_time_sleep_quality",
  "cor0_sleep_hygiene_sleep_quality",
] as const;

function correlationLabel(name: string): string {
  const body = name.replace(/^cor0_/, "");
  const constructs = [...SMALLGOLDEN_DYNAMICS].sort((left, right) => right.length - left.length);
  for (const right of constructs) {
    const suffix = `_${right}`;
    if (body.endsWith(suffix)) {
      const left = body.slice(0, -suffix.length);
      return `${titleize(left)} × ${titleize(right)}`;
    }
  }
  return titleize(body);
}

const MODEL_BLOCK_IDS = [
  ...SMALLGOLDEN_INDICATORS.map((name) => `indicator:${name}`),
  "loading:screen_time",
];

const PRIOR_BLOCK_IDS = [
  "measurement:screen_time",
  ...SMALLGOLDEN_DYNAMICS.map((name) => `dynamics:${name}`),
  ...SMALLGOLDEN_EFFECT_TARGETS.map((name) => `effects:${name}`),
  ...SMALLGOLDEN_CORRELATIONS.map((name) => `correlation:${name}`),
];

const ALL_ACCEPTABLE_BLOCK_IDS = [
  ...MODEL_BLOCK_IDS,
  "review:model_spec",
  ...PRIOR_BLOCK_IDS,
  "review:prior_system",
];

const LAST_PRIOR_BLOCK_ID = PRIOR_BLOCK_IDS[PRIOR_BLOCK_IDS.length - 1]!;

const GRAPH: Stage4Graph = {
  nodes: [
    ...SMALLGOLDEN_INDICATORS.map((name) => ({
      id: `indicator:${name}`,
      kind: "indicator_decision",
      label: titleize(name),
      phase: "model_decisions",
    })),
    {
      id: "loading:screen_time",
      kind: "loading_decision",
      label: "Screen Time",
      phase: "model_decisions",
    },
    { id: "__lock__", kind: "model_spec_lock", label: "Lock Model Spec", phase: "model_decisions" },
    {
      id: "review:model_spec",
      kind: "global_review",
      label: "Model Specification",
      phase: "global_review",
    },
    {
      id: "measurement:screen_time",
      kind: "measurement_prior",
      label: "Screen Time",
      phase: "prior_blocks",
    },
    ...SMALLGOLDEN_DYNAMICS.map((name) => ({
      id: `dynamics:${name}`,
      kind: "dynamics_prior",
      label: titleize(name),
      phase: "prior_blocks",
    })),
    ...SMALLGOLDEN_EFFECT_TARGETS.map((name) => ({
      id: `effects:${name}`,
      kind: "effect_prior",
      label: titleize(name),
      phase: "prior_blocks",
    })),
    ...SMALLGOLDEN_CORRELATIONS.map((name) => ({
      id: `correlation:${name}`,
      kind: "correlation_prior",
      label: correlationLabel(name),
      phase: "prior_blocks",
    })),
    {
      id: "__repair_barrier__",
      kind: "repair_barrier",
      label: "Validate Repair Scope",
      phase: "prior_blocks",
    },
    {
      id: "review:prior_system",
      kind: "global_prior_review",
      label: "Full Prior System",
      phase: "global_prior_review",
    },
    { id: "__done__", kind: "done", label: "Done", phase: "done" },
  ],
  edges: [
    ...chain(MODEL_BLOCK_IDS, "forward"),
    { from: MODEL_BLOCK_IDS[MODEL_BLOCK_IDS.length - 1]!, to: "__lock__", kind: "phase_advance" },
    { from: "__lock__", to: "review:model_spec", kind: "phase_advance" },
    { from: "review:model_spec", to: PRIOR_BLOCK_IDS[0]!, kind: "phase_advance" },
    ...chain(PRIOR_BLOCK_IDS, "forward"),
    { from: LAST_PRIOR_BLOCK_ID, to: "__repair_barrier__", kind: "repair_transition" },
    { from: LAST_PRIOR_BLOCK_ID, to: "review:prior_system", kind: "repair_transition" },
    { from: "__repair_barrier__", to: "review:prior_system", kind: "repair_transition" },
    { from: "__repair_barrier__", to: "__done__", kind: "repair_transition" },
    { from: "review:prior_system", to: "__done__", kind: "phase_advance" },
    { from: LAST_PRIOR_BLOCK_ID, to: "__done__", kind: "phase_advance" },
  ],
  phases: [
    { id: "model_decisions", label: "Model Decisions" },
    { id: "global_review", label: "Global Review" },
    { id: "prior_blocks", label: "Prior Elicitation" },
    { id: "global_prior_review", label: "Prior Review" },
    { id: "done", label: "Complete" },
  ],
};

// ---------------------------------------------------------------------------
// Snapshot timeline as raw Prefect events
// ---------------------------------------------------------------------------

function base(): Record<string, string> {
  const s: Record<string, string> = {};
  for (const n of GRAPH.nodes) {
    if (!n.id.startsWith("__")) {
      s[n.id] = n.id === "review:prior_system" ? "inactive" : "pending";
    }
  }
  return s;
}

function without(ids: readonly string[], removed: readonly string[]): string[] {
  const removedSet = new Set(removed);
  return ids.filter((id) => !removedSet.has(id));
}

function statusFor({
  accepted = [],
  reopened = [],
  overrides = {},
}: {
  accepted?: readonly string[];
  reopened?: readonly string[];
  overrides?: Record<string, string>;
}) {
  const next = base();
  for (const id of accepted) {
    next[id] = "accepted";
  }
  for (const id of reopened) {
    next[id] = "reopened";
  }
  return { ...next, ...overrides };
}

const b = base();
const MODEL_REVIEW_IDS = [...MODEL_BLOCK_IDS, "review:model_spec"];
const EARLY_MODEL_ACCEPTED = MODEL_BLOCK_IDS.slice(0, 5);
const MID_MODEL_ACCEPTED = MODEL_BLOCK_IDS.slice(0, 10);
const LATE_MODEL_ACCEPTED = MODEL_BLOCK_IDS.slice(0, 13);
const EARLY_DYNAMICS_ACCEPTED = [
  ...MODEL_REVIEW_IDS,
  "measurement:screen_time",
  ...SMALLGOLDEN_DYNAMICS.slice(0, 5).map((name) => `dynamics:${name}`),
];
const MID_PRIOR_ACCEPTED = [
  ...MODEL_REVIEW_IDS,
  "measurement:screen_time",
  ...SMALLGOLDEN_DYNAMICS.map((name) => `dynamics:${name}`),
  ...SMALLGOLDEN_EFFECT_TARGETS.slice(0, 4).map((name) => `effects:${name}`),
];
const EARLY_CORRELATION_ACCEPTED = [
  ...MODEL_REVIEW_IDS,
  ...PRIOR_BLOCK_IDS.slice(
    0,
    1 + SMALLGOLDEN_DYNAMICS.length + SMALLGOLDEN_EFFECT_TARGETS.length + 6,
  ),
];
const LATE_CORRELATION_ACCEPTED = [
  ...MODEL_REVIEW_IDS,
  ...PRIOR_BLOCK_IDS.slice(
    0,
    1 + SMALLGOLDEN_DYNAMICS.length + SMALLGOLDEN_EFFECT_TARGETS.length + 14,
  ),
];
const ALL_PRIOR_ACCEPTED = [...MODEL_REVIEW_IDS, ...PRIOR_BLOCK_IDS];
const REPAIR_SCOPE_IDS = [
  "dynamics:sleep_quality",
  "effects:sleep_quality",
  "correlation:cor0_screen_time_sleep_quality",
] as const;

/** The event timeline — first event is the graph, rest are snapshots. */
const EVENT_TIMELINE: PrefectStage4EventRecord[] = [
  graphEvent(GRAPH),
  snapshotEvent({
    cursor: { kind: "block", block_id: MODEL_BLOCK_IDS[0]! },
    block_status: b,
    model_spec_locked: false,
    repair_campaign: null,
    phase: "model_decisions",
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: MODEL_BLOCK_IDS[5]! },
    block_status: statusFor({ accepted: EARLY_MODEL_ACCEPTED }),
    model_spec_locked: false,
    repair_campaign: null,
    phase: "model_decisions",
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: MODEL_BLOCK_IDS[10]! },
    block_status: statusFor({ accepted: MID_MODEL_ACCEPTED }),
    model_spec_locked: false,
    repair_campaign: null,
    phase: "model_decisions",
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: "loading:screen_time" },
    block_status: statusFor({ accepted: LATE_MODEL_ACCEPTED }),
    model_spec_locked: false,
    repair_campaign: null,
    phase: "model_decisions",
  }),
  transitionEvent({
    block_id: "loading:screen_time",
    status: "accepted",
    detail_kind: "indicator_choice",
    variable: "screen_time",
    distribution: "gaussian",
    link: "identity",
    reasoning: "Continuous daily minutes are modeled on the identity scale.",
  }),
  snapshotEvent({
    cursor: { kind: "model_spec_lock" },
    block_status: statusFor({ accepted: MODEL_BLOCK_IDS }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "model_decisions",
  }),
  transitionEvent({
    block_id: "review:model_spec",
    status: "accepted",
    detail_kind: "review_approval",
    reasoning: "The likelihood and loading decisions are coherent enough to lock the model spec.",
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: "review:model_spec" },
    block_status: statusFor({ accepted: MODEL_BLOCK_IDS }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "global_review",
  }),
  transitionEvent({
    block_id: "measurement:screen_time",
    status: "accepted",
    detail_kind: "prior_bundle",
    parameter_names: ["lambda_screen_time_screen_time", "sigma_screen_time"],
    priors: [
      {
        parameter: "lambda_screen_time_screen_time",
        distribution: "HalfNormal",
        params: { sigma: 0.35 },
      },
      {
        parameter: "sigma_screen_time",
        distribution: "HalfNormal",
        params: { sigma: 0.6 },
      },
    ],
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: "measurement:screen_time" },
    block_status: statusFor({ accepted: MODEL_REVIEW_IDS }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "prior_blocks",
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: "dynamics:mood" },
    block_status: statusFor({ accepted: EARLY_DYNAMICS_ACCEPTED }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "prior_blocks",
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: "effects:sleep_quality" },
    block_status: statusFor({ accepted: MID_PRIOR_ACCEPTED }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "prior_blocks",
  }),
  snapshotEvent({
    cursor: {
      kind: "block",
      block_id: "correlation:cor0_caffeine_consumption_sleep_hygiene",
    },
    block_status: statusFor({ accepted: EARLY_CORRELATION_ACCEPTED }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "prior_blocks",
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: "correlation:cor0_sleep_hygiene_sleep_quality" },
    block_status: statusFor({ accepted: LATE_CORRELATION_ACCEPTED }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "prior_blocks",
  }),
  transitionEvent({
    block_id: "correlation:cor0_sleep_hygiene_sleep_quality",
    status: "accepted",
    detail_kind: "prior_bundle",
    parameter_names: ["cor0_sleep_hygiene_sleep_quality"],
    priors: [
      {
        parameter: "cor0_sleep_hygiene_sleep_quality",
        distribution: "Normal",
        params: { mu: 0, sigma: 0.2, lower: -1, upper: 1 },
      },
    ],
  }),
  ...REPAIR_SCOPE_IDS.map((blockId) =>
    transitionEvent({
      block_id: blockId,
      status: "reopened",
      detail_kind: "revision",
      reason: "Joint prior predictive checks showed the sleep row still drifts unrealistically.",
      scope_kind: "global_prior_consistency",
    }),
  ),
  snapshotEvent({
    cursor: { kind: "block", block_id: "dynamics:sleep_quality" },
    block_status: statusFor({
      accepted: without(ALL_PRIOR_ACCEPTED, REPAIR_SCOPE_IDS),
      reopened: REPAIR_SCOPE_IDS,
    }),
    model_spec_locked: true,
    repair_campaign: {
      scope_kind: "global_prior_consistency",
      scope_block_ids: [...REPAIR_SCOPE_IDS],
      completed_block_ids: [],
    },
    phase: "prior_blocks",
  }),
  snapshotEvent({
    cursor: { kind: "repair_barrier", scope_block_ids: [...REPAIR_SCOPE_IDS] },
    block_status: statusFor({ accepted: ALL_PRIOR_ACCEPTED }),
    model_spec_locked: true,
    repair_campaign: {
      scope_kind: "global_prior_consistency",
      scope_block_ids: [...REPAIR_SCOPE_IDS],
      completed_block_ids: [...REPAIR_SCOPE_IDS],
    },
    phase: "prior_blocks",
  }),
  snapshotEvent({
    cursor: { kind: "block", block_id: "review:prior_system" },
    block_status: statusFor({
      accepted: ALL_PRIOR_ACCEPTED,
      overrides: { "review:prior_system": "pending" },
    }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "global_prior_review",
  }),
  snapshotEvent({
    cursor: { kind: "done" },
    block_status: statusFor({ accepted: ALL_ACCEPTABLE_BLOCK_IDS }),
    model_spec_locked: true,
    repair_campaign: null,
    phase: "done",
  }),
];

// ---------------------------------------------------------------------------
// Animated story: replays raw events through the real parse → reduce pipeline
// ---------------------------------------------------------------------------

const STEP_INTERVAL_MS = 1200;

function AnimatedStage4() {
  const [state, setState] = useState<Stage4ReplayState>(EMPTY_STAGE4_REPLAY_STATE);

  useEffect(() => {
    // Reset state at the start of each loop
    let current = EMPTY_STAGE4_REPLAY_STATE;
    let i = 0;

    // Apply first event immediately (the graph event)
    const first = parseStage4Event(EVENT_TIMELINE[0]);
    if (first) current = applyStage4Event(current, first);
    i = 1;

    // Apply second event (initial snapshot) immediately too
    if (EVENT_TIMELINE[1]) {
      const second = parseStage4Event(EVENT_TIMELINE[1]);
      if (second) current = applyStage4Event(current, second);
      i = 2;
    }

    setState(current);

    const timer = setInterval(() => {
      if (i >= EVENT_TIMELINE.length) {
        // Loop: reset
        current = EMPTY_STAGE4_REPLAY_STATE;
        i = 0;
        const first = parseStage4Event(EVENT_TIMELINE[0]);
        if (first) current = applyStage4Event(current, first);
        i = 1;
        if (EVENT_TIMELINE[1]) {
          const second = parseStage4Event(EVENT_TIMELINE[1]);
          if (second) current = applyStage4Event(current, second);
          i = 2;
        }
        setState(current);
        return;
      }

      const raw = EVENT_TIMELINE[i];
      const parsed = parseStage4Event(raw);
      if (parsed) {
        current = applyStage4Event(current, parsed);
        setState(current);
      }
      i++;
    }, STEP_INTERVAL_MS);

    return () => clearInterval(timer);
  }, []);

  return (
    <Stage4RunningView
      graph={state.graph}
      snapshot={state.snapshot}
      lastBlockStateById={state.lastBlockStateById}
    />
  );
}

// ---------------------------------------------------------------------------
// Exported story
// ---------------------------------------------------------------------------

type Story = StoryObj<typeof meta>;

export const StateMachineReplay: Story = {
  args: { graph: GRAPH, snapshot: EVENT_TIMELINE[1]?.payload as unknown as Stage4Snapshot },
  render: () => (
    <StageStoryTemplate
      stage={stage}
      status="running"
      runningContent={<AnimatedStage4 />}
      logView={<StoryStageLogView storyId="stage-4-running-state-machine" status="running" />}
    />
  ),
  parameters: {
    docs: {
      description: {
        story:
          "Replays a SMALLGOLDEN-scale Stage 4 graph through the real Prefect event parser and reducer, using the same frontier shape as the saved SMALLGOLDEN fixture rather than a toy model.",
      },
    },
  },
};
