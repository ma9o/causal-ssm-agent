/**
 * Ideal materialized Stage 6 artifact for development & stories.
 *
 * Real persisted Stage 6 traces in the seed workspace (DEMO) carry
 * only the baseline ranking commentary — no `simulate` tool calls — so there is
 * no representative data for developing the simulation viewer. This fixture fills
 * that gap: it composes the REAL DEMO baselines (`intervention_results`),
 * `final_summary`, and `saved_scenarios` with a set of synthesized `simulate`
 * results materialized into the `llm_trace` in their **persisted form** — i.e.
 * each tool result is a JSON *string* in `tool_result`, exactly as a reloaded
 * workspace would store it. Driving the viewer through `buildStage6Scenarios`
 * over this trace therefore exercises the real production path, including the
 * string→object coercion for trace-sourced simulations.
 *
 * Scenarios exercised:
 *  - no-intervention baseline (clamps: [], action ≡ reference) → affective_state
 *  - baseline-start shift, trajectory, full visualization (serotonergic → affective_state) [reused]
 *  - abducted-start counterfactual, trajectory, full visualization (adherence → affective_state) [reused]
 *  - baseline-start set mode, trajectory (sleep_quality := 0.5 → affective_state)
 *  - baseline-start shift targeting a NON-global outcome (serotonergic → sleep_quality)
 *
 * The 8 baseline TreatmentEffects from the real DEMO run feed the collapsed
 * "All treatments" ranking table (not the scenario carousel).
 */
import type {
  LatentClampInput,
  LLMTrace,
  SimulateScenarioResult,
  Stage6Data,
  TraceMessage,
} from "@nof1-causal-lab/api-types";
import demoStage6 from "../../../../../../data/DEMO/run/stage-6.json";
import type { Stage6SimulationResult } from "../intervention-dag-types";
import { constructs, edges, indicators } from "./dag-base-fixtures";
import {
  counterfactualResult,
  edgePosteriors,
  interventionResult,
} from "./intervention-dag-fixture";

export { constructs, edgePosteriors, edges, indicators };

// ── small trajectory generators (mirror the rung-2 fixture shape) ──────────

function dailyGrid(length: number): number[] {
  return Array.from({ length }, (_, index) => index + 1);
}

function expApproach(steadyState: number, tau: number, days: number[]): number[] {
  return days.map((day) => +(steadyState * (1 - Math.exp(-day / tau))).toFixed(4));
}

function constantArray(value: number, length: number): number[] {
  return Array.from({ length }, () => value);
}

function addSeries(left: number[], right: number[]): number[] {
  return left.map((value, index) => +(value + (right[index] ?? 0)).toFixed(4));
}

function toEffectTrajectory(days: number[], effect: number[]) {
  return days.map((day, index) => ({ day, effect: effect[index] ?? 0 }));
}

const BASELINE_STATE: Record<string, number> = {
  affective_state: 0.38,
  serotonergic_exposure: 0.62,
  adherence: 0.86,
  sleep_quality: 0.48,
  physical_activity: 0.54,
  social_engagement: 0.44,
  prescription_event: 0.2,
  seasonal_load: -0.12,
  life_events_load: 0.16,
  cyp2c19_metabolizer_status: 0.5,
  baseline_extraversion: 0.57,
  chronic_inflammation_baseline: 0.3,
};

const BASELINE_START: SimulateScenarioResult["start"] = {
  kind: "baseline",
  time_index: null,
  time: null,
  state_source: "baseline_steady_state",
};

const DEMO_MANIFEST_EFFECTS = {
  gp_clinical_severity: -0.18,
  journal_affect_severity: -0.21,
  journal_rumination_intensity: -0.14,
  late_night_message_count: -0.09,
  state_of_mind_valence: 0.24,
};

/** Build a baseline-start trajectory scenario from per-node steady-state effect deltas. */
function buildScenarioTrajectory(args: {
  clamp: LatentClampInput;
  outcome: string;
  horizonDays: number;
  nodeSteadyEffects: Record<string, number>;
  nodeTaus: Record<string, number>;
  summary: SimulateScenarioResult["summary"];
  manifestEffects?: Record<string, number>;
}): SimulateScenarioResult {
  const days = dailyGrid(args.horizonDays);
  const nodeEffects: Record<string, number[]> = {};
  for (const node of Object.keys(BASELINE_STATE)) {
    const steady = args.nodeSteadyEffects[node] ?? 0;
    nodeEffects[node] =
      steady === 0
        ? constantArray(0, days.length)
        : expApproach(steady, args.nodeTaus[node] ?? 20, days);
  }
  const reference = Object.fromEntries(
    Object.entries(BASELINE_STATE).map(([node, value]) => [
      node,
      constantArray(value, days.length),
    ]),
  );
  const action = Object.fromEntries(
    Object.keys(BASELINE_STATE).map((node) => [
      node,
      addSeries(reference[node] ?? [], nodeEffects[node] ?? []),
    ]),
  );
  return {
    start: BASELINE_START,
    clamps: [args.clamp],
    outcome: args.outcome,
    estimand: "trajectory",
    reference_mean: BASELINE_STATE[args.outcome] ?? 0,
    summary: args.summary,
    effect_trajectory: toEffectTrajectory(days, nodeEffects[args.outcome] ?? []),
    visualization: {
      reference_node_trajectories: reference,
      action_node_trajectories: action,
      node_effect_trajectories: nodeEffects,
      start_state: null,
    },
    manifest_effects: args.manifestEffects ?? null,
    warnings: [],
  };
}

// ── synthesized simulations ────────────────────────────────────────────────

/** set mode: clamp sleep_quality to 0.5, read affective_state over 30d. */
const sleepSetResult: SimulateScenarioResult = buildScenarioTrajectory({
  clamp: { variable: "sleep_quality", mode: "set", value: 0.5, from_day: 0 },
  outcome: "affective_state",
  horizonDays: 30,
  nodeSteadyEffects: {
    affective_state: 0.16,
    sleep_quality: +(0.5 - (BASELINE_STATE.sleep_quality ?? 0)).toFixed(4),
    physical_activity: 0.05,
    social_engagement: 0.03,
    adherence: 0.02,
  },
  nodeTaus: { affective_state: 10, physical_activity: 16, social_engagement: 20, adherence: 28 },
  summary: { mean: 0.16, median: 0.16, lower_95: 0.05, upper_95: 0.27, prob_positive: 0.95 },
  manifestEffects: DEMO_MANIFEST_EFFECTS,
});

/** shift targeting a NON-global outcome: serotonergic → sleep_quality over 30d. */
const serotoninOnSleepResult: SimulateScenarioResult = buildScenarioTrajectory({
  clamp: { variable: "serotonergic_exposure", mode: "shift", amount: 1.0, from_day: 0 },
  outcome: "sleep_quality",
  horizonDays: 30,
  nodeSteadyEffects: {
    serotonergic_exposure: 1.0,
    sleep_quality: 0.12,
    affective_state: 0.08,
    physical_activity: 0.06,
  },
  nodeTaus: { sleep_quality: 18, affective_state: 22, physical_activity: 25 },
  summary: { mean: 0.12, median: 0.11, lower_95: 0.02, upper_95: 0.23, prob_positive: 0.93 },
});

/** No-intervention baseline: the reference world (no clamps), action ≡ reference. */
function buildBaselineReference(outcome: string, horizonDays: number): SimulateScenarioResult {
  const days = dailyGrid(horizonDays);
  const reference = Object.fromEntries(
    Object.entries(BASELINE_STATE).map(([node, value]) => [
      node,
      constantArray(value, days.length),
    ]),
  );
  return {
    start: BASELINE_START,
    clamps: [],
    outcome,
    estimand: "trajectory",
    reference_mean: BASELINE_STATE[outcome] ?? 0,
    summary: { mean: 0, median: 0, lower_95: 0, upper_95: 0, prob_positive: 0.5 },
    effect_trajectory: toEffectTrajectory(days, constantArray(0, days.length)),
    visualization: {
      reference_node_trajectories: reference,
      action_node_trajectories: Object.fromEntries(
        Object.entries(reference).map(([node, series]) => [node, series.slice()]),
      ),
      node_effect_trajectories: Object.fromEntries(
        Object.keys(BASELINE_STATE).map((node) => [node, constantArray(0, days.length)]),
      ),
      start_state: null,
    },
    manifest_effects: null,
    warnings: [],
  };
}

const baselineReferenceResult: SimulateScenarioResult = buildBaselineReference(
  "affective_state",
  60,
);

// ── persisted trace assembly (tool_result as JSON string) ───────────────────

function simToolInput(
  result: Stage6SimulationResult,
  horizonDays: number,
): Record<string, unknown> {
  const query = { estimand: result.estimand, horizon_days: horizonDays, projection: "latent" };
  const start =
    result.start.kind === "abducted"
      ? { kind: "abducted", time: result.start.time }
      : { kind: "baseline" };
  return { start, clamps: result.clamps, outcome: result.outcome, query };
}

interface ScenarioTurn {
  userQuery: string;
  answer: string;
  result: Stage6SimulationResult;
  horizonDays: number;
}

const SCENARIO_TURNS: ScenarioTurn[] = [
  {
    userQuery: "Before we intervene, show me the system with no do() — the reference baseline.",
    answer:
      "**No intervention — the reference world.** With nothing clamped, every construct holds near its set-point and affective_state stays at its baseline level. This is the reference all the intervention scenarios below are compared against.",
    result: baselineReferenceResult,
    horizonDays: 60,
  },
  {
    userQuery:
      "If a +1 SD serotonergic boost mainly works through sleep, what does it do to sleep_quality alone over a month?",
    answer:
      "A +1 SD serotonergic shift lifts sleep_quality by ~+0.12 SD over ~3 weeks — a smaller, slower channel than its direct effect on mood.",
    result: serotoninOnSleepResult,
    horizonDays: 30,
  },
  {
    userQuery:
      "Suppose we could pin sleep_quality at 0.5 — how much does mood improve over 30 days?",
    answer:
      "Clamping sleep_quality to 0.5 raises affective_state by ~+0.16 SD, materializing over ~10 days and comparable in magnitude to the direct SSRI effect.",
    result: sleepSetResult,
    horizonDays: 30,
  },
  {
    userQuery:
      "Counterfactually, if adherence had been 0.5 SD higher from late February, how would affective state have differed?",
    answer:
      "Conditioning on the fitted state at 2026-02-28, a +0.5 SD adherence shift improves the affective-state path by ~+0.24 SD relative to the factual forecast.",
    result: counterfactualResult,
    horizonDays: 60,
  },
  {
    userQuery: "What happens to affective state if we raise serotonergic exposure by 1 SD?",
    answer:
      "Raising serotonergic exposure lifts affective state by ~+0.31 SD at steady state, propagating through sleep quality and physical activity over roughly three weeks.",
    result: interventionResult,
    horizonDays: 60,
  },
];

function scenarioTurnMessages(turn: ScenarioTurn, index: number): TraceMessage[] {
  const toolCallId = `sim-${index}`;
  return [
    { role: "user", content: turn.userQuery, tool_is_error: false },
    {
      role: "assistant",
      content: turn.answer,
      tool_calls: [
        {
          id: toolCallId,
          type: "function",
          function: {
            name: "simulate",
            arguments: JSON.stringify(simToolInput(turn.result, turn.horizonDays)),
          },
        },
      ],
      tool_is_error: false,
    },
    {
      role: "tool",
      content: "",
      tool_call_id: toolCallId,
      tool_name: "simulate",
      // Persisted form: the result is a JSON STRING, as a reloaded workspace stores it.
      tool_result: JSON.stringify(turn.result),
      tool_is_error: false,
    },
  ];
}

const demo = demoStage6 as unknown as Stage6Data;

export const materializedTrace: LLMTrace = {
  model: "openrouter/anthropic/claude-sonnet-4",
  total_time_seconds: 9.4,
  usage: { input_tokens: 4120, output_tokens: 980, reasoning_tokens: 260 },
  messages: [
    {
      role: "system",
      content:
        "You are exploring follow-up composable scenarios for Stage 6 of a fitted causal state-space model of affective state.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content: demo.final_summary ?? "Baseline interventional ranking computed.",
      tool_is_error: false,
    },
    ...SCENARIO_TURNS.flatMap((turn, index) => scenarioTurnMessages(turn, index)),
  ],
};

/** The complete ideal materialized Stage 6 artifact (baselines + sims + summary). */
export const materializedStage6Data: Stage6Data = {
  llm_trace: materializedTrace,
  intervention_results: demo.intervention_results,
  saved_scenarios: demo.saved_scenarios ?? null,
  final_summary: demo.final_summary ?? null,
};

export const outcomeName = "affective_state";
