import type {
  LLMTrace,
  SimulateCounterfactualResult,
  SimulateInterventionResult,
} from "@nof1-causal-lab/api-types";
import type { EdgePosterior } from "../intervention-dag-types";

function buildDailyGrid(length: number): number[] {
  return Array.from({ length }, (_, index) => index + 1);
}

function expApproach(steadyState: number, tau: number, days: number[]): number[] {
  return days.map((day) => +(steadyState * (1 - Math.exp(-day / tau))).toFixed(4));
}

function constantArray(value: number, length: number): number[] {
  return Array.from({ length }, () => value);
}

function driftToward(start: number, target: number, tau: number, days: number[]): number[] {
  return days.map((day) => +(start + (target - start) * (1 - Math.exp(-day / tau))).toFixed(4));
}

function subtractSeries(left: number[], right: number[]): number[] {
  return left.map((value, index) => +(value - (right[index] ?? 0)).toFixed(4));
}

function addSeries(left: number[], right: number[]): number[] {
  return left.map((value, index) => +(value + (right[index] ?? 0)).toFixed(4));
}

function toEffectTrajectory(days: number[], effect: number[]) {
  return days.map((day, index) => ({
    day,
    effect: effect[index] ?? 0,
  }));
}

export const edgePosteriors: Record<string, EdgePosterior> = {
  "serotonergic_exposure→affective_state": {
    mean: 0.058,
    ci_lower: 0.02416,
    ci_upper: 0.09184,
  },
  "physical_activity→affective_state": {
    mean: 0.041,
    ci_lower: 0.01468,
    ci_upper: 0.06732,
  },
  "sleep_quality→affective_state": {
    mean: 0.072,
    ci_lower: 0.0344,
    ci_upper: 0.1096,
  },
  "social_engagement→affective_state": {
    mean: 0.029,
    ci_lower: -0.01236,
    ci_upper: 0.07036,
  },
  "seasonal_load→affective_state": {
    mean: -0.018,
    ci_lower: -0.03868,
    ci_upper: 0.00268,
  },
  "life_events_load→affective_state": {
    mean: -0.034,
    ci_lower: -0.05844,
    ci_upper: -0.00956,
  },
  "affective_state→physical_activity": {
    mean: 0.046,
    ci_lower: 0.01216,
    ci_upper: 0.07984,
  },
  "affective_state→sleep_quality": {
    mean: 0.052,
    ci_lower: 0.02004,
    ci_upper: 0.08396,
  },
  "serotonergic_exposure→physical_activity": {
    mean: 0.011,
    ci_lower: -0.1488,
    ci_upper: 0.1708,
  },
  "serotonergic_exposure→sleep_quality": {
    mean: 0.024,
    ci_lower: -0.12264,
    ci_upper: 0.17064,
  },
};

export const processNoise: Record<string, number> = {
  affective_state: 0.18,
  serotonergic_exposure: 0.12,
  adherence: 0.16,
  sleep_quality: 0.22,
  physical_activity: 0.31,
  social_engagement: 0.2,
  prescription_event: 0.08,
  seasonal_load: 0.04,
  life_events_load: 0.14,
  cyp2c19_metabolizer_status: 0.02,
  baseline_extraversion: 0.02,
};

const rung2Days = buildDailyGrid(60);
const rung2BaselineState: Record<string, number> = {
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
};
const rung2ReferenceTrajectories = Object.fromEntries(
  Object.entries(rung2BaselineState).map(([node, value]) => [
    node,
    constantArray(value, rung2Days.length),
  ]),
);
const rung2NodeEffects = {
  affective_state: expApproach(0.31, 18, rung2Days),
  serotonergic_exposure: constantArray(1.0, rung2Days.length),
  adherence: expApproach(0.05, 35, rung2Days),
  sleep_quality: expApproach(0.14, 22, rung2Days),
  physical_activity: expApproach(0.18, 25, rung2Days),
  social_engagement: expApproach(0.09, 30, rung2Days),
  prescription_event: expApproach(0.03, 40, rung2Days),
  seasonal_load: constantArray(0, rung2Days.length),
  life_events_load: constantArray(0, rung2Days.length),
  cyp2c19_metabolizer_status: constantArray(0, rung2Days.length),
  baseline_extraversion: constantArray(0, rung2Days.length),
};
const rung2ActionTrajectories = Object.fromEntries(
  Object.keys(rung2ReferenceTrajectories).map((node) => [
    node,
    addSeries(
      rung2ReferenceTrajectories[node] ?? constantArray(0, rung2Days.length),
      rung2NodeEffects[node as keyof typeof rung2NodeEffects] ?? constantArray(0, rung2Days.length),
    ),
  ]),
);

export const interventionResult: SimulateInterventionResult = {
  rung: 2,
  action: {
    variable: "serotonergic_exposure",
    mode: "shift",
    amount: 1.0,
  },
  outcome: "affective_state",
  estimand: "trajectory",
  baseline_treatment_mean: 0.62,
  summary: {
    mean: 0.31,
    median: 0.3,
    lower_95: 0.13,
    upper_95: 0.49,
    prob_positive: 0.98,
  },
  effect_trajectory: toEffectTrajectory(rung2Days, rung2NodeEffects.affective_state),
  visualization: {
    reference_node_trajectories: rung2ReferenceTrajectories,
    action_node_trajectories: rung2ActionTrajectories,
    node_effect_trajectories: rung2NodeEffects,
    abducted_state: null,
  },
  warnings: [],
};

const rung3Days = buildDailyGrid(60);
const abductedState: Record<string, number> = {
  affective_state: 0.34,
  serotonergic_exposure: 0.58,
  adherence: 0.74,
  sleep_quality: 0.46,
  physical_activity: 0.49,
  social_engagement: 0.4,
  prescription_event: 0.18,
  seasonal_load: -0.14,
  life_events_load: 0.18,
  cyp2c19_metabolizer_status: 0.5,
  baseline_extraversion: 0.57,
};
const factualEq: Record<string, number> = {
  affective_state: 0.4,
  serotonergic_exposure: 0.6,
  adherence: 0.74,
  sleep_quality: 0.5,
  physical_activity: 0.52,
  social_engagement: 0.43,
  prescription_event: 0.18,
  seasonal_load: -0.14,
  life_events_load: 0.18,
  cyp2c19_metabolizer_status: 0.5,
  baseline_extraversion: 0.57,
};
const cfTargets: Record<string, number> = {
  affective_state: 0.64,
  serotonergic_exposure: 0.94,
  adherence: 1.24,
  sleep_quality: 0.62,
  physical_activity: 0.67,
  social_engagement: 0.52,
  prescription_event: 0.2,
  seasonal_load: -0.14,
  life_events_load: 0.18,
  cyp2c19_metabolizer_status: 0.5,
  baseline_extraversion: 0.57,
};
const factualTrajectories = Object.fromEntries(
  Object.entries(abductedState).map(([node, start]) => [
    node,
    driftToward(start, factualEq[node] ?? start, 60, rung3Days),
  ]),
);
const counterfactualTrajectories = Object.fromEntries(
  Object.entries(abductedState).map(([node, start]) => {
    if (node === "adherence") {
      return [node, constantArray(cfTargets.adherence, rung3Days.length)];
    }
    const tau = node === "affective_state" ? 30 : node === "serotonergic_exposure" ? 18 : 24;
    return [node, driftToward(start, cfTargets[node] ?? start, tau, rung3Days)];
  }),
);
const rung3NodeEffects = Object.fromEntries(
  Object.keys(abductedState).map((node) => [
    node,
    subtractSeries(
      counterfactualTrajectories[node] ?? constantArray(0, rung3Days.length),
      factualTrajectories[node] ?? constantArray(0, rung3Days.length),
    ),
  ]),
);

export const counterfactualResult: SimulateCounterfactualResult = {
  rung: 3,
  evidence: {
    start_time: "2025-12-01T00:00:00+00:00",
    end_time: "2026-02-28T00:00:00+00:00",
    n_timepoints: 90,
    variables: ["adherence", "serotonergic_exposure", "affective_state"],
    conditioning_method: "posterior_smoother",
  },
  action: {
    variable: "adherence",
    mode: "shift",
    amount: 0.5,
  },
  outcome: "affective_state",
  estimand: "trajectory",
  baseline_forecast_mean: factualEq.affective_state,
  summary: {
    mean: 0.24,
    median: 0.23,
    lower_95: 0.08,
    upper_95: 0.39,
    prob_positive: 0.96,
  },
  effect_trajectory: toEffectTrajectory(rung3Days, rung3NodeEffects.affective_state),
  visualization: {
    reference_node_trajectories: factualTrajectories,
    action_node_trajectories: counterfactualTrajectories,
    node_effect_trajectories: rung3NodeEffects,
    abducted_state: abductedState,
  },
  warnings: [],
};

export const mockTrace: LLMTrace = {
  model: "openrouter/anthropic/claude-sonnet-4",
  total_time_seconds: 6.8,
  usage: { input_tokens: 2450, output_tokens: 520, reasoning_tokens: 180 },
  messages: [
    {
      role: "system",
      content:
        "You are generating follow-up simulations for Stage 6 of a causal state-space model. Compute intervention and counterfactual trajectories over the fitted latent graph.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "The escitalopram exposure path has a positive projected effect on affective state, with downstream movement through sleep quality and physical activity. A counterfactual adherence increase raises serotonergic exposure and improves the affective-state trajectory relative to the factual forecast.",
      tool_is_error: false,
    },
  ],
};
