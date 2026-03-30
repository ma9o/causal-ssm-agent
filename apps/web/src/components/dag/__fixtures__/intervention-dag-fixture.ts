import type {
  LLMTrace,
  SimulateCounterfactualResult,
  SimulateInterventionResult,
} from "@causal-ssm/api-types";
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

function driftToward(
  start: number,
  target: number,
  tau: number,
  days: number[],
): number[] {
  return days.map(
    (day) => +(start + (target - start) * (1 - Math.exp(-day / tau))).toFixed(4),
  );
}

function subtractSeries(left: number[], right: number[]): number[] {
  return left.map((value, index) => +(value - (right[index] ?? 0)).toFixed(4));
}

function toEffectTrajectory(days: number[], effect: number[]) {
  return days.map((day, index) => ({
    day,
    effect: effect[index] ?? 0,
  }));
}

export const edgePosteriors: Record<string, EdgePosterior> = {
  "lipid_burden→vascular_inflammation": {
    mean: 0.65,
    ci_lower: 0.48,
    ci_upper: 0.82,
  },
  "vascular_inflammation→cardiovascular_risk": {
    mean: 0.58,
    ci_lower: 0.38,
    ci_upper: 0.78,
  },
  "glycemic_control→cardiovascular_risk": {
    mean: 0.35,
    ci_lower: 0.15,
    ci_upper: 0.55,
  },
  "arterial_pressure→cardiovascular_risk": {
    mean: 0.42,
    ci_lower: 0.25,
    ci_upper: 0.59,
  },
  "medication_adherence→lipid_burden": {
    mean: -0.48,
    ci_lower: -0.68,
    ci_upper: -0.28,
  },
  "medication_adherence→glycemic_control": {
    mean: -0.32,
    ci_lower: -0.52,
    ci_upper: -0.12,
  },
  "medication_adherence→arterial_pressure": {
    mean: -0.38,
    ci_lower: -0.58,
    ci_upper: -0.18,
  },
  "genetic_predisposition→lipid_burden": {
    mean: 0.3,
    ci_lower: 0.12,
    ci_upper: 0.48,
  },
  "genetic_predisposition→cardiovascular_risk": {
    mean: 0.15,
    ci_lower: -0.05,
    ci_upper: 0.35,
  },
  "psychosocial_stress→glycemic_control": {
    mean: 0.25,
    ci_lower: 0.03,
    ci_upper: 0.47,
  },
  "psychosocial_stress→cardiovascular_risk": {
    mean: 0.2,
    ci_lower: 0.01,
    ci_upper: 0.39,
  },
};

export const processNoise: Record<string, number> = {
  cardiovascular_risk: 0.15,
  lipid_burden: 0.2,
  vascular_inflammation: 0.18,
  glycemic_control: 0.25,
  arterial_pressure: 0.22,
  medication_adherence: 0.3,
  genetic_predisposition: 0.05,
  psychosocial_stress: 0.08,
};

const rung2Days = buildDailyGrid(60);
const rung2NodeEffects = {
  lipid_burden: constantArray(1.0, rung2Days.length),
  vascular_inflammation: expApproach(0.65, 15, rung2Days),
  cardiovascular_risk: expApproach(0.43, 30, rung2Days),
  glycemic_control: constantArray(0, rung2Days.length),
  arterial_pressure: constantArray(0, rung2Days.length),
  medication_adherence: constantArray(0, rung2Days.length),
  genetic_predisposition: constantArray(0, rung2Days.length),
  psychosocial_stress: constantArray(0, rung2Days.length),
};

export const interventionResult: SimulateInterventionResult = {
  rung: 2,
  action: {
    variable: "lipid_burden",
    mode: "shift",
    amount: 1.0,
  },
  outcome: "cardiovascular_risk",
  estimand: "trajectory",
  baseline_treatment_mean: 0.85,
  summary: {
    mean: 0.43,
    median: 0.42,
    lower_95: 0.19,
    upper_95: 0.71,
    prob_positive: 0.99,
  },
  effect_trajectory: toEffectTrajectory(
    rung2Days,
    rung2NodeEffects.cardiovascular_risk,
  ),
  visualization: {
    node_effect_trajectories: rung2NodeEffects,
    abducted_state: null,
  },
  warnings: [],
};

const rung3Days = buildDailyGrid(60);
const abductedState: Record<string, number> = {
  cardiovascular_risk: 0.72,
  lipid_burden: 0.85,
  vascular_inflammation: 0.55,
  glycemic_control: 0.48,
  arterial_pressure: 0.62,
  medication_adherence: 0.35,
  genetic_predisposition: 0.4,
  psychosocial_stress: 0.3,
};

const factualEq: Record<string, number> = {
  cardiovascular_risk: 0.68,
  lipid_burden: 0.8,
  vascular_inflammation: 0.52,
  glycemic_control: 0.45,
  arterial_pressure: 0.58,
  medication_adherence: 0.35,
  genetic_predisposition: 0.4,
  psychosocial_stress: 0.3,
};

const cfTargets: Record<string, number> = {
  medication_adherence: 1.35,
  lipid_burden: 0.37,
  glycemic_control: 0.16,
  arterial_pressure: 0.24,
  vascular_inflammation: 0.238,
  cardiovascular_risk: 0.384,
  genetic_predisposition: 0.4,
  psychosocial_stress: 0.3,
};

const factualTrajectories = Object.fromEntries(
  Object.entries(abductedState).map(([node, start]) => [
    node,
    driftToward(start, factualEq[node] ?? start, 60, rung3Days),
  ]),
);

const counterfactualTrajectories = Object.fromEntries(
  Object.entries(abductedState).map(([node, start]) => {
    if (node === "medication_adherence") {
      return [node, constantArray(1.35, rung3Days.length)];
    }
    const tau =
      node === "cardiovascular_risk" ? 30 : node === "vascular_inflammation" ? 20 : 18;
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
    start_time: "2024-01-01T00:00:00+00:00",
    end_time: "2024-03-31T00:00:00+00:00",
    n_timepoints: 90,
    variables: [
      "medication_adherence",
      "lipid_burden",
      "cardiovascular_risk",
    ],
    conditioning_method: "kalman_smoother",
  },
  action: {
    variable: "medication_adherence",
    mode: "shift",
    amount: 1.0,
  },
  outcome: "cardiovascular_risk",
  estimand: "trajectory",
  baseline_forecast_mean: factualEq.cardiovascular_risk,
  summary: {
    mean: -0.296,
    median: -0.29,
    lower_95: -0.47,
    upper_95: -0.12,
    prob_positive: 0.03,
  },
  effect_trajectory: toEffectTrajectory(
    rung3Days,
    rung3NodeEffects.cardiovascular_risk,
  ),
  visualization: {
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
        "You are generating the baseline treatment ranking for Stage 6 of a causal state-space model. Compute do-operator steady-state effects for all identifiable treatments.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "Lipid burden shows the strongest positive effect on cardiovascular risk (\u03C4\u0302 = 0.43, 95% CI [0.19, 0.71]) via its causal chain through vascular inflammation. Medication adherence has the strongest protective effect (\u03C4\u0302 = \u22120.34) via simultaneous reduction of lipid burden, glycemic dysregulation, and arterial pressure. Arterial pressure and vascular inflammation have comparable direct effects (\u03C4\u0302 \u2248 0.36\u20130.38).\n\nYou can now run rung 2 interventional queries (for example, what happens if we shift lipid burden by +1?) or rung 3 counterfactual queries (for example, what would have happened had medication adherence been higher?).",
      tool_is_error: false,
    },
  ],
};
