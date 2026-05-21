import { describe, expect, it } from "vitest";
import type { Stage6SimulationResult } from "./intervention-dag-types";

import {
  formatActionDescription,
  formatActionReferenceLabel,
  formatActionShortLabel,
  formatCounterfactualStartLabel,
  getActionReference,
  getNodeActionSeries,
  getEffectTrajectoryDays,
  getNodeReferenceSeries,
} from "./intervention-dag-semantics";

describe("intervention DAG semantics", () => {
  it("formats baseline-relative intervention labels from rung-2 results", () => {
    const result = {
      rung: 2 as const,
      action: {
        variable: "lipid_burden",
        mode: "shift" as const,
        amount: 1,
      },
      effect_trajectory: [{ day: 1, effect: 0.2 }],
      visualization: {
        reference_node_trajectories: {
          lipid_burden: [0.85],
        },
        action_node_trajectories: {
          lipid_burden: [1.85],
        },
        node_effect_trajectories: {
          lipid_burden: [1],
        },
        start_state: null,
      },
    } as unknown as Stage6SimulationResult;

    expect(getActionReference(result)).toBe("baseline_steady_state");
    expect(formatActionDescription(result)).toBe(
      "do(lipid_burden = baseline +1.0)",
    );
    expect(formatActionShortLabel(result.action)).toBe("shift +1.0");
    expect(formatActionReferenceLabel(result)).toBe(
      "from baseline steady state",
    );
    expect(getEffectTrajectoryDays(result)).toEqual([1]);
    expect(getNodeReferenceSeries(result, "lipid_burden")).toEqual([0.85]);
    expect(getNodeActionSeries(result, "lipid_burden")).toEqual([1.85]);
  });

  it("formats fitted-start-state counterfactual labels from rung-3 results", () => {
    const result = {
      rung: 3 as const,
      action: {
        variable: "medication_adherence",
        mode: "shift" as const,
        amount: 1,
      },
      start: {
        time_index: 89,
        time: "2024-03-31T00:00:00+00:00",
        state_source: "fitted_latent_paths",
      },
    } as unknown as Stage6SimulationResult;

    expect(getActionReference(result)).toBe("fitted_start_state");
    expect(formatActionDescription(result)).toBe(
      "do(medication_adherence = fitted start state +1.0)",
    );
    expect(formatActionReferenceLabel(result)).toBe("from fitted start state");
    expect(formatCounterfactualStartLabel(result)).toBe(
      "started from fitted state 2024-03-31 (#89)",
    );
  });
});
