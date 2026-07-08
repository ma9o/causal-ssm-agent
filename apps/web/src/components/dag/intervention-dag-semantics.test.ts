import { describe, expect, it } from "vitest";
import {
  formatScenarioActionDescription,
  getEffectTrajectoryDays,
  getNodeActionSeries,
  getNodeReferenceSeries,
} from "./intervention-dag-semantics";
import type { AnalysisSimulationResult } from "./intervention-dag-types";

describe("intervention DAG semantics", () => {
  it("formats baseline-relative clamp labels from a baseline-start scenario", () => {
    const result = {
      start: { kind: "baseline", state_source: "baseline_steady_state" },
      clamps: [{ variable: "lipid_burden", mode: "shift", amount: 1, from_day: 0 }],
      effect_trajectory: [{ day: 1, effect: 0.2 }],
      visualization: {
        reference_node_trajectories: { lipid_burden: [0.85] },
        action_node_trajectories: { lipid_burden: [1.85] },
        node_effect_trajectories: { lipid_burden: [1] },
        start_state: null,
      },
    } as unknown as AnalysisSimulationResult;

    expect(formatScenarioActionDescription(result)).toBe("do(lipid_burden shift +1.0)");
    expect(getEffectTrajectoryDays(result)).toEqual([1]);
    expect(getNodeReferenceSeries(result, "lipid_burden")).toEqual([0.85]);
    expect(getNodeActionSeries(result, "lipid_burden")).toEqual([1.85]);
  });
});
