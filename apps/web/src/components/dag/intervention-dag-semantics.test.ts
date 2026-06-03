import { describe, expect, it } from "vitest";
import {
  formatClampReferenceLabel,
  formatClampShortLabel,
  formatScenarioActionDescription,
  formatScenarioStartLabel,
  getActionReference,
  getClampedVariables,
  getEffectTrajectoryDays,
  getNodeActionSeries,
  getNodeReferenceSeries,
  isAbductedStart,
} from "./intervention-dag-semantics";
import type { Stage6SimulationResult } from "./intervention-dag-types";

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
    } as unknown as Stage6SimulationResult;

    expect(isAbductedStart(result)).toBe(false);
    expect(getActionReference(result)).toBe("baseline_steady_state");
    expect(getClampedVariables(result)).toEqual(["lipid_burden"]);
    expect(formatScenarioActionDescription(result)).toBe("do(lipid_burden shift +1.0)");
    expect(formatClampShortLabel(result.clamps[0])).toBe("shift +1.0");
    expect(formatClampReferenceLabel(result, result.clamps[0])).toBe("from baseline");
    expect(getEffectTrajectoryDays(result)).toEqual([1]);
    expect(getNodeReferenceSeries(result, "lipid_burden")).toEqual([0.85]);
    expect(getNodeActionSeries(result, "lipid_burden")).toEqual([1.85]);
  });

  it("formats fitted-start-state labels from an abducted-start scenario", () => {
    const result = {
      start: {
        kind: "abducted",
        time_index: 89,
        time: "2024-03-31T00:00:00+00:00",
        state_source: "fitted_latent_paths",
      },
      clamps: [
        { variable: "medication_adherence", mode: "shift", amount: 1, from_day: 10, to_day: 20 },
      ],
    } as unknown as Stage6SimulationResult;

    expect(isAbductedStart(result)).toBe(true);
    expect(getActionReference(result)).toBe("fitted_start_state");
    expect(formatClampReferenceLabel(result, result.clamps[0])).toBe(
      "from fitted start state · d10–20",
    );
    expect(formatScenarioStartLabel(result)).toBe("from fitted state 2024-03-31 (#89)");
  });
});
