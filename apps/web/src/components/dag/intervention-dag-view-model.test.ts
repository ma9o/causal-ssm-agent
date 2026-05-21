import { describe, expect, it } from "vitest";
import {
  counterfactualResult,
  interventionResult,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";
import { buildInterventionDagViewModel } from "./intervention-dag-view-model";

const emptyAnimation = {
  phase: "idle",
  timeIndex: 0,
  nodePhases: {},
  nodeEffects: {},
  startStateValues: {},
} as const;

describe("buildInterventionDagViewModel", () => {
  it("uses start, midpoint, and final horizon markers without a peak marker", () => {
    const viewModel = buildInterventionDagViewModel({
      constructs: [],
      requestedHorizonDays: 60,
      result: interventionResult,
      animation: emptyAnimation,
    });

    expect(viewModel.temporalMarkers).toEqual([
      { day: 1, label: "1d" },
      { day: 7, label: "7d" },
      { day: 30, label: "30d" },
      { day: 60, label: "60d" },
    ]);
  });

  it("uses the returned trajectory endpoint when it differs from the requested horizon", () => {
    const shortenedResult = {
      ...counterfactualResult,
      effect_trajectory: counterfactualResult.effect_trajectory?.slice(0, 4),
    };
    const viewModel = buildInterventionDagViewModel({
      constructs: [],
      requestedHorizonDays: 60,
      result: shortenedResult,
      animation: emptyAnimation,
    });

    expect(viewModel.temporalMarkers).toEqual([
      { day: 1, label: "1d" },
      { day: 4, label: "4d" },
    ]);
  });
});
