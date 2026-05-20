import { describe, expect, it } from "vitest";
import type { UIMessage } from "ai";
import type { Stage1aData, Stage1bData, Stage4Data, Stage5bData } from "@nof1-causal-lab/api-types";
import stage1aFixture from "../../../../../../data/DEMO/run/stage-1a.json";
import stage1bFixture from "../../../../../../data/DEMO/run/stage-1b.json";
import stage4Fixture from "../../../../../../data/DEMO/run/stage-4.json";
import stage5bFixture from "../../../../../../data/DEMO/run/stage-5b.json";
import {
  counterfactualResult,
  interventionResult,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";
import { buildStage6DagScene } from "./stage-6-presentation";

function createAssistantToolMessage(output: unknown, input: unknown = {}): UIMessage {
  const result = output as { rung: 2 | 3 };
  return {
    id: `tool-${result.rung}`,
    role: "assistant",
    parts: [
      { type: "text", text: "Done." },
      {
        type: "dynamic-tool",
        toolCallId: `tool-${result.rung}-call`,
        toolName: result.rung === 2 ? "simulate_intervention" : "simulate_counterfactual",
        state: "output-available",
        input,
        output,
      },
    ],
  };
}

const stage1a = stage1aFixture as unknown as Stage1aData;
const stage1b = stage1bFixture as unknown as Stage1bData;
const stage4 = stage4Fixture as unknown as Stage4Data;
const stage5b = stage5bFixture as unknown as Stage5bData;

describe("buildStage6DagScene", () => {
  it("returns a scene with no simulation when no follow-up tool result exists", () => {
    const scene = buildStage6DagScene({ stage1a, stage1b });

    expect(scene?.simulationResult).toBeUndefined();
    expect(scene?.requestedHorizonDays).toBeUndefined();
  });

  it("returns the latest simulation when a follow-up tool result exists", () => {
    const scene = buildStage6DagScene({
      stage1a,
      stage1b,
      stage4,
      stage5b,
      refinementMessages: [
        createAssistantToolMessage(interventionResult, {
          query: { horizon_days: 60 },
        }),
        createAssistantToolMessage(counterfactualResult, {
          query: { horizon_days: 45 },
        }),
      ],
    });

    expect(scene?.simulationResult).toEqual(counterfactualResult);
    expect(scene?.requestedHorizonDays).toBe(45);
    expect(scene?.edgePosteriors).toMatchObject({
      "serotonergic_exposure→affective_state": {
        mean: 0.058,
        ci_lower: 0.02416,
        ci_upper: 0.09184,
      },
      "sleep_quality→affective_state": {
        mean: 0.072,
        ci_lower: 0.0344,
        ci_upper: 0.1096,
      },
      "affective_state→physical_activity": {
        mean: 0.046,
        ci_lower: 0.01216,
        ci_upper: 0.07984,
      },
    });
  });
});
