import { describe, expect, it } from "vitest";
import type { UIMessage } from "ai";
import type { Stage1aData, Stage1bData, Stage4Data, Stage5bData } from "@causal-ssm/api-types";
import stage1aFixture from "../../../../../../data/DOCTOLIB/run/stage-1a.json";
import stage1bFixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";
import stage4Fixture from "../../../../../../data/DOCTOLIB/run/stage-4.json";
import stage5bFixture from "../../../../../../data/DOCTOLIB/run/stage-5b.json";
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
  it("returns the baseline DAG when no follow-up tool result exists", () => {
    const scene = buildStage6DagScene({ stage1a, stage1b });

    expect(scene?.kind).toBe("baseline");
  });

  it("returns the latest simulation DAG when a follow-up tool result exists", () => {
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

    expect(scene?.kind).toBe("simulation");
    if (scene?.kind === "simulation") {
      expect(scene.simulationResult).toEqual(counterfactualResult);
      expect(scene.requestedHorizonDays).toBe(45);
      expect(scene.edgePosteriors).toMatchObject({
        "lipid_burden→cardiovascular_risk": {
          mean: 0.42,
          ci_lower: 0.24,
          ci_upper: 0.61,
        },
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
        "arterial_pressure→cardiovascular_risk": {
          mean: 0.35,
          ci_lower: 0.19,
          ci_upper: 0.51,
        },
        "glycemic_control→cardiovascular_risk": {
          mean: -0.27,
          ci_lower: -0.42,
          ci_upper: -0.12,
        },
        "medication_adherence→lipid_burden": {
          mean: -0.48,
          ci_lower: -0.68,
          ci_upper: -0.28,
        },
        "medication_adherence→arterial_pressure": {
          mean: -0.38,
          ci_lower: -0.58,
          ci_upper: -0.18,
        },
      });
    }
  });
});
