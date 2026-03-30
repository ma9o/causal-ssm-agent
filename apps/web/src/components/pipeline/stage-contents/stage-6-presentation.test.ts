import { describe, expect, it } from "vitest";
import type { UIMessage } from "ai";
import type { Stage1aData, Stage1bData } from "@causal-ssm/api-types";
import stage1aFixture from "../../../../../../data/DOCTOLIB/run/stage-1a.json";
import stage1bFixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";
import {
  counterfactualResult,
  interventionResult,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";
import { buildStage6DagScene } from "./stage-6-presentation";

function createAssistantToolMessage(output: unknown): UIMessage {
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
        input: {},
        output,
      },
    ],
  };
}

const stage1a = stage1aFixture as unknown as Stage1aData;
const stage1b = stage1bFixture as unknown as Stage1bData;

describe("buildStage6DagScene", () => {
  it("returns the baseline DAG when no follow-up tool result exists", () => {
    const scene = buildStage6DagScene({ stage1a, stage1b });

    expect(scene?.kind).toBe("baseline");
  });

  it("returns the latest simulation DAG when a follow-up tool result exists", () => {
    const scene = buildStage6DagScene({
      stage1a,
      stage1b,
      refinementMessages: [
        createAssistantToolMessage(interventionResult),
        createAssistantToolMessage(counterfactualResult),
      ],
    });

    expect(scene?.kind).toBe("simulation");
    if (scene?.kind === "simulation") {
      expect(scene.simulationResult).toEqual(counterfactualResult);
    }
  });
});
