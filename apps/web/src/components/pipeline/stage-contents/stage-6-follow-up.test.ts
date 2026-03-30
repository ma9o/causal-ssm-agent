import { describe, expect, it } from "vitest";
import type { UIMessage } from "ai";
import {
  counterfactualResult,
  interventionResult,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";
import { extractLatestStage6SimulationResult } from "./stage-6-follow-up";

function createAssistantToolMessage(
  toolName: string,
  output: unknown,
): UIMessage {
  return {
    id: `${toolName}-message`,
    role: "assistant",
    parts: [
      { type: "text", text: "Tool finished." },
      {
        type: "dynamic-tool",
        toolCallId: `${toolName}-call`,
        toolName,
        state: "output-available",
        input: {},
        output,
      },
    ],
  };
}

describe("extractLatestStage6SimulationResult", () => {
  it("returns the latest stage-6 simulation tool output", () => {
    const messages: UIMessage[] = [
      createAssistantToolMessage("simulate_intervention", interventionResult),
      createAssistantToolMessage("simulate_counterfactual", counterfactualResult),
    ];

    expect(extractLatestStage6SimulationResult(messages)).toEqual(counterfactualResult);
  });

  it("ignores tool errors and unrelated tool outputs", () => {
    const messages: UIMessage[] = [
      createAssistantToolMessage("simulate_intervention", { error: "No ID estimand." }),
      createAssistantToolMessage("search_literature", { hits: [] }),
    ];

    expect(extractLatestStage6SimulationResult(messages)).toBeNull();
  });
});
