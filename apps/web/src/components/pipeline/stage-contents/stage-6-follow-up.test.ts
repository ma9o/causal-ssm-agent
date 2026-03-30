import { describe, expect, it } from "vitest";
import type { UIMessage } from "ai";
import {
  counterfactualResult,
  interventionResult,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";
import { extractLatestStage6FollowUpSimulation } from "./stage-6-follow-up";

function createAssistantToolMessage(
  toolName: string,
  output: unknown,
  input: unknown = {},
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
        input,
        output,
      },
    ],
  };
}

describe("extractLatestStage6FollowUpSimulation", () => {
  it("returns the latest stage-6 simulation tool call and output", () => {
    const messages: UIMessage[] = [
      createAssistantToolMessage("simulate_intervention", interventionResult, {
        query: { horizon_days: 60 },
      }),
      createAssistantToolMessage("simulate_counterfactual", counterfactualResult, {
        query: { horizon_days: 45 },
      }),
    ];

    expect(extractLatestStage6FollowUpSimulation(messages)).toEqual({
      toolName: "simulate_counterfactual",
      input: {
        query: { horizon_days: 45 },
      },
      output: counterfactualResult,
    });
  });

  it("ignores tool errors and unrelated tool outputs", () => {
    const messages: UIMessage[] = [
      createAssistantToolMessage("simulate_intervention", { error: "No ID estimand." }),
      createAssistantToolMessage("search_literature", { hits: [] }),
    ];

    expect(extractLatestStage6FollowUpSimulation(messages)).toBeNull();
  });
});
