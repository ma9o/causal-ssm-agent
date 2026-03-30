import type { Stage6SimulationResult } from "@/components/dag/intervention-dag-types";
import type { UIMessage } from "ai";

const STAGE6_TOOL_NAMES = new Set([
  "simulate_intervention",
  "simulate_counterfactual",
]);

function isStage6SimulationResult(value: unknown): value is Stage6SimulationResult {
  if (typeof value !== "object" || value == null) {
    return false;
  }

  const candidate = value as Partial<Stage6SimulationResult> & { error?: unknown };
  if (candidate.error != null) {
    return false;
  }

  return (
    (candidate.rung === 2 || candidate.rung === 3) &&
    typeof candidate.outcome === "string" &&
    typeof candidate.action?.variable === "string"
  );
}

export function extractLatestStage6SimulationResult(
  messages: UIMessage[],
): Stage6SimulationResult | null {
  for (let messageIndex = messages.length - 1; messageIndex >= 0; messageIndex -= 1) {
    const message = messages[messageIndex];
    if (message?.role !== "assistant") {
      continue;
    }

    for (let partIndex = message.parts.length - 1; partIndex >= 0; partIndex -= 1) {
      const part = message.parts[partIndex];
      if (
        part.type !== "dynamic-tool" ||
        part.state !== "output-available" ||
        !STAGE6_TOOL_NAMES.has(part.toolName)
      ) {
        continue;
      }

      if (isStage6SimulationResult(part.output)) {
        return part.output;
      }
    }
  }

  return null;
}
