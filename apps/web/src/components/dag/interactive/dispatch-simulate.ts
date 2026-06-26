import type { Stage6SimulationResult } from "../intervention-dag-types";
import type { SimulateFn } from "./simulate-input";

/**
 * The production `onSimulate`: run a do() scenario by dispatching the stage-6
 * `simulate` tool directly (no LLM) via `POST /api/refine/dispatch`, the same
 * non-LLM seam the suggestion chips use. Returns the `SimulateScenarioResult`.
 */
export function createSimulateDispatch(workspaceId: string): SimulateFn {
  return async (input): Promise<Stage6SimulationResult> => {
    const response = await fetch("/api/refine/dispatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspaceId, stageId: "stage-6", tool: "simulate", input }),
    });
    const payload = (await response.json()) as { output?: unknown; error?: string };
    if (!response.ok || payload.error) {
      throw new Error(payload.error ?? `simulate dispatch failed (HTTP ${response.status})`);
    }
    return payload.output as Stage6SimulationResult;
  };
}
