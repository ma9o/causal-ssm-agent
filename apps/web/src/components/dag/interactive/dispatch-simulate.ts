import type { AnalysisSimulationResult } from "../intervention-dag-types";
import type { SimulateFn } from "./simulate-input";

/**
 * The production `onSimulate`: run a do() scenario by dispatching the ranking
 * `simulate` tool directly (no LLM) via `POST /api/tools/dispatch`. Returns
 * the `SimulateScenarioResult`.
 */
export function createSimulateDispatch(workspaceId: string): SimulateFn {
  return async (input): Promise<AnalysisSimulationResult> => {
    const response = await fetch("/api/tools/dispatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspaceId, contextId: "ranking", tool: "simulate", input }),
    });
    const payload = (await response.json()) as { output?: unknown; error?: string };
    if (!response.ok || payload.error) {
      throw new Error(payload.error ?? `simulate dispatch failed (HTTP ${response.status})`);
    }
    return payload.output as AnalysisSimulationResult;
  };
}
