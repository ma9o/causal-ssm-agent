"use client";

import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { useWorkspaceView } from "@/lib/contexts/workspace-view-context";
import { LLMTracePanelView } from "./llm-trace-panel-view";

export { LLMTracePanelView } from "./llm-trace-panel-view";

/**
 * Read-only rendering of a stage's persisted LLM trace. Interactive
 * refinement moved out of the app entirely — an external agent drives the
 * episode machine over MCP/HTTP, and the journal is the conversation.
 * Clicking a `simulate` tool call still focuses that scenario in the
 * Stage 6 viewer.
 */
export function LLMTracePanel({ trace }: { trace: LLMTrace }) {
  const { selectedScenarioKey, selectScenario } = useWorkspaceView();

  return (
    <LLMTracePanelView
      trace={trace}
      selectedSimulationKey={selectedScenarioKey ?? undefined}
      onSelectSimulation={(key) => selectScenario(key)}
    />
  );
}
