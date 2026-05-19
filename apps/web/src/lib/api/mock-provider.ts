import type { StageId } from "@nof1-causal-lab/api-types";
import { STAGES } from "@nof1-causal-lab/api-types";

export function isMockMode(): boolean {
  const v = process.env.NEXT_PUBLIC_MOCK_DATA;
  return !!v && v !== "false";
}

/** Returns the fixture user ID, e.g. "DEFAULT". */
export function getMockFixture(): string {
  const v = process.env.NEXT_PUBLIC_MOCK_DATA;
  if (!v || v === "true") return "DEFAULT";
  const fixture = v.toUpperCase();
  return fixture === "DEMO_HEALTH" ? "DEMO" : fixture;
}

export interface MockEventHandler {
  onStageStart: (stageId: StageId) => void;
  onStageComplete: (stageId: StageId) => void;
}

export function simulatePipelineEvents(handlers: MockEventHandler): () => void {
  for (const stage of STAGES) {
    handlers.onStageStart(stage.id);
    handlers.onStageComplete(stage.id);
  }

  return () => {};
}
