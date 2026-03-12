import type { StageId } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";

export function isMockMode(): boolean {
  const v = process.env.NEXT_PUBLIC_MOCK_DATA;
  return !!v && v !== "false";
}

/** Returns the fixture directory name, e.g. "default". */
export function getMockFixture(): string {
  return process.env.NEXT_PUBLIC_MOCK_DATA || "default";
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
