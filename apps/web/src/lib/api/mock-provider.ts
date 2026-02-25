import type { StageId } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";

export const MOCK_RUN_ID = "mock-run-001";

const STAGE_DELAYS_MS: Record<StageId, number> = {
  "stage-0": 500,
  "stage-1a": 2000,
  "stage-1b": 3500,
  "stage-2": 5000,
  "stage-3": 6500,
  "stage-4": 8000,
  "stage-4b": 9500,
  "stage-5": 11000,
  "stage-6": 12500,
};

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
  const timers: ReturnType<typeof setTimeout>[] = [];

  for (const stage of STAGES) {
    const delay = STAGE_DELAYS_MS[stage.id];

    timers.push(
      setTimeout(() => {
        handlers.onStageStart(stage.id);
      }, delay - 400),
    );

    timers.push(
      setTimeout(() => {
        handlers.onStageComplete(stage.id);
      }, delay),
    );
  }

  return () => {
    for (const t of timers) clearTimeout(t);
  };
}
