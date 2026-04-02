import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage2Data } from "@causal-ssm/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import type { Stage2WorkerRecord } from "@/lib/stage2-runtime";
import Stage2Content from "./stage-2-content";
import { Stage2RunningView } from "./stage-2-running-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-2.json";

const stage = STAGES.find((s) => s.id === "stage-2")!;
const data = fixture as unknown as Stage2Data;
const workspaceId = "demo-user";

/* ── Mock data for running state ── */

function makeWorkers(
  completed: number,
  running: number,
  failed: number,
  pending: number,
): Stage2WorkerRecord[] {
  let idx = 0;
  return [
    ...Array.from({ length: completed }, () => ({
      worker_id: idx++,
      state: "completed" as const,
      n_windows: 1,
      n_extractions: 6,
      n_llm_calls: 1,
      error: null,
      completed_at: "2026-04-02T10:00:00.000Z",
    })),
    ...Array.from({ length: running }, () => ({
      worker_id: idx++,
      state: "running" as const,
      n_windows: 1,
      n_extractions: null,
      n_llm_calls: null,
      error: null,
      completed_at: null,
    })),
    ...Array.from({ length: failed }, () => ({
      worker_id: idx++,
      state: "failed" as const,
      n_windows: 1,
      n_extractions: null,
      n_llm_calls: null,
      error: "Error code: 402",
      completed_at: "2026-04-02T10:00:00.000Z",
    })),
    ...Array.from({ length: pending }, () => ({
      worker_id: idx++,
      state: "pending" as const,
      n_windows: 0,
      n_extractions: null,
      n_llm_calls: null,
      error: null,
      completed_at: null,
    })),
  ];
}

const mockWorkers = makeWorkers(8, 3, 1, 4);
const mockWorkers1k = makeWorkers(620, 45, 12, 323);

const meta = {
  title: "Pipeline/Stages/2 – Data Extraction",
  component: Stage2Content,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage2Content>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running", {
  runningContent: <Stage2RunningView workers={mockWorkers} total={mockWorkers.length} rpm={285} />,
});

export const RunningHighRpm = {
  ...createStageStatusStory(stage, "running", {
    runningContent: <Stage2RunningView workers={mockWorkers} total={mockWorkers.length} rpm={420} />,
  }),
  name: "Running (High RPM)",
};

export const Running1kWorkers = {
  ...createStageStatusStory(stage, "running", {
    runningContent: <Stage2RunningView workers={mockWorkers1k} total={mockWorkers1k.length} rpm={440} />,
  }),
  name: "Running (1000 Workers)",
};

export const Completed = createCompletedStageStory({
  stage,
  args: { data, workspaceId },
  outcome: data.outcome,
  elapsedMs: 45_200,
  renderContent: (args) => <Stage2Content {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
