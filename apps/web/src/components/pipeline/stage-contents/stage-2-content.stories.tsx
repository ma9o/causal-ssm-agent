import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@nof1-causal-lab/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import {
  EMPTY_STAGE2_REPLAY_STATE,
  STAGE2_EVENT_PREFIX,
  applyStage2Event,
  getStage2RequestsPerMinute,
  listStage2Workers,
  parseStage2Event,
  summarizeStage2State,
  type Stage2EventRecord,
  type Stage2ReplayState,
  type Stage2WorkerRecord,
  type Stage2WorkerState,
} from "@/lib/stage2-runtime";
import Stage2Content from "./stage-2-content";
import { Stage2RunningView } from "./stage-2-running-content";
import { StageStoryTemplate } from "../stage-story-template";
import { useEffect, useMemo, useState } from "react";
import { stage2Data as data } from "@/components/__fixtures__/stage2-data";

const stage = STAGES.find((s) => s.id === "stage-2")!;
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

/* ── Mock event replay for the large running story ── */

const STAGE2_STORY_TOTAL_WORKERS = 1_000;
const STAGE2_STORY_MAX_RUNNING = 72;
const STAGE2_STORY_MAX_RPM = 450;
const STAGE2_STORY_INTERVAL_MS = 700;
const STAGE2_STORY_FAILED_WORKERS = 12;

type RunningStage2StoryWorker = {
  workerId: number;
  readyFrame: number;
};

function planEvent(occurred: string): Stage2EventRecord {
  return {
    event: `${STAGE2_EVENT_PREFIX}plan`,
    occurred,
    payload: {
      stage_id: "stage-2",
      type: "plan",
      total_workers: STAGE2_STORY_TOTAL_WORKERS,
      max_concurrent_workers: STAGE2_STORY_MAX_RUNNING,
      max_rpm: STAGE2_STORY_MAX_RPM,
    },
  };
}

function workerEvent({
  error,
  nExtractions,
  nLlmCalls,
  nWindows,
  occurred,
  state,
  workerId,
}: {
  error?: string;
  nExtractions?: number;
  nLlmCalls?: number;
  nWindows: number;
  occurred: string;
  state: Stage2WorkerState;
  workerId: number;
}): Stage2EventRecord {
  return {
    event: `${STAGE2_EVENT_PREFIX}worker`,
    occurred,
    payload: {
      stage_id: "stage-2",
      type: "worker",
      worker_id: workerId,
      state,
      n_windows: nWindows,
      ...(nExtractions !== undefined ? { n_extractions: nExtractions } : {}),
      ...(nLlmCalls !== undefined ? { n_llm_calls: nLlmCalls } : {}),
      ...(error ? { error } : {}),
    },
  };
}

function snapshotEvent({
  completed,
  failed,
  occurred,
  pending,
  rpm,
  running,
}: {
  completed: number;
  failed: number;
  occurred: string;
  pending: number;
  rpm: number;
  running: number;
}): Stage2EventRecord {
  return {
    event: `${STAGE2_EVENT_PREFIX}snapshot`,
    occurred,
    payload: {
      stage_id: "stage-2",
      type: "snapshot",
      total_workers: STAGE2_STORY_TOTAL_WORKERS,
      pending_workers: pending,
      running_workers: running,
      completed_workers: completed,
      failed_workers: failed,
      llm_requests_last_60s: rpm,
    },
  };
}

function applyRawStage2Event(state: Stage2ReplayState, record: Stage2EventRecord) {
  const parsed = parseStage2Event(record);
  return parsed ? applyStage2Event(state, parsed) : state;
}

function randomInt(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function sampleWorkerDurationFrames() {
  const draw = Math.random();
  if (draw < 0.16) return randomInt(1, 2);
  if (draw < 0.7) return randomInt(3, 5);
  if (draw < 0.93) return randomInt(6, 10);
  return randomInt(11, 18);
}

function sampleCompletionBudget(eligibleWorkers: number) {
  if (eligibleWorkers === 0) return 0;
  if (eligibleWorkers < 8) return randomInt(1, eligibleWorkers);

  const draw = Math.random();
  if (draw < 0.14) return randomInt(1, Math.min(7, eligibleWorkers));
  if (draw < 0.82) return randomInt(8, Math.min(34, eligibleWorkers));
  return randomInt(35, Math.min(64, eligibleWorkers));
}

function shuffledWorkers(workers: RunningStage2StoryWorker[]) {
  const result = [...workers];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = randomInt(0, index);
    [result[index], result[swapIndex]] = [result[swapIndex]!, result[index]!];
  }
  return result;
}

function createStage2ReplayController() {
  let frame = 0;
  let nextWorkerId = 0;
  let pending = STAGE2_STORY_TOTAL_WORKERS;
  let running = 0;
  let completed = 0;
  let failed = 0;
  let clockMs = Date.UTC(2026, 3, 2, 10, 0, 0);
  const runningWorkers: RunningStage2StoryWorker[] = [];
  const failedWorkerIds = new Set<number>();

  while (failedWorkerIds.size < STAGE2_STORY_FAILED_WORKERS) {
    failedWorkerIds.add(Math.floor(Math.random() * STAGE2_STORY_TOTAL_WORKERS));
  }

  function nextOccurred() {
    clockMs += Math.random() < 0.08 ? randomInt(1_500, 4_200) : randomInt(120, 900);
    return new Date(clockMs).toISOString();
  }

  function workerWindowCount(workerId: number) {
    return (workerId % 4) + 1;
  }

  function startWorker(workerId: number): Stage2EventRecord {
    pending -= 1;
    running += 1;
    runningWorkers.push({
      workerId,
      readyFrame: frame + sampleWorkerDurationFrames(),
    });
    return workerEvent({
      workerId,
      state: "running",
      nWindows: workerWindowCount(workerId),
      occurred: nextOccurred(),
    });
  }

  function finishWorker(workerId: number): Stage2EventRecord {
    running -= 1;
    const didFail = failedWorkerIds.has(workerId);
    const nWindows = workerWindowCount(workerId);

    if (didFail) {
      failed += 1;
      return workerEvent({
        workerId,
        state: "failed",
        nWindows,
        error: "Error code: 402",
        occurred: nextOccurred(),
      });
    }

    completed += 1;
    return workerEvent({
      workerId,
      state: "completed",
      nWindows,
      nExtractions: nWindows * 6,
      nLlmCalls: 1,
      occurred: nextOccurred(),
    });
  }

  function refill(events: Stage2EventRecord[]) {
    while (
      runningWorkers.length < STAGE2_STORY_MAX_RUNNING &&
      nextWorkerId < STAGE2_STORY_TOTAL_WORKERS
    ) {
      events.push(startWorker(nextWorkerId));
      nextWorkerId += 1;
    }
  }

  function currentRpm() {
    if (completed + failed === 0) return 0;
    return Math.min(
      STAGE2_STORY_MAX_RPM,
      Math.round(190 + running * 2.1 + Math.random() * 90 + Math.min(80, (completed + failed) / 5)),
    );
  }

  function completeEligibleWorkers(events: Stage2EventRecord[]) {
    const eligibleWorkers = runningWorkers.filter((worker) => worker.readyFrame <= frame);
    const completionBudget = sampleCompletionBudget(eligibleWorkers.length);
    const workersToComplete = shuffledWorkers(eligibleWorkers).slice(0, completionBudget);

    for (const worker of workersToComplete) {
      const workerIndex = runningWorkers.findIndex(
        (candidate) => candidate.workerId === worker.workerId,
      );
      if (workerIndex !== -1) {
        runningWorkers.splice(workerIndex, 1);
        events.push(finishWorker(worker.workerId));
      }
    }
  }

  return {
    isFinished() {
      return completed + failed >= STAGE2_STORY_TOTAL_WORKERS;
    },
    nextFrame(): Stage2EventRecord[] {
      const events: Stage2EventRecord[] = [];
      if (frame === 0) {
        events.push(planEvent(nextOccurred()));
      }

      if (frame > 0) {
        completeEligibleWorkers(events);
      }

      refill(events);
      events.push(
        snapshotEvent({
          completed,
          failed,
          pending,
          running,
          rpm: currentRpm(),
          occurred: nextOccurred(),
        }),
      );
      frame += 1;
      return events;
    },
  };
}

function AnimatedStage2Running() {
  const [state, setState] = useState<Stage2ReplayState>(EMPTY_STAGE2_REPLAY_STATE);

  useEffect(() => {
    let current = EMPTY_STAGE2_REPLAY_STATE;
    let controller = createStage2ReplayController();

    function publishNextFrame() {
      if (controller.isFinished()) {
        controller = createStage2ReplayController();
        current = EMPTY_STAGE2_REPLAY_STATE;
      }

      current = controller.nextFrame().reduce<Stage2ReplayState>(applyRawStage2Event, current);
      setState(current);
    }

    publishNextFrame();
    const timer = setInterval(publishNextFrame, STAGE2_STORY_INTERVAL_MS);
    return () => clearInterval(timer);
  }, []);

  const workers = useMemo(() => listStage2Workers(state), [state]);
  const summary = useMemo(() => summarizeStage2State(state), [state]);
  const rpm = useMemo(() => getStage2RequestsPerMinute(state), [state]);

  return (
    <Stage2RunningView
      workers={workers}
      total={summary.total}
      failed={summary.failed}
      running={summary.running}
      rpm={rpm}
      maxRpm={state.plan?.max_rpm ?? STAGE2_STORY_MAX_RPM}
    />
  );
}

const meta = {
  title: "Pipeline/Stages/2 – Data Extraction/Panel",
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
    runningContent: (
      <Stage2RunningView workers={mockWorkers} total={mockWorkers.length} rpm={420} />
    ),
  }),
  name: "Running (High RPM)",
};

export const Running1kWorkers = {
  name: "Running (1000 Workers)",
  render: () => (
    <StageStoryTemplate stage={stage} status="running" runningContent={<AnimatedStage2Running />} />
  ),
  parameters: {
    docs: {
      description: {
        story:
          "Replays raw Stage 2 plan, worker, and snapshot events through the production parser and reducer so the 1000-worker running view streams from the first frame.",
      },
    },
  },
};

export const Completed = createCompletedStageStory({
  stage,
  args: { data, workspaceId },
  elapsedMs: 45_200,
  renderContent: (args) => <Stage2Content {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
