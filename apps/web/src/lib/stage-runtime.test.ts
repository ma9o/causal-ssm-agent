import { describe, expect, it } from "vitest";
import type { AnalysisStageRun } from "@/lib/api/analysis";
import {
  getStageRunStatus,
  patchStageRun,
  resolveStageObservedStatus,
  summarizeStageProgressEvents,
  type StageRuntimeEventRecord,
} from "./stage-runtime";

function emptyStageRun(): AnalysisStageRun {
  return {
    ownerRootFlowRunId: null,
    stageSubflowRunId: null,
    initialLogFlowRunIds: [],
    execution: null,
  };
}

describe("stage runtime reducers", () => {
  it("keeps the earliest running timestamp when a nested stage later attaches its subflow id", () => {
    const events: StageRuntimeEventRecord[] = [
      {
        status: "running",
        occurred: "2026-03-10T08:00:00.000Z",
      },
      {
        status: "running",
        occurred: "2026-03-10T08:00:05.000Z",
        stageSubflowRunId: "subflow-123",
      },
      {
        status: "completed",
        occurred: "2026-03-10T08:00:10.000Z",
      },
    ];

    expect(summarizeStageProgressEvents(events)).toEqual({
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-10T08:00:00.000Z",
        endTime: "2026-03-10T08:00:10.000Z",
      },
      stageSubflowRunId: "subflow-123",
      initialLogFlowRunIds: ["subflow-123"],
    });

    const stageRun = events.reduce(
      (current, event) => patchStageRun(current, "root-123", event),
      emptyStageRun(),
    );

    expect(stageRun).toEqual({
      ownerRootFlowRunId: "root-123",
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-10T08:00:00.000Z",
        endTime: "2026-03-10T08:00:10.000Z",
      },
      stageSubflowRunId: "subflow-123",
      initialLogFlowRunIds: ["subflow-123"],
    });
  });

  it("does not regress a completed stage back to running when an old running event arrives late", () => {
    const completedStageRun = patchStageRun(
      patchStageRun(emptyStageRun(), "root-123", {
        status: "running",
        occurred: "2026-03-10T08:00:00.000Z",
      }),
      "root-123",
      {
        status: "completed",
        occurred: "2026-03-10T08:00:10.000Z",
      },
    );

    const nextStageRun = patchStageRun(completedStageRun, "root-123", {
      status: "running",
      occurred: "2026-03-10T08:00:05.000Z",
      stageSubflowRunId: "subflow-123",
    });

    expect(nextStageRun.execution).toEqual({
      stateType: "COMPLETED",
      startTime: "2026-03-10T08:00:00.000Z",
      endTime: "2026-03-10T08:00:10.000Z",
    });
    expect(nextStageRun.stageSubflowRunId).toBe("subflow-123");
    expect(nextStageRun.initialLogFlowRunIds).toEqual(["subflow-123"]);
  });

  it("derives a stage status from the manifest execution summary", () => {
    expect(
      getStageRunStatus({
        execution: {
          stateType: "COMPLETED",
          startTime: "2026-03-10T08:00:00.000Z",
          endTime: "2026-03-10T08:00:10.000Z",
        },
      }),
    ).toBe("completed");

    expect(
      getStageRunStatus({
        execution: {
          stateType: "FAILED",
          startTime: "2026-03-10T08:00:00.000Z",
          endTime: "2026-03-10T08:00:10.000Z",
        },
      }),
    ).toBe("failed");
  });

  it("lets terminal manifest execution override a stale running section status", () => {
    const completedStageRun: AnalysisStageRun = {
      ownerRootFlowRunId: "root-123",
      stageSubflowRunId: "subflow-123",
      initialLogFlowRunIds: ["subflow-123"],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-10T08:00:00.000Z",
        endTime: "2026-03-10T08:00:10.000Z",
      },
    };

    expect(resolveStageObservedStatus("running", completedStageRun)).toBe("completed");
    expect(resolveStageObservedStatus("pending", completedStageRun)).toBe("completed");
    expect(resolveStageObservedStatus("completed", completedStageRun)).toBe("completed");
  });
});
