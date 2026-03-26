import { describe, expect, it } from "vitest";
import type { AnalysisStageRun } from "@/lib/api/analysis";
import { patchStageRun, summarizeStageProgressEvents, type StageRuntimeEventRecord } from "./stage-runtime";

function emptyStageRun(): AnalysisStageRun {
  return {
    ownerRootFlowRunId: null,
    stageSubflowRunId: null,
    logFlowRunIds: [],
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
      logFlowRunIds: ["subflow-123"],
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
      logFlowRunIds: ["subflow-123"],
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
    expect(nextStageRun.logFlowRunIds).toEqual(["subflow-123"]);
  });
});
