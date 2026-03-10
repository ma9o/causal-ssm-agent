import { describe, expect, it } from "vitest";
import {
  applyStageUpdate,
  initialProgress,
  mapPrefectTaskState,
  type PipelineProgress,
} from "./pipeline-progress";

describe("initialProgress", () => {
  it("starts with every stage pending", () => {
    const progress = initialProgress();

    expect(progress.currentStage).toBeNull();
    expect(progress.isComplete).toBe(false);
    expect(progress.isFailed).toBe(false);
    expect(Object.values(progress.stages).every((status) => status === "pending")).toBe(true);
  });
});

describe("applyStageUpdate", () => {
  it("tracks only stages Prefect has actually started", () => {
    let progress = initialProgress();

    progress = applyStageUpdate(progress, "stage-0", "running", 1000);
    progress = applyStageUpdate(progress, "stage-0", "completed", 2000);

    expect(progress.stages["stage-0"]).toBe("completed");
    expect(progress.stages["stage-1a"]).toBe("pending");
    expect(progress.currentStage).toBeNull();
    expect(progress.timings["stage-0"]).toEqual({ startedAt: 1000, completedAt: 2000 });
  });

  it("fills in terminal timings even when the first event is terminal", () => {
    const progress = applyStageUpdate(undefined, "stage-1a", "completed", 3500);

    expect(progress.timings["stage-1a"]).toEqual({ startedAt: 3500, completedAt: 3500 });
    expect(progress.currentStage).toBeNull();
  });

  it("does not regress a terminal stage back to running", () => {
    let progress = initialProgress();

    progress = applyStageUpdate(progress, "stage-1b", "running", 1000);
    progress = applyStageUpdate(progress, "stage-1b", "completed", 2000);
    const afterCompletion = progress;
    progress = applyStageUpdate(progress, "stage-1b", "running", 3000);

    expect(progress).toEqual(afterCompletion);
  });

  it("preserves outcome-level failure flags while later stages update", () => {
    const progressWithFailedOutcome: PipelineProgress = {
      ...initialProgress(),
      isFailed: true,
    };

    const progress = applyStageUpdate(progressWithFailedOutcome, "stage-2", "running", 4000);

    expect(progress.isFailed).toBe(true);
    expect(progress.currentStage).toBe("stage-2");
  });
});

describe("mapPrefectTaskState", () => {
  it("maps running and terminal Prefect states", () => {
    expect(mapPrefectTaskState("running")).toBe("running");
    expect(mapPrefectTaskState("COMPLETED")).toBe("completed");
    expect(mapPrefectTaskState("failed")).toBe("failed");
    expect(mapPrefectTaskState("cancelled")).toBe("failed");
    expect(mapPrefectTaskState("pending")).toBeNull();
  });
});
