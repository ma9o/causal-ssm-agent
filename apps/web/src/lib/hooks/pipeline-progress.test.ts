import { describe, expect, it } from "vitest";
import {
  applyStageUpdate,
  initialProgress,
  mapExecutionStateType,
  restartStageAttempt,
  type PipelineProgress,
} from "./pipeline-progress";

describe("initialProgress", () => {
  it("starts every stage pending with no errors", () => {
    const progress = initialProgress();

    expect(progress.stages["stage-0"]).toBe("pending");
    expect(progress.stages["stage-6"]).toBe("pending");
    expect(progress.stageErrors).toEqual({});
    expect(progress.currentStage).toBeNull();
    expect(progress.isComplete).toBe(false);
    expect(progress.isFailed).toBe(false);
  });
});

describe("mapExecutionStateType", () => {
  it("maps execution state types onto run statuses", () => {
    expect(mapExecutionStateType("RUNNING")).toBe("running");
    expect(mapExecutionStateType("COMPLETED")).toBe("completed");
    expect(mapExecutionStateType("FAILED")).toBe("failed");
    expect(mapExecutionStateType("UNKNOWN")).toBeNull();
  });
});

describe("applyStageUpdate", () => {
  it("records running and completion timings", () => {
    let progress: PipelineProgress | undefined;
    progress = applyStageUpdate(progress, "stage-0", "running", 1_000);
    progress = applyStageUpdate(progress, "stage-0", "completed", 5_000);

    expect(progress.stages["stage-0"]).toBe("completed");
    expect(progress.timings["stage-0"]).toEqual({ startedAt: 1_000, completedAt: 5_000 });
  });

  it("never regresses a completed stage back to running", () => {
    let progress = applyStageUpdate(undefined, "stage-0", "completed", 5_000);
    progress = applyStageUpdate(progress, "stage-0", "running", 6_000);

    expect(progress.stages["stage-0"]).toBe("completed");
  });

  it("tracks the current running stage", () => {
    let progress = applyStageUpdate(undefined, "stage-0", "completed", 1_000);
    progress = applyStageUpdate(progress, "stage-1a", "running", 2_000);

    expect(progress.currentStage).toBe("stage-1a");
  });

  it("stores the failure detail and flips isFailed", () => {
    const progress = applyStageUpdate(
      undefined,
      "stage-1a",
      "failed",
      2_000,
      "SchemaValidationError: bad payload",
    );

    expect(progress.stages["stage-1a"]).toBe("failed");
    expect(progress.stageErrors["stage-1a"]).toBe("SchemaValidationError: bad payload");
    expect(progress.isFailed).toBe(true);
  });

  it("restarts a failed stage on a new running attempt", () => {
    let progress = applyStageUpdate(undefined, "stage-1a", "failed", 2_000, "boom");
    progress = restartStageAttempt(progress, "stage-1a", 3_000);

    expect(progress.stages["stage-1a"]).toBe("running");
    expect(progress.timings["stage-1a"]).toEqual({ startedAt: 3_000 });
    expect(progress.stageErrors["stage-1a"]).toBeUndefined();
    expect(progress.isFailed).toBe(false);
  });

  it("restarts a completed stage when its inputs are recomputed", () => {
    let progress = applyStageUpdate(undefined, "stage-1b", "completed", 2_000);
    progress = restartStageAttempt(progress, "stage-1b", 9_000);
    progress = applyStageUpdate(progress, "stage-1b", "completed", 12_000);

    expect(progress.stages["stage-1b"]).toBe("completed");
    expect(progress.timings["stage-1b"]).toEqual({ startedAt: 9_000, completedAt: 12_000 });
  });

  it("keeps the original start when the stage is already running", () => {
    let progress = applyStageUpdate(undefined, "stage-2", "running", 1_000);
    progress = restartStageAttempt(progress, "stage-2", 4_000);

    expect(progress.timings["stage-2"]).toEqual({ startedAt: 1_000, completedAt: undefined });
  });

  it("marks the pipeline complete when every stage completed", () => {
    let progress: PipelineProgress | undefined;
    for (const stageId of [
      "stage-0",
      "stage-1a",
      "stage-1b",
      "stage-2",
      "stage-3",
      "stage-4",
      "stage-5b",
      "stage-6",
    ] as const) {
      progress = applyStageUpdate(progress, stageId, "completed");
    }

    expect(progress?.isComplete).toBe(true);
  });
});
