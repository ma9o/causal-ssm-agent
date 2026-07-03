import { describe, expect, it } from "vitest";
import { cursorTimestampMs, parseStageProgressEvent } from "./use-run-events";

describe("cursorTimestampMs", () => {
  it("parses the nanosecond prefix of an event cursor", () => {
    const cursor = "01750000000000000000-abcd1234.json";
    expect(cursorTimestampMs(cursor)).toBe(Math.floor(1_750_000_000_000_000_000 / 1_000_000));
  });

  it("returns undefined for malformed cursors", () => {
    expect(cursorTimestampMs("not-a-cursor.json")).toBeUndefined();
  });
});

describe("parseStageProgressEvent", () => {
  const cursor = "01750000000000000000-abcd1234.json";

  it("parses running events", () => {
    const event = parseStageProgressEvent({
      event: "nof1-causal-lab.pipeline-stage.running",
      payload: { stage_id: "stage-1a", status: "running" },
      cursor,
    });

    expect(event).toEqual({
      stageId: "stage-1a",
      status: "running",
      eventTime: cursorTimestampMs(cursor),
      error: undefined,
    });
  });

  it("parses failed events with error detail", () => {
    const event = parseStageProgressEvent({
      event: "nof1-causal-lab.pipeline-stage.failed",
      payload: {
        stage_id: "stage-2",
        status: "failed",
        error: { type: "RuntimeError", message: "boom" },
      },
      cursor,
    });

    expect(event?.status).toBe("failed");
    expect(event?.error).toEqual({ type: "RuntimeError", message: "boom" });
  });

  it("ignores non-stage-progress events", () => {
    expect(
      parseStageProgressEvent({
        event: "nof1-causal-lab.stage2.worker",
        payload: { stage_id: "stage-2", type: "worker", worker_id: 1, state: "running" },
        cursor,
      }),
    ).toBeNull();
  });

  it("ignores malformed payloads", () => {
    expect(
      parseStageProgressEvent({
        event: "nof1-causal-lab.pipeline-stage.running",
        payload: { stage_id: "not-a-stage", status: "running" },
        cursor,
      }),
    ).toBeNull();
  });
});
