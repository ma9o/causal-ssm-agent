import { describe, expect, it } from "vitest";
import {
  EMPTY_STAGE2_REPLAY_STATE,
  STAGE2_EVENT_PREFIX,
  applyStage2Event,
  getStage2RequestsPerMinute,
  parseStage2Event,
  reduceStage2Events,
  summarizeStage2State,
} from "./stage2-runtime";

describe("stage2-runtime", () => {
  it("parses and reduces Stage 2 plan/worker events", () => {
    const planRecord = {
      event: `${STAGE2_EVENT_PREFIX}plan`,
      occurred: "2026-04-02T10:00:00.000Z",
      payload: {
        type: "plan",
        total_workers: 3,
        max_concurrent_workers: 30,
        max_rpm: 450,
      },
    };
    const runningRecord = {
      event: `${STAGE2_EVENT_PREFIX}worker`,
      occurred: "2026-04-02T10:00:01.000Z",
      payload: {
        type: "worker",
        worker_id: 0,
        state: "running",
        n_windows: 1,
      },
    };
    const completedRecord = {
      event: `${STAGE2_EVENT_PREFIX}worker`,
      occurred: "2026-04-02T10:00:02.000Z",
      payload: {
        type: "worker",
        worker_id: 0,
        state: "completed",
        n_windows: 1,
        n_extractions: 6,
        n_llm_calls: 1,
      },
    };
    const snapshotRecord = {
      event: `${STAGE2_EVENT_PREFIX}snapshot`,
      occurred: "2026-04-02T10:00:03.000Z",
      payload: {
        type: "snapshot",
        total_workers: 3,
        pending_workers: 2,
        running_workers: 0,
        completed_workers: 1,
        failed_workers: 0,
        llm_requests_last_60s: 9,
      },
    };

    expect(parseStage2Event(planRecord)?.type).toBe("plan");
    expect(parseStage2Event(snapshotRecord)?.type).toBe("snapshot");
    expect(parseStage2Event(completedRecord)?.type).toBe("worker");

    expect(reduceStage2Events([planRecord, runningRecord, completedRecord, snapshotRecord])).toEqual({
      plan: {
        total_workers: 3,
        max_concurrent_workers: 30,
        max_rpm: 450,
      },
      snapshot: {
        total_workers: 3,
        pending_workers: 2,
        running_workers: 0,
        completed_workers: 1,
        failed_workers: 0,
        llm_requests_last_60s: 9,
      },
      workers: {
        "0": {
          worker_id: 0,
          state: "completed",
          n_windows: 1,
          n_extractions: 6,
          n_llm_calls: 1,
          error: null,
          completed_at: "2026-04-02T10:00:02.000Z",
        },
        "1": {
          worker_id: 1,
          state: "pending",
          n_windows: 0,
          n_extractions: null,
          n_llm_calls: null,
          error: null,
          completed_at: null,
        },
        "2": {
          worker_id: 2,
          state: "pending",
          n_windows: 0,
          n_extractions: null,
          n_llm_calls: null,
          error: null,
          completed_at: null,
        },
      },
    });
  });

  it("applies live worker events incrementally without regressing terminal states", () => {
    const next = applyStage2Event(EMPTY_STAGE2_REPLAY_STATE, {
      type: "worker",
      worker: {
        worker_id: 8,
        state: "completed",
        n_windows: 1,
        n_extractions: 4,
        n_llm_calls: 2,
        error: null,
        completed_at: "2026-04-02T10:00:30.000Z",
      },
    });
    const regressed = applyStage2Event(next, {
      type: "worker",
      worker: {
        worker_id: 8,
        state: "running",
        n_windows: 1,
        n_extractions: null,
        n_llm_calls: null,
        error: null,
        completed_at: null,
      },
    });

    expect(regressed.workers["8"]?.state).toBe("completed");
    expect(regressed.workers["8"]?.n_llm_calls).toBe(2);
  });

  it("derives summary counts and rolling RPM from replay state", () => {
    const state = reduceStage2Events([
      {
        event: `${STAGE2_EVENT_PREFIX}plan`,
        occurred: "2026-04-02T10:00:00.000Z",
        payload: {
          type: "plan",
          total_workers: 2,
          max_concurrent_workers: 30,
          max_rpm: 450,
        },
      },
      {
        event: `${STAGE2_EVENT_PREFIX}worker`,
        occurred: "2026-04-02T10:00:10.000Z",
        payload: {
          type: "worker",
          worker_id: 0,
          state: "completed",
          n_windows: 1,
          n_extractions: 6,
          n_llm_calls: 3,
        },
      },
      {
        event: `${STAGE2_EVENT_PREFIX}worker`,
        occurred: "2026-04-02T10:00:11.000Z",
        payload: {
          type: "worker",
          worker_id: 1,
          state: "running",
          n_windows: 1,
        },
      },
      {
        event: `${STAGE2_EVENT_PREFIX}snapshot`,
        occurred: "2026-04-02T10:00:12.000Z",
        payload: {
          type: "snapshot",
          total_workers: 2,
          pending_workers: 0,
          running_workers: 1,
          completed_workers: 1,
          failed_workers: 0,
          llm_requests_last_60s: 7,
        },
      },
    ]);

    expect(summarizeStage2State(state)).toEqual({
      total: 2,
      pending: 0,
      running: 1,
      completed: 1,
      failed: 0,
    });
    expect(getStage2RequestsPerMinute(state, Date.parse("2026-04-02T10:00:40.000Z"))).toBe(7);
  });
});
