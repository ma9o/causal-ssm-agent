import {
  EMPTY_STAGE4_REPLAY_STATE,
  STAGE4_EVENT_PREFIX,
  applyStage4Event,
  parseStage4Event,
  reduceStage4Events,
} from "./stage4-runtime";
import { describe, expect, it } from "vitest";

describe("stage4-runtime", () => {
  it("parses and reduces Stage 4 graph/snapshot events", () => {
    const graphRecord = {
      event: `${STAGE4_EVENT_PREFIX}graph`,
      payload: {
        type: "graph",
        nodes: [
          { id: "indicator:x", kind: "indicator_decision", label: "X", phase: "model_decisions" },
        ],
        edges: [],
        phases: [{ id: "model_decisions", label: "Model Decisions" }],
      },
    };
    const snapshotRecord = {
      event: `${STAGE4_EVENT_PREFIX}snapshot`,
      payload: {
        type: "snapshot",
        cursor: { kind: "block", block_id: "indicator:x" },
        block_status: { "indicator:x": "pending" },
        model_spec_locked: false,
        repair_campaign: null,
        phase: "model_decisions",
      },
    };

    expect(parseStage4Event(graphRecord)?.type).toBe("graph");
    expect(parseStage4Event(snapshotRecord)?.type).toBe("snapshot");

    expect(reduceStage4Events([graphRecord, snapshotRecord])).toEqual({
      graph: {
        nodes: graphRecord.payload.nodes,
        edges: graphRecord.payload.edges,
        phases: graphRecord.payload.phases,
      },
      snapshot: {
        cursor: snapshotRecord.payload.cursor,
        block_status: snapshotRecord.payload.block_status,
        model_spec_locked: snapshotRecord.payload.model_spec_locked,
        repair_campaign: snapshotRecord.payload.repair_campaign,
        phase: snapshotRecord.payload.phase,
      },
    });
  });

  it("applies live events incrementally", () => {
    const next = applyStage4Event(EMPTY_STAGE4_REPLAY_STATE, {
      type: "snapshot",
      snapshot: {
        cursor: { kind: "done" },
        block_status: {},
        model_spec_locked: true,
        repair_campaign: null,
        phase: "done",
      },
    });

    expect(next.snapshot?.phase).toBe("done");
    expect(next.graph).toBeNull();
  });
});
