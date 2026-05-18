import { describe, expect, it } from "vitest";
import { buildPrefectEventFilterMessage, parsePrefectStageProgressEvent } from "./use-run-events";

describe("buildPrefectEventFilterMessage", () => {
  it("subscribes to the run's custom stage events across a future time window", () => {
    const now = new Date("2026-03-10T08:06:15.000Z");

    expect(buildPrefectEventFilterMessage("run-123", now)).toEqual({
      type: "filter",
      filter: {
        event: { prefix: ["nof1-causal-lab."] },
        resource: {
          id: ["prefect.flow-run.run-123"],
        },
        occurred: {
          since: "2026-03-10T08:05:15.000Z",
          until: "2027-03-10T08:06:15.000Z",
        },
      },
    });
  });
});

describe("parsePrefectStageProgressEvent", () => {
  it("extracts a valid stage update from a custom Prefect event", () => {
    expect(
      parsePrefectStageProgressEvent({
        event: "nof1-causal-lab.pipeline-stage.completed",
        occurred: "2026-03-10T08:06:15.000Z",
        payload: {
          stage_id: "stage-2",
          status: "completed",
        },
      }),
    ).toEqual({
      stageId: "stage-2",
      status: "completed",
      eventTime: new Date("2026-03-10T08:06:15.000Z").getTime(),
      occurred: "2026-03-10T08:06:15.000Z",
    });
  });

  it("captures nested stage runtime metadata from the event payload", () => {
    expect(
      parsePrefectStageProgressEvent({
        event: "nof1-causal-lab.pipeline-stage.running",
        occurred: "2026-03-10T08:06:15.000Z",
        payload: {
          stage_id: "stage-4",
          status: "running",
          stage_subflow_run_id: "subflow-123",
          log_flow_run_ids: ["subflow-123"],
        },
      }),
    ).toEqual({
      stageId: "stage-4",
      status: "running",
      eventTime: new Date("2026-03-10T08:06:15.000Z").getTime(),
      occurred: "2026-03-10T08:06:15.000Z",
      stageSubflowRunId: "subflow-123",
      logFlowRunIds: ["subflow-123"],
    });
  });

  it("ignores events with an invalid prefix or payload", () => {
    expect(
      parsePrefectStageProgressEvent({
        event: "prefect.task-run.Completed",
        payload: {
          stage_id: "stage-2",
          status: "completed",
        },
      }),
    ).toBeNull();

    expect(
      parsePrefectStageProgressEvent({
        event: "nof1-causal-lab.pipeline-stage.completed",
        payload: {
          stage_id: "not-a-stage",
          status: "completed",
        },
      }),
    ).toBeNull();
  });
});
