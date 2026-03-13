import { afterEach, describe, expect, it, vi } from "vitest";
import { buildPrefectEventFilterMessage, parsePrefectStageProgressEvent } from "./use-run-events";
import { fetchStageFlowRunId } from "./use-stage-logs";

const originalFetch = globalThis.fetch;

afterEach(() => {
  vi.restoreAllMocks();
  globalThis.fetch = originalFetch;
});

function jsonResponse(data: unknown) {
  return {
    ok: true,
    json: async () => data,
  } as Response;
}

describe("buildPrefectEventFilterMessage", () => {
  it("subscribes to the run's custom stage events across a future time window", () => {
    const now = new Date("2026-03-10T08:06:15.000Z");

    expect(buildPrefectEventFilterMessage("run-123", now)).toEqual({
      type: "filter",
      filter: {
        event: { prefix: ["causal-ssm."] },
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
        event: "causal-ssm.pipeline-stage.completed",
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
        event: "causal-ssm.pipeline-stage.completed",
        payload: {
          stage_id: "not-a-stage",
          status: "completed",
        },
      }),
    ).toBeNull();
  });
});

describe("fetchStageFlowRunId", () => {
  it("resolves a stage flow run from its parent task run", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse([
          { id: "task-1", name: "stage-2-flow" },
          { id: "task-2", name: "stage-4b-flow" },
        ]),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "flow-123" }]));

    globalThis.fetch = fetchMock as typeof fetch;

    await expect(fetchStageFlowRunId("run-abc", "stage-4b")).resolves.toBe("flow-123");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/prefect/task_runs/filter",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          flow_runs: { id: { any_: ["run-abc"] } },
          sort: "EXPECTED_START_TIME_DESC",
        }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/prefect/flow_runs/filter",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          flows: { name: { any_: ["stage-4b-flow"] } },
          flow_runs: { parent_task_run_id: { any_: ["task-2"] } },
          sort: "START_TIME_DESC",
          limit: 1,
        }),
      }),
    );
  });

  it("returns null when the stage wrapper task run is absent", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse([{ id: "task-1", name: "stage-2-flow" }]),
      ) as typeof fetch;

    await expect(fetchStageFlowRunId("run-abc", "stage-5b")).resolves.toBeNull();
  });
});
