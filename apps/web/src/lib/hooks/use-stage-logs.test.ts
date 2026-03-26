import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildPrefectLogFilterBody,
  buildPrefectLogStreamFilterMessage,
  fetchIncrementalPrefectLogs,
  mergePrefectLogs,
  type PrefectLogEntry,
} from "@/lib/prefect-log-client";

const originalFetch = globalThis.fetch;

function logEntry(id: string, timestamp: string): PrefectLogEntry {
  return {
    id,
    created: timestamp,
    name: "prefect.flow_runs",
    level: 20,
    message: `log-${id}`,
    timestamp,
    flow_run_id: "run-1",
    task_run_id: null,
  };
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("mergePrefectLogs", () => {
  it("appends only unseen log ids", () => {
    const existing = [
      logEntry("log-1", "2026-03-22T10:00:00.000Z"),
      logEntry("log-2", "2026-03-22T10:00:01.000Z"),
    ];
    const incoming = [
      logEntry("log-2", "2026-03-22T10:00:01.000Z"),
      logEntry("log-3", "2026-03-22T10:00:02.000Z"),
    ];

    expect(mergePrefectLogs(existing, incoming).map((entry) => entry.id)).toEqual([
      "log-1",
      "log-2",
      "log-3",
    ]);
  });
});

describe("buildPrefectLogFilterBody", () => {
  it("uses offset-based pagination in the Prefect logs filter body", () => {
    expect(buildPrefectLogFilterBody(["run-1", "run-2"], 25, 500)).toEqual({
      offset: 25,
      logs: {
        flow_run_id: {
          any_: ["run-1", "run-2"],
        },
      },
      sort: "TIMESTAMP_ASC",
      limit: 500,
    });
  });
});

describe("buildPrefectLogStreamFilterMessage", () => {
  it("subscribes to live logs with the same time window shape as Prefect's client", () => {
    const now = new Date("2026-03-22T10:00:00.000Z");

    expect(buildPrefectLogStreamFilterMessage(["run-1", "run-2"], now)).toEqual({
      type: "filter",
      filter: {
        flow_run_id: {
          any_: ["run-1", "run-2"],
        },
        timestamp: {
          after_: "2026-03-22T09:59:00.000Z",
          before_: "2027-03-22T10:00:00.000Z",
        },
      },
    });
  });
});

describe("fetchIncrementalPrefectLogs", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("starts paging from the existing log count and keeps fetching until exhaustion", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse([
          logEntry("log-3", "2026-03-22T10:00:02.000Z"),
          logEntry("log-4", "2026-03-22T10:00:03.000Z"),
        ]),
      )
      .mockResolvedValueOnce(jsonResponse([logEntry("log-5", "2026-03-22T10:00:04.000Z")]));

    globalThis.fetch = fetchMock as typeof fetch;

    const existing = [
      logEntry("log-1", "2026-03-22T10:00:00.000Z"),
      logEntry("log-2", "2026-03-22T10:00:01.000Z"),
    ];

    const logs = await fetchIncrementalPrefectLogs(["run-1"], existing, { limit: 2 });

    expect(logs.map((entry) => entry.id)).toEqual([
      "log-1",
      "log-2",
      "log-3",
      "log-4",
      "log-5",
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const firstBody = JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string);
    const secondBody = JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string);

    expect(firstBody).toMatchObject({
      offset: 2,
      sort: "TIMESTAMP_ASC",
      limit: 2,
      logs: { flow_run_id: { any_: ["run-1"] } },
    });
    expect(secondBody).toMatchObject({
      offset: 4,
      sort: "TIMESTAMP_ASC",
      limit: 2,
      logs: { flow_run_id: { any_: ["run-1"] } },
    });
  });

  it("can restart paging from offset zero and merge against cached logs when the scope widens", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse([
          logEntry("log-1", "2026-03-22T10:00:00.000Z"),
          logEntry("log-2", "2026-03-22T10:00:01.000Z"),
        ]),
      )
      .mockResolvedValueOnce(jsonResponse([logEntry("log-3", "2026-03-22T10:00:02.000Z")]));

    globalThis.fetch = fetchMock as typeof fetch;

    const existing = [
      logEntry("log-1", "2026-03-22T10:00:00.000Z"),
      logEntry("log-2", "2026-03-22T10:00:01.000Z"),
    ];

    const logs = await fetchIncrementalPrefectLogs(["run-1", "run-2"], existing, {
      limit: 2,
      offset: 0,
    });

    expect(logs.map((entry) => entry.id)).toEqual([
      "log-1",
      "log-2",
      "log-3",
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const firstBody = JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string);
    const secondBody = JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string);

    expect(firstBody).toMatchObject({
      offset: 0,
      sort: "TIMESTAMP_ASC",
      limit: 2,
      logs: { flow_run_id: { any_: ["run-1", "run-2"] } },
    });
    expect(secondBody).toMatchObject({
      offset: 2,
      sort: "TIMESTAMP_ASC",
      limit: 2,
      logs: { flow_run_id: { any_: ["run-1", "run-2"] } },
    });
  });
});
