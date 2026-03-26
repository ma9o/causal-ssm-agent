import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/storage", () => ({
  readData: vi.fn(),
}));

import { readData } from "@/lib/storage";
import { buildAnalysisManifest } from "./_shared";

const originalFetch = globalThis.fetch;

vi.mocked(readData).mockRejectedValue(new Error("missing"));

function jsonResponse(
  data: unknown,
  status = 200,
  headers?: Record<string, string>,
): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers(headers),
    json: async () => data,
  } as Response;
}

function parseBody(init?: RequestInit): Record<string, unknown> {
  return JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
}

function getEventRootFlowRunId(body: Record<string, unknown>): string | undefined {
  const resourceId =
    ((body.filter as { resource?: { id?: string[] } } | undefined)?.resource?.id ?? [])[0];
  return resourceId?.replace(/^prefect\.flow-run\./, "");
}

function stageEvent(
  stageId: string,
  status: "running" | "completed" | "failed",
  occurred: string,
  outcomeOrRuntime?:
    | string
    | {
        outcome?: string;
        stageSubflowRunId?: string;
        logFlowRunIds?: string[];
      },
) {
  const runtime =
    typeof outcomeOrRuntime === "string" ? { outcome: outcomeOrRuntime } : (outcomeOrRuntime ?? {});
  return {
    occurred,
    event: `causal-ssm.pipeline-stage.${status}`,
    payload: {
      stage_id: stageId,
      status,
      ...(runtime.outcome ? { outcome: runtime.outcome } : {}),
      ...(runtime.stageSubflowRunId
        ? { stage_subflow_run_id: runtime.stageSubflowRunId }
        : {}),
      ...(runtime.logFlowRunIds ? { log_flow_run_ids: runtime.logFlowRunIds } : {}),
    },
  };
}

function eventPage(events: ReturnType<typeof stageEvent>[]) {
  return {
    events,
    total: events.length,
    next_page: null,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  vi.mocked(readData).mockReset();
  vi.mocked(readData).mockRejectedValue(new Error("missing"));
  globalThis.fetch = originalFetch;
});

describe("buildAnalysisManifest", () => {
  it("assigns each stage to the latest run that actually executed it", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;
        const tags = (flowRuns?.tags as { all_?: string[] } | undefined)?.all_;

        if (tags?.[0] === "workspace:user-123") {
          return jsonResponse([{ id: "resume-run" }, { id: "full-run" }]);
        }
        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/full-run") {
        return jsonResponse({
          id: "full-run",
          created: "2026-03-13T18:15:00.000Z",
          parameters: { query: "Why does this happen?" },
        });
      }

      if (url === "http://localhost:4200/api/flow_runs/resume-run") {
        return jsonResponse({
          id: "resume-run",
          created: "2026-03-13T18:33:00.000Z",
          parameters: { start_stage: "stage-4b" },
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        const rootFlowRunId = getEventRootFlowRunId(parseBody(init));

        if (rootFlowRunId === "full-run") {
          return jsonResponse(
            eventPage([
              stageEvent("stage-3", "completed", "2026-03-13T18:18:00.000Z"),
              stageEvent("stage-4", "running", "2026-03-13T18:18:30.000Z", {
                stageSubflowRunId: "stage-4-subflow",
                logFlowRunIds: ["stage-4-subflow"],
              }),
              stageEvent("stage-4", "completed", "2026-03-13T18:20:00.000Z"),
            ]),
          );
        }

        if (rootFlowRunId === "resume-run") {
          return jsonResponse(
            eventPage([
              stageEvent("stage-4", "completed", "2026-03-13T18:33:00.000Z"),
              stageEvent("stage-4b", "running", "2026-03-13T18:33:05.000Z", {
                stageSubflowRunId: "stage-4b-subflow",
                logFlowRunIds: ["stage-4b-subflow"],
              }),
              stageEvent("stage-4b", "completed", "2026-03-13T18:35:00.000Z"),
              stageEvent("stage-5b", "running", "2026-03-13T18:35:10.000Z"),
              stageEvent("stage-5b", "completed", "2026-03-13T18:45:00.000Z", "warn"),
            ]),
          );
        }
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest).toMatchObject({
      workspaceId: "user-123",
      createdAt: "2026-03-13T18:15:00.000Z",
      question: "Why does this happen?",
      rootFlowRunIds: ["full-run", "resume-run"],
      latestRootFlowRunId: "resume-run",
    });

    expect(manifest?.stages["stage-3"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:18:00.000Z",
        endTime: "2026-03-13T18:18:00.000Z",
      },
    });
    expect(manifest?.stages["stage-4"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: "stage-4-subflow",
      logFlowRunIds: ["stage-4-subflow"],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:18:30.000Z",
        endTime: "2026-03-13T18:20:00.000Z",
      },
    });
    expect(manifest?.stages["stage-4b"]).toEqual({
      ownerRootFlowRunId: "resume-run",
      stageSubflowRunId: "stage-4b-subflow",
      logFlowRunIds: ["stage-4b-subflow"],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:33:05.000Z",
        endTime: "2026-03-13T18:35:00.000Z",
      },
    });
    expect(manifest?.stages["stage-5b"]).toEqual({
      ownerRootFlowRunId: "resume-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:35:10.000Z",
        endTime: "2026-03-13T18:45:00.000Z",
      },
    });
  });

  it("hydrates child flow runs directly from stage runtime events", async () => {
    const fetchMock = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;

        if ((flowRuns?.tags as { all_?: string[] } | undefined)?.all_?.[0] === "workspace:user-123") {
          return jsonResponse([{ id: "run-abc" }]);
        }
        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/run-abc") {
        return jsonResponse({
          id: "run-abc",
          created: "2026-03-13T18:33:00.000Z",
          parameters: {},
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        return jsonResponse(
          eventPage([
            stageEvent("stage-4b", "running", "2026-03-13T18:33:00.000Z", {
              stageSubflowRunId: "flow-123",
              logFlowRunIds: ["flow-123"],
            }),
            stageEvent("stage-4b", "completed", "2026-03-13T18:35:00.000Z"),
          ]),
        );
      }

      throw new Error(`Unexpected fetch: ${url}`);
    });

    globalThis.fetch = fetchMock as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-4b"]).toEqual({
      ownerRootFlowRunId: "run-abc",
      stageSubflowRunId: "flow-123",
      logFlowRunIds: ["flow-123"],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:33:00.000Z",
        endTime: "2026-03-13T18:35:00.000Z",
      },
    });
  });

  it("ignores newer root runs that never emitted a stage event", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;

        if ((flowRuns?.tags as { all_?: string[] } | undefined)?.all_?.[0] === "workspace:user-123") {
          return jsonResponse([{ id: "empty-run" }, { id: "full-run" }]);
        }

        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/full-run") {
        return jsonResponse({
          id: "full-run",
          created: "2026-03-13T18:15:00.000Z",
          parameters: { query: "Why does this happen?" },
        });
      }

      if (url === "http://localhost:4200/api/flow_runs/empty-run") {
        return jsonResponse({
          id: "empty-run",
          created: "2026-03-13T18:33:00.000Z",
          parameters: { query: "Why does this happen?" },
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        const rootFlowRunId = getEventRootFlowRunId(parseBody(init));
        if (rootFlowRunId === "full-run") {
          return jsonResponse(eventPage([stageEvent("stage-0", "completed", "2026-03-13T18:16:00.000Z")]));
        }
        return jsonResponse(eventPage([]));
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-0"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:16:00.000Z",
        endTime: "2026-03-13T18:16:00.000Z",
      },
    });
  });

  it("hydrates stage-2 log sources directly from stage runtime events", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;
        const tags = (flowRuns?.tags as { all_?: string[] } | undefined)?.all_;

        if (tags?.[0] === "workspace:user-123") {
          return jsonResponse([{ id: "run-abc" }]);
        }
        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/run-abc") {
        return jsonResponse({
          id: "run-abc",
          created: "2026-03-13T18:33:00.000Z",
          parameters: {},
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        return jsonResponse(
          eventPage([
            stageEvent("stage-2", "running", "2026-03-13T18:33:00.000Z", {
              stageSubflowRunId: "stage-2-subflow",
              logFlowRunIds: ["stage-2-subflow"],
            }),
          ]),
        );
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-2"]).toEqual({
      ownerRootFlowRunId: "run-abc",
      stageSubflowRunId: "stage-2-subflow",
      logFlowRunIds: ["stage-2-subflow"],
      execution: {
        stateType: "RUNNING",
        startTime: "2026-03-13T18:33:00.000Z",
        endTime: null,
      },
    });
  });

  it("keeps downstream stage ownership on the original run for single-stage reruns", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;
        const tags = (flowRuns?.tags as { all_?: string[] } | undefined)?.all_;

        if (tags?.[0] === "workspace:user-123") {
          return jsonResponse([{ id: "rerun-run" }, { id: "full-run" }]);
        }
        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/full-run") {
        return jsonResponse({
          id: "full-run",
          created: "2026-03-13T18:40:00.000Z",
          parameters: { query: "Why does this happen?" },
        });
      }

      if (url === "http://localhost:4200/api/flow_runs/rerun-run") {
        return jsonResponse({
          id: "rerun-run",
          created: "2026-03-13T18:55:00.000Z",
          parameters: {
            query: "Why does this happen?",
            start_stage: "stage-4",
            end_stage: "stage-4",
          },
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        const rootFlowRunId = getEventRootFlowRunId(parseBody(init));

        if (rootFlowRunId === "full-run") {
          return jsonResponse(
            eventPage([
              stageEvent("stage-4", "completed", "2026-03-13T18:45:00.000Z"),
              stageEvent("stage-4b", "completed", "2026-03-13T18:46:00.000Z"),
              stageEvent("stage-6", "completed", "2026-03-13T18:50:00.000Z"),
            ]),
          );
        }

        if (rootFlowRunId === "rerun-run") {
          return jsonResponse(
            eventPage([
              stageEvent("stage-4", "running", "2026-03-13T18:55:00.000Z", {
                stageSubflowRunId: "stage-4-subflow",
                logFlowRunIds: ["stage-4-subflow"],
              }),
              stageEvent("stage-4", "completed", "2026-03-13T19:00:00.000Z"),
            ]),
          );
        }
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-4"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: "stage-4-subflow",
      logFlowRunIds: ["stage-4-subflow"],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:55:00.000Z",
        endTime: "2026-03-13T19:00:00.000Z",
      },
    });
    expect(manifest?.stages["stage-4b"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:46:00.000Z",
        endTime: "2026-03-13T18:46:00.000Z",
      },
    });
    expect(manifest?.stages["stage-6"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:50:00.000Z",
        endTime: "2026-03-13T18:50:00.000Z",
      },
    });
  });

  it("treats an active rerun as authoritative for its downstream stage window", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;
        const tags = (flowRuns?.tags as { all_?: string[] } | undefined)?.all_;

        if (tags?.[0] === "workspace:user-123") {
          return jsonResponse([{ id: "rerun-run" }, { id: "full-run" }]);
        }

        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/full-run") {
        return jsonResponse({
          id: "full-run",
          created: "2026-03-13T18:15:00.000Z",
          parameters: { query: "Why does this happen?" },
        });
      }

      if (url === "http://localhost:4200/api/flow_runs/rerun-run") {
        return jsonResponse({
          id: "rerun-run",
          created: "2026-03-13T18:55:00.000Z",
          parameters: {
            query: "Why does this happen?",
            start_stage: "stage-1a",
          },
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        const rootFlowRunId = getEventRootFlowRunId(parseBody(init));

        if (rootFlowRunId === "full-run") {
          return jsonResponse(
            eventPage([
              stageEvent("stage-0", "completed", "2026-03-13T18:16:00.000Z"),
              stageEvent("stage-1a", "completed", "2026-03-13T18:17:00.000Z"),
              stageEvent("stage-1b", "completed", "2026-03-13T18:18:00.000Z"),
              stageEvent("stage-2", "completed", "2026-03-13T18:19:00.000Z"),
              stageEvent("stage-3", "completed", "2026-03-13T18:20:00.000Z"),
              stageEvent("stage-4", "failed", "2026-03-13T18:21:00.000Z"),
            ]),
          );
        }

        if (rootFlowRunId === "rerun-run") {
          return jsonResponse(
            eventPage([
              stageEvent("stage-0", "completed", "2026-03-13T18:55:01.000Z"),
              stageEvent("stage-1a", "running", "2026-03-13T18:55:02.000Z"),
              stageEvent("stage-1a", "completed", "2026-03-13T18:56:00.000Z"),
              stageEvent("stage-1b", "running", "2026-03-13T18:56:01.000Z"),
            ]),
          );
        }
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-0"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:16:00.000Z",
        endTime: "2026-03-13T18:16:00.000Z",
      },
    });
    expect(manifest?.stages["stage-1a"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:55:02.000Z",
        endTime: "2026-03-13T18:56:00.000Z",
      },
    });
    expect(manifest?.stages["stage-1b"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "RUNNING",
        startTime: "2026-03-13T18:56:01.000Z",
        endTime: null,
      },
    });
    expect(manifest?.stages["stage-2"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: null,
    });
    expect(manifest?.stages["stage-4"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: null,
    });
  });

  it("can bootstrap a manifest directly from an explicit root flow run when Prefect tags are unavailable", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;
        const tags = (flowRuns?.tags as { all_?: string[] } | undefined)?.all_;
        if (tags?.[0] === "workspace:user-123") {
          return jsonResponse([]);
        }
        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/live-run") {
        return jsonResponse({
          id: "live-run",
          created: "2026-03-14T10:00:00.000Z",
          parameters: { query: "Why did this launch?" },
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        return jsonResponse(eventPage([stageEvent("stage-0", "running", "2026-03-14T10:00:05.000Z")]));
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123", ["live-run"]);

    expect(manifest).toMatchObject({
      workspaceId: "user-123",
      createdAt: "2026-03-14T10:00:00.000Z",
      question: "Why did this launch?",
      rootFlowRunIds: ["live-run"],
      latestRootFlowRunId: "live-run",
    });
    expect(manifest?.stages["stage-0"]).toEqual({
      ownerRootFlowRunId: "live-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      execution: {
        stateType: "RUNNING",
        startTime: "2026-03-14T10:00:05.000Z",
        endTime: null,
      },
    });
  });

  it("falls back to query.txt when Prefect flow runs omit the question", async () => {
    vi.mocked(readData).mockResolvedValue("Stored workspace question\n");

    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;
        const tags = (flowRuns?.tags as { all_?: string[] } | undefined)?.all_;
        if (tags?.[0] === "workspace:user-123") {
          return jsonResponse([{ id: "resume-run" }]);
        }
        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/resume-run") {
        return jsonResponse({
          id: "resume-run",
          created: "2026-03-13T18:33:00.000Z",
          parameters: { start_stage: "stage-4b" },
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        return jsonResponse(eventPage([]));
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.question).toBe("Stored workspace question");
  });

  it("retries retryable Prefect responses while loading the manifest", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ error: "rate limited" }, 429, { "Retry-After": "0" }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "run-abc" }]))
      .mockResolvedValueOnce(
        jsonResponse({
          id: "run-abc",
          created: "2026-03-13T18:33:00.000Z",
          parameters: { query: "Why does this happen?" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse(eventPage([])));

    globalThis.fetch = fetchMock as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest).toMatchObject({
      workspaceId: "user-123",
      question: "Why does this happen?",
      rootFlowRunIds: ["run-abc"],
      latestRootFlowRunId: "run-abc",
    });
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:4200/api/flow_runs/filter",
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:4200/api/flow_runs/filter",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
