import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/storage", () => ({
  isStorageNotFoundError: vi.fn((error: unknown) =>
    error instanceof Error && error.message.startsWith("missing"),
  ),
  prefixExists: vi.fn(),
  readData: vi.fn(),
}));

import { prefixExists, readData } from "@/lib/storage";
import {
  buildAnalysisManifest,
  buildStage2ReplayState,
  buildStage4ReplayState,
  resolveStageLogScopeFlowRunIds,
} from "./_shared";

const originalFetch = globalThis.fetch;

vi.mocked(readData).mockRejectedValue(new Error("missing"));

function jsonResponse(data: unknown, status = 200, headers?: Record<string, string>): Response {
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
  const resourceId = ((body.filter as { resource?: { id?: string[] } } | undefined)?.resource?.id ??
    [])[0];
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
    event: `nof1-causal-lab.pipeline-stage.${status}`,
    payload: {
      stage_id: stageId,
      status,
      ...(runtime.outcome ? { outcome: runtime.outcome } : {}),
      ...(runtime.stageSubflowRunId ? { stage_subflow_run_id: runtime.stageSubflowRunId } : {}),
      ...(runtime.logFlowRunIds ? { log_flow_run_ids: runtime.logFlowRunIds } : {}),
    },
  };
}

function stage4GraphEvent(occurred: string) {
  return {
    occurred,
    event: "nof1-causal-lab.stage4.graph",
    payload: {
      type: "graph",
      nodes: [
        {
          id: "indicator:sleep_quality",
          kind: "indicator_decision",
          label: "Sleep Quality",
          phase: "model_decisions",
        },
      ],
      edges: [],
      phases: [{ id: "model_decisions", label: "Model Decisions" }],
    },
  };
}

function stage4SnapshotEvent(occurred: string, phase: string) {
  return {
    occurred,
    event: "nof1-causal-lab.stage4.snapshot",
    payload: {
      type: "snapshot",
      cursor: { kind: "block", block_id: "indicator:sleep_quality" },
      block_status: { "indicator:sleep_quality": "accepted" },
      model_spec_locked: phase !== "model_decisions",
      repair_campaign: null,
      phase,
    },
  };
}

function stage4TransitionEvent(occurred: string) {
  return {
    occurred,
    event: "nof1-causal-lab.stage4.block_transition",
    payload: {
      type: "block_transition",
      block_id: "indicator:sleep_quality",
      status: "accepted",
      detail_kind: "indicator_choice",
      variable: "sleep_quality",
      distribution: "gaussian",
      link: "identity",
      reasoning: "Continuous rating scale.",
    },
  };
}

function stage2PlanEvent(occurred: string) {
  return {
    occurred,
    event: "nof1-causal-lab.stage2.plan",
    payload: {
      type: "plan",
      total_workers: 3,
      max_concurrent_workers: 30,
      max_rpm: 450,
    },
  };
}

function stage2WorkerEvent(
  occurred: string,
  workerId: number,
  state: "running" | "completed" | "failed",
  overrides: Record<string, unknown> = {},
) {
  return {
    occurred,
    event: "nof1-causal-lab.stage2.worker",
    payload: {
      type: "worker",
      worker_id: workerId,
      state,
      n_windows: 1,
      ...overrides,
    },
  };
}

function stage2SnapshotEvent(
  occurred: string,
  overrides: Record<string, unknown> = {},
) {
  return {
    occurred,
    event: "nof1-causal-lab.stage2.snapshot",
    payload: {
      type: "snapshot",
      total_workers: 3,
      pending_workers: 1,
      running_workers: 0,
      completed_workers: 1,
      failed_workers: 1,
      llm_requests_last_60s: 17,
      ...overrides,
    },
  };
}

function eventPage(
  events: { occurred: string; event: string; payload: Record<string, unknown> }[],
) {
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
  vi.mocked(prefixExists).mockReset();
  vi.mocked(prefixExists).mockResolvedValue(false);
  globalThis.fetch = originalFetch;
});

describe("buildAnalysisManifest", () => {
  it("builds a shared workspace manifest from persisted artifacts without contacting Prefect", async () => {
    vi.mocked(readData).mockImplementation(async (path: string) => {
      if (path === "DEMO/query.txt") {
        return "Did escitalopram help?";
      }
      if (path === "DEMO/session.json") {
        return JSON.stringify({
          createdAt: "2026-05-07T00:00:00.000Z",
          rootFlowRunIds: [],
        });
      }
      throw new Error("missing");
    });
    vi.mocked(prefixExists).mockImplementation(
      async (path: string) =>
        path === "DEMO/run/stage-1a.json" || path === "DEMO/run/stage-1b.json",
    );
    const fetchMock = vi.fn(async (input) => {
      throw new Error(`Unexpected fetch: ${String(input)}`);
    });
    globalThis.fetch = fetchMock as typeof fetch;

    const manifest = await buildAnalysisManifest("DEMO");

    expect(manifest).toMatchObject({
      workspaceId: "DEMO",
      createdAt: "2026-05-07T00:00:00.000Z",
      question: "Did escitalopram help?",
      rootFlowRunIds: [],
      latestRootFlowRunId: null,
    });
    expect(manifest?.stages["stage-1a"].execution).toEqual({
      stateType: "COMPLETED",
      startTime: "2026-05-07T00:00:00.000Z",
      endTime: "2026-05-07T00:00:00.000Z",
    });
    expect(manifest?.stages["stage-1b"].execution).toEqual({
      stateType: "COMPLETED",
      startTime: "2026-05-07T00:00:00.000Z",
      endTime: "2026-05-07T00:00:00.000Z",
    });
    expect(manifest?.stages["stage-0"].execution).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("supplements Prefect lineage with persisted artifacts for stages without events", async () => {
    vi.mocked(readData).mockImplementation(async (path: string) => {
      if (path === "user-123/query.txt") {
        return "Did escitalopram help?";
      }
      throw new Error("missing");
    });
    vi.mocked(prefixExists).mockImplementation(
      async (path: string) =>
        path === "user-123/run/stage-1a.json" || path === "user-123/run/stage-1b.json",
    );
    globalThis.fetch = vi.fn(async (input) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        return jsonResponse([{ id: "run-abc" }]);
      }

      if (url === "http://localhost:4200/api/flow_runs/run-abc") {
        return jsonResponse({
          id: "run-abc",
          created: "2026-05-07T12:58:19.530Z",
          parameters: {},
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        return jsonResponse(
          eventPage([stageEvent("stage-0", "completed", "2026-05-07T13:00:07.619Z")]),
        );
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.rootFlowRunIds).toEqual(["run-abc"]);
    expect(manifest?.stages["stage-0"]).toMatchObject({
      ownerRootFlowRunId: "run-abc",
      execution: { stateType: "COMPLETED" },
    });
    expect(manifest?.stages["stage-1a"]).toEqual({
      ownerRootFlowRunId: null,
      stageSubflowRunId: null,
      initialLogFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-05-07T12:58:19.530Z",
        endTime: "2026-05-07T12:58:19.530Z",
      },
    });
    expect(manifest?.stages["stage-1b"].execution?.stateType).toBe("COMPLETED");
  });

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
          parameters: { start_stage: "stage-5b" },
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
              stageEvent("stage-5b", "running", "2026-03-13T18:35:10.000Z", {
                stageSubflowRunId: "stage-5b-subflow",
                logFlowRunIds: ["stage-5b-subflow"],
              }),
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
      initialLogFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:18:00.000Z",
        endTime: "2026-03-13T18:18:00.000Z",
      },
    });
    expect(manifest?.stages["stage-4"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: "stage-4-subflow",
      initialLogFlowRunIds: ["stage-4-subflow"],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:18:30.000Z",
        endTime: "2026-03-13T18:20:00.000Z",
      },
    });
    expect(manifest?.stages["stage-5b"]).toEqual({
      ownerRootFlowRunId: "resume-run",
      stageSubflowRunId: "stage-5b-subflow",
      initialLogFlowRunIds: ["stage-5b-subflow"],
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

        if (
          (flowRuns?.tags as { all_?: string[] } | undefined)?.all_?.[0] === "workspace:user-123"
        ) {
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
            stageEvent("stage-5b", "running", "2026-03-13T18:33:00.000Z", {
              stageSubflowRunId: "flow-123",
              logFlowRunIds: ["flow-123"],
            }),
            stageEvent("stage-5b", "completed", "2026-03-13T18:35:00.000Z"),
          ]),
        );
      }

      throw new Error(`Unexpected fetch: ${url}`);
    });

    globalThis.fetch = fetchMock as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-5b"]).toEqual({
      ownerRootFlowRunId: "run-abc",
      stageSubflowRunId: "flow-123",
      initialLogFlowRunIds: ["flow-123"],
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

        if (
          (flowRuns?.tags as { all_?: string[] } | undefined)?.all_?.[0] === "workspace:user-123"
        ) {
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
          return jsonResponse(
            eventPage([stageEvent("stage-0", "completed", "2026-03-13T18:16:00.000Z")]),
          );
        }
        return jsonResponse(eventPage([]));
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-0"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      initialLogFlowRunIds: [],
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
      initialLogFlowRunIds: ["stage-2-subflow"],
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
              stageEvent("stage-5b", "completed", "2026-03-13T18:46:00.000Z"),
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
      initialLogFlowRunIds: ["stage-4-subflow"],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:55:00.000Z",
        endTime: "2026-03-13T19:00:00.000Z",
      },
    });
    expect(manifest?.stages["stage-5b"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      initialLogFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:46:00.000Z",
        endTime: "2026-03-13T18:46:00.000Z",
      },
    });
    expect(manifest?.stages["stage-6"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      initialLogFlowRunIds: [],
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
      initialLogFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:16:00.000Z",
        endTime: "2026-03-13T18:16:00.000Z",
      },
    });
    expect(manifest?.stages["stage-1a"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: null,
      initialLogFlowRunIds: [],
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-03-13T18:55:02.000Z",
        endTime: "2026-03-13T18:56:00.000Z",
      },
    });
    expect(manifest?.stages["stage-1b"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: null,
      initialLogFlowRunIds: [],
      execution: {
        stateType: "RUNNING",
        startTime: "2026-03-13T18:56:01.000Z",
        endTime: null,
      },
    });
    expect(manifest?.stages["stage-2"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: null,
      initialLogFlowRunIds: [],
      execution: null,
    });
    expect(manifest?.stages["stage-4"]).toEqual({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: null,
      initialLogFlowRunIds: [],
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
        return jsonResponse(
          eventPage([stageEvent("stage-0", "running", "2026-03-14T10:00:05.000Z")]),
        );
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
      initialLogFlowRunIds: [],
      execution: {
        stateType: "RUNNING",
        startTime: "2026-03-14T10:00:05.000Z",
        endTime: null,
      },
    });
  });

  it("orders lineage by actual start time before original creation time", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = parseBody(init);
        const flowRuns = body.flow_runs as Record<string, unknown> | undefined;
        const tags = (flowRuns?.tags as { all_?: string[] } | undefined)?.all_;
        if (tags?.[0] === "workspace:user-123") {
          return jsonResponse([{ id: "older-scheduled-run" }, { id: "newer-created-run" }]);
        }
        return jsonResponse([]);
      }

      if (url === "http://localhost:4200/api/flow_runs/older-scheduled-run") {
        return jsonResponse({
          id: "older-scheduled-run",
          created: "2026-03-13T18:00:00.000Z",
          start_time: "2026-03-13T19:00:00.000Z",
          parameters: { query: "Why did this start late?" },
        });
      }

      if (url === "http://localhost:4200/api/flow_runs/newer-created-run") {
        return jsonResponse({
          id: "newer-created-run",
          created: "2026-03-13T18:30:00.000Z",
          start_time: "2026-03-13T18:31:00.000Z",
          parameters: {},
        });
      }

      if (url === "http://localhost:4200/api/events/filter") {
        const rootFlowRunId = getEventRootFlowRunId(parseBody(init));

        if (rootFlowRunId === "older-scheduled-run") {
          return jsonResponse(
            eventPage([stageEvent("stage-2", "running", "2026-03-13T19:00:05.000Z")]),
          );
        }

        if (rootFlowRunId === "newer-created-run") {
          return jsonResponse(
            eventPage([stageEvent("stage-1b", "completed", "2026-03-13T18:31:10.000Z")]),
          );
        }
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest).toMatchObject({
      rootFlowRunIds: ["newer-created-run", "older-scheduled-run"],
      latestRootFlowRunId: "older-scheduled-run",
      createdAt: "2026-03-13T18:31:00.000Z",
      question: "Why did this start late?",
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
          parameters: { start_stage: "stage-5b" },
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
      .mockResolvedValueOnce(jsonResponse({ error: "rate limited" }, 429, { "Retry-After": "0" }))
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

describe("buildStage4ReplayState", () => {
  it("reduces historical Stage 4 custom events into the latest replay state", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/events/filter") {
        const body = parseBody(init);
        const prefix = ((body.filter as { event?: { prefix?: string[] } } | undefined)?.event
          ?.prefix ?? [])[0];
        const rootFlowRunId = getEventRootFlowRunId(body);
        expect(prefix).toBe("nof1-causal-lab.stage4.");
        expect(rootFlowRunId).toBe("run-abc");
        return jsonResponse(
          eventPage([
            stage4GraphEvent("2026-03-31T11:00:00.000Z"),
            stage4TransitionEvent("2026-03-31T11:00:00.500Z"),
            stage4SnapshotEvent("2026-03-31T11:00:01.000Z", "model_decisions"),
            stage4SnapshotEvent("2026-03-31T11:00:02.000Z", "global_review"),
          ]),
        );
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    await expect(buildStage4ReplayState("run-abc")).resolves.toEqual({
      graph: {
        nodes: stage4GraphEvent("2026-03-31T11:00:00.000Z").payload.nodes,
        edges: stage4GraphEvent("2026-03-31T11:00:00.000Z").payload.edges,
        phases: stage4GraphEvent("2026-03-31T11:00:00.000Z").payload.phases,
      },
      snapshot: {
        cursor: stage4SnapshotEvent("2026-03-31T11:00:02.000Z", "global_review").payload.cursor,
        block_status: stage4SnapshotEvent("2026-03-31T11:00:02.000Z", "global_review").payload
          .block_status,
        model_spec_locked: stage4SnapshotEvent("2026-03-31T11:00:02.000Z", "global_review").payload
          .model_spec_locked,
        repair_campaign: stage4SnapshotEvent("2026-03-31T11:00:02.000Z", "global_review").payload
          .repair_campaign,
        phase: stage4SnapshotEvent("2026-03-31T11:00:02.000Z", "global_review").payload.phase,
      },
      lastBlockStateById: {
        "indicator:sleep_quality": {
          block_id: "indicator:sleep_quality",
          status: "accepted",
          detail_kind: "indicator_choice",
          variable: "sleep_quality",
          distribution: "gaussian",
          link: "identity",
          reasoning: "Continuous rating scale.",
        },
      },
    });
  });

});

describe("buildStage2ReplayState", () => {
  it("reduces historical Stage 2 custom events into the latest replay state", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/events/filter") {
        const body = parseBody(init);
        const prefix = ((body.filter as { event?: { prefix?: string[] } } | undefined)?.event
          ?.prefix ?? [])[0];
        const rootFlowRunId = getEventRootFlowRunId(body);
        expect(prefix).toBe("nof1-causal-lab.stage2.");
        expect(rootFlowRunId).toBe("run-stage2");
        return jsonResponse(
          eventPage([
            stage2PlanEvent("2026-04-02T10:00:00.000Z"),
            stage2WorkerEvent("2026-04-02T10:00:01.000Z", 0, "running"),
            stage2WorkerEvent("2026-04-02T10:00:02.000Z", 0, "completed", {
              n_extractions: 6,
              n_llm_calls: 1,
            }),
            stage2WorkerEvent("2026-04-02T10:00:03.000Z", 2, "failed", {
              error: "Error code: 402",
            }),
            stage2SnapshotEvent("2026-04-02T10:00:04.000Z"),
          ]),
        );
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    await expect(buildStage2ReplayState("run-stage2")).resolves.toEqual({
      plan: {
        total_workers: 3,
        max_concurrent_workers: 30,
        max_rpm: 450,
      },
      snapshot: {
        total_workers: 3,
        pending_workers: 1,
        running_workers: 0,
        completed_workers: 1,
        failed_workers: 1,
        llm_requests_last_60s: 17,
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
          state: "failed",
          n_windows: 1,
          n_extractions: null,
          n_llm_calls: null,
          error: "Error code: 402",
          completed_at: "2026-04-02T10:00:03.000Z",
        },
      },
    });
  });
});

describe("resolveStageLogScopeFlowRunIds", () => {
  it("expands stage-2 log scope to include child worker flows", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);
      if (url !== "http://localhost:4200/api/flow_runs/filter") {
        throw new Error(`Unexpected fetch: ${url}`);
      }

      expect(parseBody(init)).toEqual({
        flow_runs: { parent_flow_run_id: { any_: ["stage-2-subflow"] } },
        sort: "START_TIME_ASC",
        limit: 50,
      });

      return jsonResponse([{ id: "worker-flow-1" }, { id: "worker-flow-2" }]);
    }) as typeof fetch;

    await expect(resolveStageLogScopeFlowRunIds("stage-2", "stage-2-subflow")).resolves.toEqual([
      "stage-2-subflow",
      "worker-flow-1",
      "worker-flow-2",
    ]);
  });
});
