import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../sessions/_shared", () => ({
  getLatestSessionRootFlowRunId: vi.fn((session) => session?.rootFlowRunIds?.at(-1) ?? null),
  readQuestion: vi.fn(),
  readSession: vi.fn(),
}));

import { readQuestion, readSession } from "../sessions/_shared";
import { buildAnalysisManifest } from "./_shared";

const originalFetch = globalThis.fetch;

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
  } as Response;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  globalThis.fetch = originalFetch;
});

describe("buildAnalysisManifest", () => {
  it("assigns each stage to the root run that owns it after a resume", async () => {
    vi.mocked(readSession).mockResolvedValue({
      createdAt: "2026-03-13T18:33:26.268Z",
      rootFlowRunIds: ["full-run", "resume-run"],
    });
    vi.mocked(readQuestion).mockResolvedValue("Why does this happen?");

    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/full-run") {
        return jsonResponse({ id: "full-run", parameters: {} });
      }

      if (url === "http://localhost:4200/api/flow_runs/resume-run") {
        return jsonResponse({ id: "resume-run", parameters: { start_stage: "stage-4b" } });
      }

      if (url === "http://localhost:4200/api/task_runs/filter") {
        const body = JSON.parse(String(init?.body ?? "{}")) as {
          flow_runs?: { id?: { any_?: string[] } };
        };
        const rootFlowRunId = body.flow_runs?.id?.any_?.[0];

        if (rootFlowRunId === "full-run") {
          return jsonResponse([
            {
              id: "stage-4-task",
              name: "stage-4-flow-0",
              state_type: "COMPLETED",
              start_time: "2026-03-13T18:15:00.000Z",
              end_time: "2026-03-13T18:20:00.000Z",
            },
          ]);
        }

        if (rootFlowRunId === "resume-run") {
          return jsonResponse([
            {
              id: "stage-4b-task",
              name: "stage-4b-flow-0",
              state_type: "COMPLETED",
              start_time: "2026-03-13T18:33:00.000Z",
              end_time: "2026-03-13T18:35:00.000Z",
            },
          ]);
        }
      }

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = JSON.parse(String(init?.body ?? "{}")) as {
          flow_runs?: { parent_task_run_id?: { any_?: string[] } };
        };
        const parentTaskRunId = body.flow_runs?.parent_task_run_id?.any_?.[0];

        if (parentTaskRunId === "stage-4-task") {
          return jsonResponse([{ id: "stage-4-subflow" }]);
        }

        if (parentTaskRunId === "stage-4b-task") {
          return jsonResponse([{ id: "stage-4b-subflow" }]);
        }
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest).toMatchObject({
      workspaceId: "user-123",
      createdAt: "2026-03-13T18:33:26.268Z",
      question: "Why does this happen?",
      rootFlowRunIds: ["full-run", "resume-run"],
      latestRootFlowRunId: "resume-run",
    });

    expect(manifest?.stages["stage-3"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      wrapperTaskRun: null,
    });
    expect(manifest?.stages["stage-4"]).toMatchObject({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: "stage-4-subflow",
      logFlowRunIds: ["stage-4-subflow"],
      wrapperTaskRun: {
        id: "stage-4-task",
        name: "stage-4-flow-0",
        stateType: "COMPLETED",
      },
    });
    expect(manifest?.stages["stage-4b"]).toMatchObject({
      ownerRootFlowRunId: "resume-run",
      stageSubflowRunId: "stage-4b-subflow",
      logFlowRunIds: ["stage-4b-subflow"],
      wrapperTaskRun: {
        id: "stage-4b-task",
        name: "stage-4b-flow-0",
        stateType: "COMPLETED",
      },
    });
    expect(manifest?.stages["stage-5b"]).toEqual({
      ownerRootFlowRunId: "resume-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      wrapperTaskRun: null,
    });
  });

  it("matches Prefect's suffixed wrapper task names when resolving a stage subflow", async () => {
    vi.mocked(readSession).mockResolvedValue({
      createdAt: "2026-03-13T18:33:26.268Z",
      rootFlowRunIds: ["run-abc"],
    });
    vi.mocked(readQuestion).mockResolvedValue(undefined);

    const fetchMock = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/run-abc") {
        return jsonResponse({ id: "run-abc", parameters: {} });
      }

      if (url === "http://localhost:4200/api/task_runs/filter") {
        return jsonResponse([
          {
            id: "task-1",
            name: "stage-4b-flow-0",
            state_type: "COMPLETED",
            start_time: "2026-03-13T18:33:00.000Z",
            end_time: "2026-03-13T18:35:00.000Z",
          },
        ]);
      }

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body ?? "{}"))).toEqual({
          flows: { name: { any_: ["stage-4b-flow"] } },
          flow_runs: { parent_task_run_id: { any_: ["task-1"] } },
          sort: "START_TIME_DESC",
          limit: 1,
        });
        return jsonResponse([{ id: "flow-123" }]);
      }

      throw new Error(`Unexpected fetch: ${url}`);
    });

    globalThis.fetch = fetchMock as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-4b"]).toMatchObject({
      ownerRootFlowRunId: "run-abc",
      stageSubflowRunId: "flow-123",
      logFlowRunIds: ["flow-123"],
      wrapperTaskRun: {
        id: "task-1",
        name: "stage-4b-flow-0",
      },
    });
  });

  it("resolves stage-2 log flow sources server-side, including nested worker flows", async () => {
    vi.mocked(readSession).mockResolvedValue({
      createdAt: "2026-03-13T18:33:26.268Z",
      rootFlowRunIds: ["run-abc"],
    });
    vi.mocked(readQuestion).mockResolvedValue(undefined);

    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/run-abc") {
        return jsonResponse({ id: "run-abc", parameters: {} });
      }

      if (url === "http://localhost:4200/api/task_runs/filter") {
        return jsonResponse([
          {
            id: "stage-2-task",
            name: "stage-2-flow-0",
            state_type: "RUNNING",
            start_time: "2026-03-13T18:33:00.000Z",
            end_time: null,
          },
        ]);
      }

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = JSON.parse(String(init?.body ?? "{}")) as {
          flow_runs?: {
            parent_task_run_id?: { any_?: string[] };
            parent_flow_run_id?: { any_?: string[] };
          };
        };

        if (body.flow_runs?.parent_task_run_id?.any_?.[0] === "stage-2-task") {
          return jsonResponse([{ id: "stage-2-subflow" }]);
        }

        if (body.flow_runs?.parent_flow_run_id?.any_?.[0] === "stage-2-subflow") {
          return jsonResponse([{ id: "worker-flow-1" }, { id: "worker-flow-2" }]);
        }
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-2"]).toMatchObject({
      ownerRootFlowRunId: "run-abc",
      stageSubflowRunId: "stage-2-subflow",
      logFlowRunIds: ["stage-2-subflow", "worker-flow-1", "worker-flow-2"],
      wrapperTaskRun: {
        id: "stage-2-task",
        name: "stage-2-flow-0",
      },
    });
  });

  it("keeps downstream stage ownership on the original run for single-stage reruns", async () => {
    vi.mocked(readSession).mockResolvedValue({
      createdAt: "2026-03-13T18:33:26.268Z",
      rootFlowRunIds: ["full-run", "rerun-run"],
    });
    vi.mocked(readQuestion).mockResolvedValue("Why does this happen?");

    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/full-run") {
        return jsonResponse({ id: "full-run", parameters: {} });
      }

      if (url === "http://localhost:4200/api/flow_runs/rerun-run") {
        return jsonResponse({
          id: "rerun-run",
          parameters: { start_stage: "stage-4", end_stage: "stage-4" },
        });
      }

      if (url === "http://localhost:4200/api/task_runs/filter") {
        const body = JSON.parse(String(init?.body ?? "{}")) as {
          flow_runs?: { id?: { any_?: string[] } };
        };
        const rootFlowRunId = body.flow_runs?.id?.any_?.[0];

        if (rootFlowRunId === "full-run") {
          return jsonResponse([
            {
              id: "stage-6-task",
              name: "stage-6-flow-0",
              state_type: "COMPLETED",
              start_time: "2026-03-13T18:40:00.000Z",
              end_time: "2026-03-13T18:50:00.000Z",
            },
          ]);
        }

        if (rootFlowRunId === "rerun-run") {
          return jsonResponse([
            {
              id: "stage-4-task",
              name: "stage-4-flow-0",
              state_type: "COMPLETED",
              start_time: "2026-03-13T18:55:00.000Z",
              end_time: "2026-03-13T19:00:00.000Z",
            },
          ]);
        }
      }

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        const body = JSON.parse(String(init?.body ?? "{}")) as {
          flow_runs?: { parent_task_run_id?: { any_?: string[] } };
        };
        const parentTaskRunId = body.flow_runs?.parent_task_run_id?.any_?.[0];

        if (parentTaskRunId === "stage-4-task") {
          return jsonResponse([{ id: "stage-4-subflow" }]);
        }

        if (parentTaskRunId === "stage-6-task") {
          return jsonResponse([{ id: "stage-6-subflow" }]);
        }
      }

      throw new Error(`Unexpected fetch: ${url}`);
    }) as typeof fetch;

    const manifest = await buildAnalysisManifest("user-123");

    expect(manifest?.stages["stage-4"]).toMatchObject({
      ownerRootFlowRunId: "rerun-run",
      stageSubflowRunId: "stage-4-subflow",
      logFlowRunIds: ["stage-4-subflow"],
    });
    expect(manifest?.stages["stage-4b"]).toEqual({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: null,
      logFlowRunIds: [],
      wrapperTaskRun: null,
    });
    expect(manifest?.stages["stage-6"]).toMatchObject({
      ownerRootFlowRunId: "full-run",
      stageSubflowRunId: "stage-6-subflow",
      logFlowRunIds: ["stage-6-subflow"],
      wrapperTaskRun: {
        id: "stage-6-task",
      },
    });
  });

  it("can bootstrap a manifest directly from a root flow run when session registration fails", async () => {
    vi.mocked(readSession).mockResolvedValue(null);
    vi.mocked(readQuestion).mockResolvedValue(undefined);

    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url === "http://localhost:4200/api/flow_runs/live-run") {
        return jsonResponse({
          id: "live-run",
          created: "2026-03-14T10:00:00.000Z",
          parameters: { query: "Why did this launch?" },
        });
      }

      if (url === "http://localhost:4200/api/task_runs/filter") {
        expect(JSON.parse(String(init?.body ?? "{}"))).toEqual({
          flow_runs: { id: { any_: ["live-run"] } },
          sort: "EXPECTED_START_TIME_DESC",
        });
        return jsonResponse([
          {
            id: "stage-0-task",
            name: "stage-0-flow-0",
            state_type: "RUNNING",
            start_time: "2026-03-14T10:00:05.000Z",
            end_time: null,
          },
        ]);
      }

      if (url === "http://localhost:4200/api/flow_runs/filter") {
        return jsonResponse([{ id: "stage-0-subflow" }]);
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
    expect(manifest?.stages["stage-0"]).toMatchObject({
      ownerRootFlowRunId: "live-run",
      stageSubflowRunId: "stage-0-subflow",
      logFlowRunIds: ["stage-0-subflow"],
      wrapperTaskRun: {
        id: "stage-0-task",
        stateType: "RUNNING",
      },
    });
  });
});
