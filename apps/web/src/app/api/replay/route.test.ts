import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../sessions/_shared", () => ({
  readSessions: vi.fn(),
  getLatestSessionRootFlowRunId: vi.fn((session) => session?.rootFlowRunIds?.at(-1) ?? null),
  appendSessionRootFlowRunId: vi.fn((session, rootFlowRunId) => ({
    createdAt: session?.createdAt ?? "2026-03-14T00:00:00.000Z",
    rootFlowRunIds: [...(session?.rootFlowRunIds ?? []), rootFlowRunId],
  })),
  writeSessions: vi.fn().mockResolvedValue(undefined),
}));

import { readSessions, writeSessions } from "../sessions/_shared";
import { POST } from "./route";

const originalFetch = globalThis.fetch;

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
  } as Response;
}

describe("POST /api/replay", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    globalThis.fetch = originalFetch;
  });

  it("cancels the current flow run before starting the replay and appends the new run to the session lineage", async () => {
    vi.mocked(readSessions).mockResolvedValue({
      "user-123": {
        createdAt: "2026-03-13T10:00:00.000Z",
        rootFlowRunIds: ["older-run", "old-run"],
      },
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "old-run",
          parameters: { user_id: "user-123", query: "Why?" },
          state: { type: "RUNNING", name: "Running" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ status: "ACCEPT" }))
      .mockResolvedValueOnce(
        jsonResponse({
          id: "old-run",
          parameters: { user_id: "user-123", query: "Why?" },
          state: { type: "CANCELLED", name: "Cancelled" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: "user-123",
          stageId: "stage-1a",
          stageData: { latent_model: { constructs: [] } },
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      resumeFrom: "stage-1b",
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:4200/api/flow_runs/old-run",
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:4200/api/flow_runs/old-run/set_state",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          state: { type: "CANCELLING", name: "Cancelling" },
          force: false,
        }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "http://localhost:4200/api/deployments/dep-1/create_flow_run",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          parameters: {
            user_id: "user-123",
            query: "Why?",
            stage_overrides: {
              "stage-1a": { latent_model: { constructs: [] } },
            },
          },
        }),
      }),
    );
    expect(writeSessions).toHaveBeenCalledWith({
      "user-123": {
        createdAt: "2026-03-13T10:00:00.000Z",
        rootFlowRunIds: ["older-run", "old-run", "new-run"],
      },
    });
  });

  it("skips cancellation when the tracked flow run is already terminal", async () => {
    vi.mocked(readSessions).mockResolvedValue({
      "user-123": {
        createdAt: "2026-03-13T10:00:00.000Z",
        rootFlowRunIds: ["done-run"],
      },
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "done-run",
          parameters: { user_id: "user-123" },
          state: { type: "COMPLETED", name: "Completed" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: "user-123",
          stageId: "stage-4",
          stageData: { model_spec: {}, priors: {} },
        }),
      }),
    );

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://localhost:4200/api/flow_runs/done-run/set_state",
      expect.anything(),
    );
  });

  it("drops stale resume bounds from the previous run before creating the replay", async () => {
    vi.mocked(readSessions).mockResolvedValue({
      "user-123": {
        createdAt: "2026-03-13T10:00:00.000Z",
        rootFlowRunIds: ["resume-run"],
      },
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "resume-run",
          parameters: {
            user_id: "user-123",
            start_stage: "stage-4b",
            end_stage: "stage-6",
            stage_overrides: {
              "stage-1a": { latent_model: { constructs: ["existing"] } },
            },
          },
          state: { type: "COMPLETED", name: "Completed" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: "user-123",
          stageId: "stage-4",
          stageData: { model_spec: { nodes: [] }, priors: {} },
        }),
      }),
    );

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://localhost:4200/api/deployments/dep-1/create_flow_run",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          parameters: {
            user_id: "user-123",
            stage_overrides: {
              "stage-1a": { latent_model: { constructs: ["existing"] } },
              "stage-4": { model_spec: { nodes: [] }, priors: {} },
            },
          },
        }),
      }),
    );
  });
});
