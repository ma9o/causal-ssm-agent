import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../sessions/_shared", () => ({
  readSessions: vi.fn(),
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

  it("cancels the current flow run before starting the replay and stores the new flowRunId", async () => {
    vi.mocked(readSessions).mockResolvedValue({
      ABC123: {
        createdAt: "2026-03-13T10:00:00.000Z",
        flowRunId: "old-run",
      },
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "old-run",
          parameters: { code: "ABC123", query: "Why?" },
          state: { type: "RUNNING", name: "Running" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ status: "ACCEPT" }))
      .mockResolvedValueOnce(
        jsonResponse({
          id: "old-run",
          parameters: { code: "ABC123", query: "Why?" },
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
          code: "abc123",
          stageId: "stage-1a",
          stageData: { latent_model: { constructs: [] } },
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      resumeFrom: "stage-1b",
      flowRunId: "new-run",
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
            code: "ABC123",
            query: "Why?",
            stage_overrides: {
              "stage-1a": { latent_model: { constructs: [] } },
            },
          },
        }),
      }),
    );
    expect(writeSessions).toHaveBeenCalledWith({
      ABC123: {
        createdAt: "2026-03-13T10:00:00.000Z",
        flowRunId: "new-run",
      },
    });
  });

  it("skips cancellation when the tracked flow run is already terminal", async () => {
    vi.mocked(readSessions).mockResolvedValue({
      ABC123: {
        createdAt: "2026-03-13T10:00:00.000Z",
        flowRunId: "done-run",
      },
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "done-run",
          parameters: { code: "ABC123" },
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
          code: "ABC123",
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
});
