import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn().mockImplementation(async (_request: Request, workspaceId: string) => ({
    ok: true,
    workspaceId,
  })),
}));

vi.mock("@/lib/server/openrouter-access", async (importOriginal) => ({
  ...await importOriginal(),
  resolveOpenRouterAccess: vi.fn(),
}));

vi.mock("@/lib/server/byok-secret-store", () => ({
  createByokSecretRef: vi.fn(),
  deleteByokSecretRef: vi.fn(),
}));

vi.mock("@/lib/server/workspace-run-lock", () => ({
  claimWorkspaceRunSlot: vi.fn(),
  releaseWorkspaceRunSlot: vi.fn(),
}));

import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { createByokSecretRef, deleteByokSecretRef } from "@/lib/server/byok-secret-store";
import { resolveOpenRouterAccess } from "@/lib/server/openrouter-access";
import { claimWorkspaceRunSlot, releaseWorkspaceRunSlot } from "@/lib/server/workspace-run-lock";
import { POST } from "./route";

const originalFetch = globalThis.fetch;

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

describe("POST /api/replay", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    globalThis.fetch = originalFetch;
  });

  it("returns 409 when another run is already active for the workspace", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "anonymous",
      apiKey: "anonymous-key",
      creditStatus: "available",
    });
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-busy",
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "done-run",
          parameters: { workspace_id: "user-123" },
          state: { type: "COMPLETED", name: "Completed" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse([{ id: "active-run" }]));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-4",
          stageData: { model_spec: {}, authored_priors: {}, resolved_priors: [] },
          rootFlowRunId: "done-run",
        }),
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: "A run is already active for this workspace.",
      rootFlowRunId: "active-run",
    });
    expect(releaseWorkspaceRunSlot).toHaveBeenCalledWith("user-123", "slot-busy");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("cancels the current flow run before starting the replay", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "anonymous",
      apiKey: "anonymous-key",
      creditStatus: "available",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("anonymous-ref-1");
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-1",
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "old-run",
          parameters: { workspace_id: "user-123", query: "Why?" },
          state: { type: "RUNNING", name: "Running" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse({ status: "ACCEPT" }))
      .mockResolvedValueOnce(
        jsonResponse({
          id: "old-run",
          parameters: { workspace_id: "user-123", query: "Why?" },
          state: { type: "CANCELLED", name: "Cancelled" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-1a",
          stageData: { latent_model: { constructs: [] } },
          rootFlowRunId: "old-run",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      resumeFrom: "stage-1b",
      rootFlowRunId: "new-run",
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:4200/api/flow_runs/old-run",
      undefined,
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:4200/api/deployments/filter",
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
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
      "http://localhost:4200/api/flow_runs/filter",
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      "http://localhost:4200/api/deployments/dep-1/create_flow_run",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }),
    );
    const createCall = fetchMock.mock.calls[5]?.[1] as { body?: string };
    const createBody = JSON.parse(createCall.body ?? "{}");
    expect(createBody).toMatchObject({
      name: expect.stringMatching(/^replay-user-123-stage-1a-\d+$/),
      tags: ["replay", "interactive", "stage-1a", "workspace:user-123"],
      context: {
        replay_kind: "stage_override",
        edited_stage_id: "stage-1a",
        source_root_flow_run_id: "old-run",
      },
      labels: {
        replay: true,
        workspace_id: "user-123",
        edited_stage: "stage-1a",
        source_root_flow_run_id: "old-run",
      },
      parameters: {
        workspace_id: "user-123",
        query: "Why?",
        start_stage: "stage-1a",
        stage_overrides: {
          "stage-1a": { latent_model: { constructs: [] } },
        },
        openrouter_access_mode: "anonymous",
        openrouter_secret_ref: "anonymous-ref-1",
      },
    });
    expect(createBody.idempotency_key).toMatch(/^replay:user-123:stage-1a:[0-9a-f]{64}$/);
    expect(releaseWorkspaceRunSlot).toHaveBeenCalledWith("user-123", "slot-1");
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(expect.any(Request), "user-123", {
      requireMutable: true,
    });
  });

  it("skips cancellation when the tracked flow run is already terminal", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "anonymous",
      apiKey: "anonymous-key",
      creditStatus: "available",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("anonymous-ref-2");
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-2",
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "done-run",
          parameters: { workspace_id: "user-123" },
          state: { type: "COMPLETED", name: "Completed" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-4",
          stageData: { model_spec: {}, authored_priors: {}, resolved_priors: [] },
          rootFlowRunId: "done-run",
        }),
      }),
    );

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(4);
    const createCall = fetchMock.mock.calls[3]?.[1] as { body?: string };
    expect(JSON.parse(createCall.body ?? "{}")).toMatchObject({
      parameters: {
        openrouter_access_mode: "anonymous",
        openrouter_secret_ref: "anonymous-ref-2",
      },
    });
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://localhost:4200/api/flow_runs/done-run/set_state",
      expect.anything(),
    );
  });

  it("drops stale resume bounds from the previous run before creating the replay", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "anonymous",
      apiKey: "anonymous-key",
      creditStatus: "available",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("anonymous-ref-3");
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-3",
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "resume-run",
          parameters: {
            workspace_id: "user-123",
            start_stage: "stage-5b",
            end_stage: "stage-6",
            stage_overrides: {
              "stage-1a": { latent_model: { constructs: ["existing"] } },
            },
          },
          state: { type: "COMPLETED", name: "Completed" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-4",
          stageData: { model_spec: { nodes: [] }, authored_priors: {}, resolved_priors: [] },
          rootFlowRunId: "resume-run",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      resumeFrom: "stage-5b",
      rootFlowRunId: "new-run",
    });
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://localhost:4200/api/deployments/dep-1/create_flow_run",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }),
    );
    const createCall = fetchMock.mock.calls[3]?.[1] as { body?: string };
    expect(JSON.parse(createCall.body ?? "{}")).toMatchObject({
      name: expect.stringMatching(/^replay-user-123-stage-4-\d+$/),
      tags: ["replay", "interactive", "stage-4", "workspace:user-123"],
      context: {
        replay_kind: "stage_override",
        edited_stage_id: "stage-4",
        source_root_flow_run_id: "resume-run",
      },
      labels: {
        replay: true,
        workspace_id: "user-123",
        edited_stage: "stage-4",
        source_root_flow_run_id: "resume-run",
      },
      parameters: {
        workspace_id: "user-123",
        start_stage: "stage-4",
        stage_overrides: {
          "stage-1a": { latent_model: { constructs: ["existing"] } },
          "stage-4": { model_spec: { nodes: [] }, authored_priors: {}, resolved_priors: [] },
        },
        openrouter_access_mode: "anonymous",
        openrouter_secret_ref: "anonymous-ref-3",
      },
    });
  });

  it("mints a fresh OpenRouter secret ref for replay instead of reusing the stale one", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "user",
      apiKey: "user-key",
      userId: "or-user-123",
    });
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-4",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("fresh-byok-ref");

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "resume-run",
          parameters: {
            workspace_id: "user-123",
            query: "Why?",
            openrouter_access_mode: "anonymous",
            openrouter_secret_ref: "stale-openrouter-ref",
          },
          state: { type: "COMPLETED", name: "Completed" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-4",
          stageData: { model_spec: {}, authored_priors: {}, resolved_priors: [] },
          rootFlowRunId: "resume-run",
        }),
      }),
    );

    expect(response.status).toBe(200);
    expect(createByokSecretRef).toHaveBeenCalledWith("user-key");
    expect(deleteByokSecretRef).not.toHaveBeenCalled();
    const createCall = fetchMock.mock.calls[3]?.[1] as { body?: string };
    expect(JSON.parse(createCall.body ?? "{}")).toMatchObject({
      tags: ["replay", "interactive", "stage-4", "workspace:user-123"],
      parameters: {
        workspace_id: "user-123",
        query: "Why?",
        start_stage: "stage-4",
        stage_overrides: {
          "stage-4": { model_spec: {}, authored_priors: {}, resolved_priors: [] },
        },
        openrouter_access_mode: "user",
        openrouter_secret_ref: "fresh-byok-ref",
      },
    });
  });

  it("retries retryable Prefect API responses during replay creation", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "anonymous",
      apiKey: "anonymous-key",
      creditStatus: "available",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("anonymous-ref-5");
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-5",
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          id: "done-run",
          parameters: { workspace_id: "user-123" },
          state: { type: "COMPLETED", name: "Completed" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ error: "rate limited" }, 429, { "Retry-After": "0" }))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-4",
          stageData: { model_spec: {}, authored_priors: {}, resolved_priors: [] },
          rootFlowRunId: "done-run",
        }),
      }),
    );

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://localhost:4200/api/deployments/dep-1/create_flow_run",
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "http://localhost:4200/api/deployments/dep-1/create_flow_run",
      expect.objectContaining({ method: "POST" }),
    );
    const createCall = fetchMock.mock.calls[4]?.[1] as { body?: string };
    expect(JSON.parse(createCall.body ?? "{}")).toMatchObject({
      parameters: {
        openrouter_access_mode: "anonymous",
        openrouter_secret_ref: "anonymous-ref-5",
      },
    });
  });

  it("falls back to the latest tagged workspace run when no rootFlowRunId is provided", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "anonymous",
      apiKey: "anonymous-key",
      creditStatus: "available",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("anonymous-ref-6");
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-6",
    });

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([{ id: "bootstrap-run" }]))
      .mockResolvedValueOnce(
        jsonResponse({
          id: "bootstrap-run",
          parameters: { workspace_id: "user-123", query: "Why?" },
          state: { type: "COMPLETED", name: "Completed" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-1" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "new-run" }));

    globalThis.fetch = fetchMock as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-4",
          stageData: { model_spec: {}, authored_priors: {}, resolved_priors: [] },
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      resumeFrom: "stage-5b",
      rootFlowRunId: "new-run",
    });
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:4200/api/flow_runs/filter",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          flow_runs: {
            tags: { all_: ["workspace:user-123"] },
            parent_task_run_id: { is_null_: true },
          },
          sort: "START_TIME_DESC",
          limit: 1,
        }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:4200/api/flow_runs/bootstrap-run",
      undefined,
    );
    const createCall = fetchMock.mock.calls[4]?.[1] as { body?: string };
    expect(JSON.parse(createCall.body ?? "{}")).toMatchObject({
      parameters: {
        openrouter_access_mode: "anonymous",
        openrouter_secret_ref: "anonymous-ref-6",
      },
    });
  });
});
