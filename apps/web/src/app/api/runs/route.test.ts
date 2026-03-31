import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn(),
}));

vi.mock("@/lib/server/openrouter-access", () => ({
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

import {
  requireWorkspaceAccess,
} from "@/lib/workspace-access";
import {
  createByokSecretRef,
  deleteByokSecretRef,
} from "@/lib/server/byok-secret-store";
import { resolveOpenRouterAccess } from "@/lib/server/openrouter-access";
import {
  claimWorkspaceRunSlot,
  releaseWorkspaceRunSlot,
} from "@/lib/server/workspace-run-lock";
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

describe("POST /api/runs", () => {
  afterEach(() => {
    vi.clearAllMocks();
    globalThis.fetch = originalFetch;
  });

  it("requires a launchId so initial launches can be idempotent", async () => {
    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          query: "Why?",
        }),
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "launchId is required",
    });
  });

  it("returns 409 when Prefect already has an active run for the workspace", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
    });
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "trial",
      apiKey: "trial-key",
      creditStatus: "available",
    });
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-busy",
    });

    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-123" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(
        jsonResponse([{ id: "run-active" }]),
      ) as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          launchId: "launch-123",
          query: "Why?",
        }),
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: "A run is already active for this workspace.",
      rootFlowRunId: "run-active",
    });
    expect(createByokSecretRef).not.toHaveBeenCalled();
    expect(releaseWorkspaceRunSlot).toHaveBeenCalledWith(
      "USER123",
      "slot-busy",
    );
  });

  it("returns the existing root flow run when the same launchId is retried", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
    });
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-repeat",
    });

    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-123" }]))
      .mockResolvedValueOnce(
        jsonResponse([{ id: "run-existing" }]),
      ) as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          launchId: "launch-123",
          query: "Why?",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      rootFlowRunId: "run-existing",
    });
    expect(resolveOpenRouterAccess).not.toHaveBeenCalled();
    expect(createByokSecretRef).not.toHaveBeenCalled();
    expect(claimWorkspaceRunSlot).not.toHaveBeenCalled();
    expect(releaseWorkspaceRunSlot).not.toHaveBeenCalled();
  });

  it("creates a run with a user-scoped OpenRouter key on the server", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
    });
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "user",
      apiKey: "user-key",
    });
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-123",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("byok-ref-123");

    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-123" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "run-456" })) as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          launchId: "launch-123",
          query: "Why is sleep worse after travel?",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      rootFlowRunId: "run-456",
    });
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(
      expect.any(Request),
      "USER123",
    );
    expect(claimWorkspaceRunSlot).toHaveBeenCalledWith("USER123");
    expect(releaseWorkspaceRunSlot).toHaveBeenCalledWith("USER123", "slot-123");
    expect(resolveOpenRouterAccess).toHaveBeenCalledWith();
    expect(globalThis.fetch).toHaveBeenNthCalledWith(
      4,
      "http://localhost:4200/api/deployments/dep-123/create_flow_run",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          tags: ["workspace:USER123"],
          idempotency_key: "launch:USER123:launch-123",
          parameters: {
            workspace_id: "USER123",
            query: "Why is sleep worse after travel?",
            openrouter_access_mode: "user",
            openrouter_secret_ref: "byok-ref-123",
          },
        }),
      }),
    );
    expect(deleteByokSecretRef).not.toHaveBeenCalled();
  });

  it("creates a run with a trial-scoped OpenRouter key on the server", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
    });
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "trial",
      apiKey: "trial-key",
      creditStatus: "available",
    });
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-trial",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("trial-ref-123");

    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-123" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "run-trial" })) as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          launchId: "launch-trial",
          query: "Why?",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      rootFlowRunId: "run-trial",
    });
    expect(createByokSecretRef).toHaveBeenCalledWith("trial-key");
    expect(globalThis.fetch).toHaveBeenNthCalledWith(
      4,
      "http://localhost:4200/api/deployments/dep-123/create_flow_run",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          tags: ["workspace:USER123"],
          idempotency_key: "launch:USER123:launch-trial",
          parameters: {
            workspace_id: "USER123",
            query: "Why?",
            openrouter_access_mode: "trial",
            openrouter_secret_ref: "trial-ref-123",
          },
        }),
      }),
    );
  });

  it("blocks anonymous runs when trial exhaustion is known", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
    });
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "none",
      reason: "trial_exhausted",
    });
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-123" }]))
      .mockResolvedValueOnce(jsonResponse([])) as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          launchId: "launch-123",
          query: "Why?",
        }),
      }),
    );

    expect(response.status).toBe(402);
    await expect(response.json()).resolves.toEqual({
      error: "Trial credits exhausted. Sign in with OpenRouter to continue.",
    });
    expect(claimWorkspaceRunSlot).not.toHaveBeenCalled();
  });

  it("cleans up the BYOK ref when Prefect run creation fails", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
    });
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "user",
      apiKey: "user-key",
    });
    vi.mocked(claimWorkspaceRunSlot).mockResolvedValue({
      status: "claimed",
      reservationId: "slot-456",
    });
    vi.mocked(createByokSecretRef).mockResolvedValue("byok-ref-456");

    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([{ id: "dep-123" }]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValue(
        jsonResponse({ error: "boom" }, 502, { "Retry-After": "0" }),
      ) as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          launchId: "launch-123",
          query: "Why?",
        }),
      }),
    );

    expect(response.status).toBe(502);
    expect(deleteByokSecretRef).toHaveBeenCalledWith("byok-ref-456");
    expect(releaseWorkspaceRunSlot).toHaveBeenCalledWith("USER123", "slot-456");
  });
});
