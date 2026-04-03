import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn(),
}));

vi.mock("@/lib/server/prefect-runs", () => ({
  findCausalInferenceDeploymentId: vi.fn(),
  findFlowRunIdByIdempotencyKey: vi.fn(),
  launchWorkspaceRootFlowRun: vi.fn(),
  PrefectRunError: class PrefectRunError extends Error {
    status: number;

    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
}));

import { requireWorkspaceAccess } from "@/lib/workspace-access";
import {
  findCausalInferenceDeploymentId,
  findFlowRunIdByIdempotencyKey,
  launchWorkspaceRootFlowRun,
  PrefectRunError,
} from "@/lib/server/prefect-runs";
import { POST } from "./route";

describe("POST /api/runs", () => {
  afterEach(() => {
    vi.clearAllMocks();
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

  it("returns the access error when the workspace is not authorized", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: false,
      response: new Response(JSON.stringify({ error: "Workspace access denied" }), {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }),
    });

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

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({
      error: "Workspace access denied",
    });
    expect(findCausalInferenceDeploymentId).not.toHaveBeenCalled();
  });

  it("returns 502 when the deployment cannot be found", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
      creationPending: false,
    });
    vi.mocked(findCausalInferenceDeploymentId).mockResolvedValue(null);

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
    await expect(response.json()).resolves.toEqual({
      error: "causal-inference deployment not found",
    });
  });

  it("returns the existing root flow run when the same launchId is retried", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
      creationPending: false,
    });
    vi.mocked(findCausalInferenceDeploymentId).mockResolvedValue("dep-123");
    vi.mocked(findFlowRunIdByIdempotencyKey).mockResolvedValue("run-existing");

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
    expect(launchWorkspaceRootFlowRun).not.toHaveBeenCalled();
  });

  it("returns 409 when another run is already active for the workspace", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
      creationPending: false,
    });
    vi.mocked(findCausalInferenceDeploymentId).mockResolvedValue("dep-123");
    vi.mocked(findFlowRunIdByIdempotencyKey).mockResolvedValue(null);
    vi.mocked(launchWorkspaceRootFlowRun).mockResolvedValue({
      status: "busy",
      message: "A run is already active for this workspace.",
      rootFlowRunId: "run-active",
    });

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
  });

  it("launches a new workspace root flow run with the normalized route parameters", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
      creationPending: false,
    });
    vi.mocked(findCausalInferenceDeploymentId).mockResolvedValue("dep-123");
    vi.mocked(findFlowRunIdByIdempotencyKey).mockResolvedValue(null);
    vi.mocked(launchWorkspaceRootFlowRun).mockResolvedValue({
      status: "created",
      rootFlowRunId: "run-456",
    });

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          launchId: " launch-123 ",
          query: " Why is sleep worse after travel? ",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      rootFlowRunId: "run-456",
    });
    expect(launchWorkspaceRootFlowRun).toHaveBeenCalledWith({
      deploymentId: "dep-123",
      idempotencyKey: "launch:USER123:launch-123",
      parameters: {
        workspace_id: "USER123",
        query: "Why is sleep worse after travel?",
      },
      workspaceId: "USER123",
    });
  });

  it("maps Prefect access failures to their HTTP status", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
      creationPending: false,
    });
    vi.mocked(findCausalInferenceDeploymentId).mockResolvedValue("dep-123");
    vi.mocked(findFlowRunIdByIdempotencyKey).mockResolvedValue(null);
    vi.mocked(launchWorkspaceRootFlowRun).mockRejectedValue(
      new PrefectRunError(
        402,
        "Anonymous credits exhausted. Sign in with OpenRouter to continue.",
      ),
    );

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
      error: "Anonymous credits exhausted. Sign in with OpenRouter to continue.",
    });
  });

  it("returns 502 on unexpected launch failures", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "USER123",
      creationPending: false,
    });
    vi.mocked(findCausalInferenceDeploymentId).mockResolvedValue("dep-123");
    vi.mocked(findFlowRunIdByIdempotencyKey).mockResolvedValue(null);
    vi.mocked(launchWorkspaceRootFlowRun).mockRejectedValue(new Error("boom"));

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
    await expect(response.json()).resolves.toEqual({
      error: "Failed to trigger pipeline",
    });
  });
});
