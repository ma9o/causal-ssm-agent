import { afterEach, describe, expect, it, vi } from "vitest";
import { NextResponse } from "next/server";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn(),
}));

vi.mock("@/lib/server/episode-runs", () => ({
  EpisodeRunError: class EpisodeRunError extends Error {
    status: number;

    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  resolveAutoRunExecOptions: vi.fn(),
  startAutoRun: vi.fn(),
  startEpisode: vi.fn(),
}));

import { requireWorkspaceAccess } from "@/lib/workspace-access";
import {
  EpisodeRunError,
  resolveAutoRunExecOptions,
  startAutoRun,
  startEpisode,
} from "@/lib/server/episode-runs";
import { POST } from "./route";

function grantAccess(workspaceId = "USER123") {
  vi.mocked(requireWorkspaceAccess).mockResolvedValue({
    ok: true,
    workspaceId,
    creationPending: false,
    readOnly: false,
  });
}

describe("POST /api/runs", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("requires a query", async () => {
    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123" }),
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "query is required",
    });
  });

  it("returns the access error when the workspace is not authorized", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: false,
      response: NextResponse.json({ error: "Workspace access denied" }, { status: 403 }),
    });

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({
      error: "Workspace access denied",
    });
    expect(startEpisode).not.toHaveBeenCalled();
  });

  it("writes the question and starts the auto-run driver", async () => {
    grantAccess();
    vi.mocked(resolveAutoRunExecOptions).mockResolvedValue({
      openrouter_access_mode: "local",
    });
    vi.mocked(startEpisode).mockResolvedValue({} as never);
    vi.mocked(startAutoRun).mockResolvedValue();

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          query: " Why is sleep worse after travel? ",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ workspaceId: "USER123" });
    expect(startEpisode).toHaveBeenCalledWith("USER123", "Why is sleep worse after travel?");
    expect(startAutoRun).toHaveBeenCalledWith("USER123", {
      openrouter_access_mode: "local",
    });
  });

  it("returns 409 when an auto-run is already active for the workspace", async () => {
    grantAccess();
    vi.mocked(resolveAutoRunExecOptions).mockResolvedValue({
      openrouter_access_mode: "local",
    });
    vi.mocked(startEpisode).mockResolvedValue({} as never);
    vi.mocked(startAutoRun).mockRejectedValue(
      new EpisodeRunError(409, "auto-run already active for USER123"),
    );

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: "A run is already active for this workspace.",
    });
  });

  it("maps access resolution failures to their HTTP status", async () => {
    grantAccess();
    vi.mocked(resolveAutoRunExecOptions).mockRejectedValue(
      new EpisodeRunError(402, "Anonymous credits exhausted. Sign in with OpenRouter to continue."),
    );

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(402);
    await expect(response.json()).resolves.toEqual({
      error: "Anonymous credits exhausted. Sign in with OpenRouter to continue.",
    });
    expect(startEpisode).not.toHaveBeenCalled();
  });

  it("returns 502 on unexpected launch failures", async () => {
    grantAccess();
    vi.mocked(resolveAutoRunExecOptions).mockResolvedValue({
      openrouter_access_mode: "local",
    });
    vi.mocked(startEpisode).mockRejectedValue(new Error("boom"));

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      error: "Failed to trigger pipeline",
    });
  });
});
