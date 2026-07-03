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
}));

import {
  EpisodeRunError,
  resolveAutoRunExecOptions,
  startAutoRun,
} from "@/lib/server/episode-runs";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { POST } from "./route";

function makeRequest(): Request {
  return new Request("http://localhost/api/analysis/user-1/recompute", { method: "POST" });
}

function grantAccess(workspaceId = "user-1") {
  vi.mocked(requireWorkspaceAccess).mockResolvedValue({
    ok: true,
    workspaceId,
    creationPending: false,
    readOnly: false,
  });
}

describe("POST /api/analysis/[workspaceId]/recompute", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("requires mutable workspace access", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: false,
      response: NextResponse.json({ error: "Workspace access denied" }, { status: 403 }),
    });

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(403);
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(expect.any(Request), "user-1", {
      requireMutable: true,
    });
    expect(startAutoRun).not.toHaveBeenCalled();
  });

  it("starts the auto-run driver with resolved exec options", async () => {
    grantAccess();
    vi.mocked(resolveAutoRunExecOptions).mockResolvedValue({
      openrouter_access_mode: "local",
    });
    vi.mocked(startAutoRun).mockResolvedValue();

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true, workspaceId: "user-1" });
    expect(startAutoRun).toHaveBeenCalledWith("user-1", {
      openrouter_access_mode: "local",
    });
  });

  it("treats an already-running auto-run (facade 409) as success", async () => {
    grantAccess();
    vi.mocked(resolveAutoRunExecOptions).mockResolvedValue({
      openrouter_access_mode: "local",
    });
    vi.mocked(startAutoRun).mockRejectedValue(
      new EpisodeRunError(409, "auto-run already active for user-1"),
    );

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true, workspaceId: "user-1" });
  });

  it("maps access resolution failures to their HTTP status", async () => {
    grantAccess();
    vi.mocked(resolveAutoRunExecOptions).mockRejectedValue(
      new EpisodeRunError(402, "Anonymous credits exhausted. Sign in with OpenRouter to continue."),
    );

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(402);
    expect(startAutoRun).not.toHaveBeenCalled();
  });

  it("returns 502 on unexpected failures", async () => {
    grantAccess();
    vi.mocked(resolveAutoRunExecOptions).mockResolvedValue({
      openrouter_access_mode: "local",
    });
    vi.mocked(startAutoRun).mockRejectedValue(new Error("boom"));

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({ error: "Failed to start recompute" });
  });
});
