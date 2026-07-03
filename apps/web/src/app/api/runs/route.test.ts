import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/episode-runs", () => ({
  EpisodeRunError: class EpisodeRunError extends Error {
    status: number;

    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  startAutoRun: vi.fn(),
  startEpisode: vi.fn(),
}));

import { EpisodeRunError, startAutoRun, startEpisode } from "@/lib/server/episode-runs";
import { POST } from "./route";

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

  it("rejects malformed workspace ids", async () => {
    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "../etc", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(400);
    expect(startEpisode).not.toHaveBeenCalled();
  });

  it("writes the question and starts the auto-run driver", async () => {
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
    expect(startAutoRun).toHaveBeenCalledWith("USER123");
  });

  it("returns 409 when an auto-run is already active for the workspace", async () => {
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

  it("maps facade errors to their HTTP status", async () => {
    vi.mocked(startEpisode).mockRejectedValue(new EpisodeRunError(403, "facade is read-only"));

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({ error: "facade is read-only" });
  });

  it("returns 502 on unexpected launch failures", async () => {
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
