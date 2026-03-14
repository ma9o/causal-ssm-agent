import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("node:fs/promises", () => ({
  mkdir: vi.fn().mockResolvedValue(undefined),
  writeFile: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./_shared", () => ({
  DATA_DIR: "/tmp/data",
  SESSIONS_PATH: "/tmp/data/sessions.json",
  readSessions: vi.fn().mockResolvedValue({}),
  normalizeSession: vi.fn((session) => ({
    createdAt: session?.createdAt ?? "2026-03-14T00:00:00.000Z",
    rootFlowRunIds: session?.rootFlowRunIds ?? [],
  })),
  appendSessionRootFlowRunId: vi.fn((session, rootFlowRunId) => ({
    createdAt: session?.createdAt ?? "2026-03-14T00:00:00.000Z",
    rootFlowRunIds: [...(session?.rootFlowRunIds ?? []), rootFlowRunId],
  })),
  writeSessions: vi.fn().mockResolvedValue(undefined),
}));

import { mkdir, writeFile } from "node:fs/promises";
import { readSessions, writeSessions } from "./_shared";
import { POST } from "./route";

describe("POST /api/sessions", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("writes query.txt and sessions.json", async () => {
    const response = await POST(
      new Request("http://localhost/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: "openrouter-user-123",
          question: "How does screen time affect sleep?",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true });

    // Should write query.txt to data/{userId}/
    expect(mkdir).toHaveBeenCalledWith("/tmp/data/openrouter-user-123", { recursive: true });
    expect(writeFile).toHaveBeenCalledWith(
      "/tmp/data/openrouter-user-123/query.txt",
      "How does screen time affect sleep?",
    );

    // Should write sessions.json without the question
    expect(writeSessions).toHaveBeenCalledWith(
      expect.objectContaining({
        "openrouter-user-123": expect.objectContaining({
          createdAt: expect.any(String),
        }),
      }),
    );

    // Verify question is NOT in sessions.json
    const mock = writeSessions as unknown as { mock: { calls: [Record<string, unknown>][] } };
    const written = mock.mock.calls[0][0];
    expect(written["openrouter-user-123"]).not.toHaveProperty("question");
    expect(written["openrouter-user-123"]).toHaveProperty("createdAt");
    expect(written["openrouter-user-123"]).toHaveProperty("rootFlowRunIds", []);
  });

  it("appends a new flow run to the stored session lineage", async () => {
    vi.mocked(readSessions).mockResolvedValue({
      "openrouter-user-123": {
        createdAt: "2026-03-13T00:00:00.000Z",
        rootFlowRunIds: ["older-run", "old-run"],
      },
    });

    const response = await POST(
      new Request("http://localhost/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: "openrouter-user-123",
          question: "How does screen time affect sleep?",
          rootFlowRunId: "new-run",
        }),
      }),
    );

    expect(response.status).toBe(200);
    expect(writeSessions).toHaveBeenCalledWith({
      "openrouter-user-123": {
        createdAt: "2026-03-13T00:00:00.000Z",
        rootFlowRunIds: ["older-run", "old-run", "new-run"],
      },
    });
  });
});
