import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/storage", () => ({
  writeData: vi.fn().mockResolvedValue(undefined),
  ensureDir: vi.fn().mockResolvedValue(undefined),
  LOCAL_DATA_DIR: "/tmp/data",
}));

vi.mock("./_shared", () => ({
  readSession: vi.fn().mockResolvedValue(null),
  normalizeSession: vi.fn((session) => ({
    createdAt: session?.createdAt ?? "2026-03-14T00:00:00.000Z",
    rootFlowRunIds: session?.rootFlowRunIds ?? [],
  })),
  appendSessionRootFlowRunId: vi.fn((session, rootFlowRunId) => ({
    createdAt: session?.createdAt ?? "2026-03-14T00:00:00.000Z",
    rootFlowRunIds: [...(session?.rootFlowRunIds ?? []), rootFlowRunId],
  })),
  writeSession: vi.fn().mockResolvedValue(undefined),
}));

import { writeData, ensureDir } from "@/lib/storage";
import { readSession, writeSession } from "./_shared";
import { POST } from "./route";

describe("POST /api/sessions", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("writes query.txt and the per-user session lineage file", async () => {
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

    // Should create user directory and write query.txt
    expect(ensureDir).toHaveBeenCalledWith("openrouter-user-123");
    expect(writeData).toHaveBeenCalledWith(
      "openrouter-user-123/query.txt",
      "How does screen time affect sleep?",
    );

    expect(writeSession).toHaveBeenCalledWith("openrouter-user-123", {
      createdAt: "2026-03-14T00:00:00.000Z",
      rootFlowRunIds: [],
    });
  });

  it("appends a new flow run to the stored session lineage", async () => {
    vi.mocked(readSession).mockResolvedValue({
      createdAt: "2026-03-13T00:00:00.000Z",
      rootFlowRunIds: ["older-run", "old-run"],
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
    expect(writeSession).toHaveBeenCalledWith("openrouter-user-123", {
      createdAt: "2026-03-13T00:00:00.000Z",
      rootFlowRunIds: ["older-run", "old-run", "new-run"],
    });
  });
});
