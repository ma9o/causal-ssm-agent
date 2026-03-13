import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("node:fs/promises", () => ({
  mkdir: vi.fn().mockResolvedValue(undefined),
  writeFile: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./_shared", () => ({
  DATA_DIR: "/tmp/data",
  SESSIONS_PATH: "/tmp/data/sessions.json",
  readSessions: vi.fn().mockResolvedValue({}),
  writeSessions: vi.fn().mockResolvedValue(undefined),
}));

import { mkdir, writeFile } from "node:fs/promises";
import { writeSessions } from "./_shared";
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
          code: "kxxsv2",
          question: "How does screen time affect sleep?",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true });

    // Should write query.txt to data/{CODE}/
    expect(mkdir).toHaveBeenCalledWith("/tmp/data/KXXSV2", { recursive: true });
    expect(writeFile).toHaveBeenCalledWith(
      "/tmp/data/KXXSV2/query.txt",
      "How does screen time affect sleep?",
    );

    // Should write sessions.json without the question
    expect(writeSessions).toHaveBeenCalledWith(
      expect.objectContaining({
        KXXSV2: expect.objectContaining({
          createdAt: expect.any(String),
        }),
      }),
    );

    // Verify question is NOT in sessions.json
    const mock = writeSessions as unknown as { mock: { calls: [Record<string, unknown>][] } };
    const written = mock.mock.calls[0][0];
    expect(written.KXXSV2).not.toHaveProperty("question");
    expect(written.KXXSV2).toHaveProperty("createdAt");
  });
});
