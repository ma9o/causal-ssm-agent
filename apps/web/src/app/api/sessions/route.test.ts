import { afterEach, describe, expect, it, vi } from "vitest";

const { mkdir, writeFile, readSessions } = vi.hoisted(() => ({
  mkdir: vi.fn(),
  writeFile: vi.fn(),
  readSessions: vi.fn(),
}));

vi.mock("node:fs/promises", () => ({
  mkdir,
  writeFile,
}));

vi.mock("./_shared", () => ({
  SESSIONS_PATH: "/tmp/results/sessions.json",
  readSessions,
}));

import { POST } from "./route";

describe("POST /api/sessions", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("creates the sessions directory before writing the file", async () => {
    readSessions.mockResolvedValue({});
    mkdir.mockResolvedValue(undefined);
    writeFile.mockResolvedValue(undefined);

    const response = await POST(
      new Request("http://localhost/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: "kxxsv2",
          runId: "run-123",
          question: "How does screen time affect sleep?",
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true });
    expect(mkdir).toHaveBeenCalledWith("/tmp/results", { recursive: true });
    expect(writeFile).toHaveBeenCalledWith(
      "/tmp/results/sessions.json",
      expect.stringContaining('"KXXSV2"'),
    );
  });
});
