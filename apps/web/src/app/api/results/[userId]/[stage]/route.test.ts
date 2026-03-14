import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("node:fs/promises", () => ({
  readFile: vi.fn(),
}));

import { readFile } from "node:fs/promises";
import { GET } from "./route";

describe("GET /api/results/[userId]/[stage]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("unwraps Prefect payloads that contain non-finite numbers instead of returning a fake 404", async () => {
    vi.mocked(readFile).mockResolvedValue(
      JSON.stringify({
        metadata: { storage_key: "/tmp/data/user/run/stage-5a.json" },
        result:
          '{"value":NaN,"upper":Infinity,"lower":-Infinity,"label":"Infinity should stay a string"}',
      }),
    );

    const response = await GET(new Request("http://localhost/api/results/user/stage-5a"), {
      params: Promise.resolve({ userId: "user", stage: "stage-5a" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      value: null,
      upper: null,
      lower: null,
      label: "Infinity should stay a string",
    });
  });

  it("returns a parse error when the persisted payload is invalid", async () => {
    vi.mocked(readFile).mockResolvedValue('{"metadata":{},"result":"{"}');

    const response = await GET(new Request("http://localhost/api/results/user/stage-5a"), {
      params: Promise.resolve({ userId: "user", stage: "stage-5a" }),
    });

    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        error: expect.stringContaining("Invalid persisted data for stage-5a"),
      }),
    );
  });
});
