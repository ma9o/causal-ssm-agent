import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn().mockImplementation(async (_request: Request, workspaceId: string) => ({
    ok: true,
    workspaceId,
  })),
}));

vi.mock("@/lib/storage", () => ({
  readData: vi.fn(),
  LOCAL_DATA_DIR: "/tmp/data",
}));

import { readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { GET } from "./route";

describe("GET /api/results/[workspaceId]/[stage]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("unwraps Prefect payloads that contain non-finite numbers instead of returning a fake 404", async () => {
    vi.mocked(readData).mockResolvedValue(
      JSON.stringify({
        metadata: { storage_key: "/tmp/data/user/run/stage-5a.json" },
        result:
          '{"value":NaN,"upper":Infinity,"lower":-Infinity,"label":"Infinity should stay a string"}',
      }),
    );

    const response = await GET(new Request("http://localhost/api/results/user/stage-5a"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5a" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      value: null,
      upper: null,
      lower: null,
      label: "Infinity should stay a string",
    });
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(expect.any(Request), "user");
  });

  it("returns a parse error when the persisted payload is invalid", async () => {
    vi.mocked(readData).mockResolvedValue('{"metadata":{},"result":"{"}');

    const response = await GET(new Request("http://localhost/api/results/user/stage-5a"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5a" }),
    });

    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        error: expect.stringContaining("Invalid persisted data for stage-5a"),
      }),
    );
  });
});
