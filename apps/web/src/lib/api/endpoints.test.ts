import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock the client module before importing endpoints
vi.mock("./client", () => ({
  apiFetch: vi.fn(),
}));

import { apiFetch } from "./client";
import { getStageResult, uploadFile } from "./endpoints";

describe("getStageResult", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("calls apiFetch with correct path", async () => {
    const mockData = { some: "data" };
    vi.mocked(apiFetch).mockResolvedValue(mockData);

    const result = await getStageResult("ABC123", "stage-0");

    expect(apiFetch).toHaveBeenCalledWith("/api/results/ABC123/stage-0");
    expect(result).toEqual(mockData);
  });

  it("propagates errors from apiFetch", async () => {
    vi.mocked(apiFetch).mockRejectedValue(new Error("API error 500: Server Error"));

    await expect(getStageResult("DEF456", "stage-3")).rejects.toThrow("API error 500");
  });
});

describe("uploadFile", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllGlobals();
  });

  it("sends FormData with file, workspaceId, and accessCode", async () => {
    const mockResponse = { path: "/uploads/test.json" };
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as Response);

    const file = new File(["content"], "test.json", { type: "application/json" });
    const result = await uploadFile(file, "user-1", "access-code-1");

    expect(result).toEqual(mockResponse);

    const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/upload");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    const formData = init.body as FormData;
    expect(formData.get("workspaceId")).toBe("user-1");
    expect(formData.get("accessCode")).toBe("access-code-1");
  });

  it("throws on upload failure", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 413,
    } as Response);

    const file = new File(["x"], "big.json");
    await expect(uploadFile(file, "user-1", "access-code-1")).rejects.toThrow(
      "Upload failed: 413",
    );
  });
});
