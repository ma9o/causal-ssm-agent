import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { uploadFile } from "./endpoints";

describe("uploadFile", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllGlobals();
  });

  it("sends FormData with file and workspaceId", async () => {
    const mockResponse = { path: "/uploads/test.json" };
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as Response);

    const file = new File(["content"], "test.json", { type: "application/json" });
    const result = await uploadFile(file, "user-1");

    expect(result).toEqual(mockResponse);

    const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/upload");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    const formData = init.body as FormData;
    expect(formData.get("workspaceId")).toBe("user-1");
  });

  it("throws on upload failure", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 413,
    } as Response);

    const file = new File(["x"], "big.json");
    await expect(uploadFile(file, "user-1")).rejects.toThrow("Upload failed: 413");
  });
});
