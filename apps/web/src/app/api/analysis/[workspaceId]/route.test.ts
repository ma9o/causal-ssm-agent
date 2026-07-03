import { afterEach, describe, expect, it, vi } from "vitest";
import type { AnalysisManifest } from "@/lib/api/analysis";

vi.mock("../_shared", () => ({
  buildAnalysisManifest: vi.fn(),
}));

vi.mock("@/lib/server/episode-runs", () => ({
  getFacadeCapabilities: vi.fn(),
}));

import { getFacadeCapabilities } from "@/lib/server/episode-runs";
import { buildAnalysisManifest } from "../_shared";
import { GET } from "./route";

const manifest = {
  workspaceId: "user-123",
  createdAt: "2026-05-07T00:00:00.000Z",
  question: "Did escitalopram help?",
  stages: {},
} as unknown as AnalysisManifest;

describe("GET /api/analysis/[workspaceId]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("derives readOnly from the facade capabilities", async () => {
    vi.mocked(buildAnalysisManifest).mockResolvedValue(manifest);
    vi.mocked(getFacadeCapabilities).mockResolvedValue({ moves_enabled: false });

    const response = await GET(new Request("http://localhost/api/analysis/user-123"), {
      params: Promise.resolve({ workspaceId: "user-123" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      workspaceId: "user-123",
      readOnly: true,
    });
  });

  it("reports writable when the facade serves moves", async () => {
    vi.mocked(buildAnalysisManifest).mockResolvedValue(manifest);
    vi.mocked(getFacadeCapabilities).mockResolvedValue({ moves_enabled: true });

    const response = await GET(new Request("http://localhost/api/analysis/user-123"), {
      params: Promise.resolve({ workspaceId: "user-123" }),
    });

    await expect(response.json()).resolves.toMatchObject({ readOnly: false });
  });

  it("rejects malformed workspace ids", async () => {
    const response = await GET(new Request("http://localhost/api/analysis/bad"), {
      params: Promise.resolve({ workspaceId: "../etc" }),
    });

    expect(response.status).toBe(400);
    expect(buildAnalysisManifest).not.toHaveBeenCalled();
  });

  it("returns 404 when the manifest cannot be built", async () => {
    vi.mocked(buildAnalysisManifest).mockResolvedValue(null);
    vi.mocked(getFacadeCapabilities).mockResolvedValue({ moves_enabled: true });

    const response = await GET(new Request("http://localhost/api/analysis/user-123"), {
      params: Promise.resolve({ workspaceId: "user-123" }),
    });

    expect(response.status).toBe(404);
  });
});
