import { afterEach, describe, expect, it, vi } from "vitest";
import type { AnalysisManifest } from "@/lib/api/analysis";
import { SHARED_WORKSPACE_CACHE_CONTROL } from "@/lib/shared-workspace-cache";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi
    .fn()
    .mockImplementation(async (_request: Request, workspaceId: string) => ({
      ok: true,
      workspaceId,
    })),
}));

vi.mock("../_shared", () => ({
  buildAnalysisManifest: vi.fn(),
}));

import { buildAnalysisManifest } from "../_shared";
import { GET } from "./route";

const manifest = {
  workspaceId: "DEMO",
  createdAt: "2026-05-07T00:00:00.000Z",
  question: "Did escitalopram help?",
  stages: {},
} as unknown as AnalysisManifest;

describe("GET /api/analysis/[workspaceId]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("adds public CDN caching to shared workspace manifests", async () => {
    vi.mocked(buildAnalysisManifest).mockResolvedValue(manifest);

    const response = await GET(new Request("http://localhost/api/analysis/DEMO"), {
      params: Promise.resolve({ workspaceId: "DEMO" }),
    });

    expect(response.status).toBe(200);
    expect(response.headers.get("Cache-Control")).toBe(SHARED_WORKSPACE_CACHE_CONTROL);
  });

  it("keeps non-shared workspace manifests uncached", async () => {
    vi.mocked(buildAnalysisManifest).mockResolvedValue({
      ...manifest,
      workspaceId: "user-123",
    });

    const response = await GET(new Request("http://localhost/api/analysis/user-123"), {
      params: Promise.resolve({ workspaceId: "user-123" }),
    });

    expect(response.status).toBe(200);
    expect(response.headers.get("Cache-Control")).toBeNull();
  });

  it("returns 404 when the manifest cannot be built", async () => {
    vi.mocked(buildAnalysisManifest).mockResolvedValue(null);

    const response = await GET(new Request("http://localhost/api/analysis/user-123"), {
      params: Promise.resolve({ workspaceId: "user-123" }),
    });

    expect(response.status).toBe(404);
  });
});
