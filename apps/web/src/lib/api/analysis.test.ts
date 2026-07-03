import { afterEach, describe, expect, it, vi } from "vitest";
import {
  applyRefinement,
  getAnalysisManifest,
  getEpisodeProgress,
  replayStageOverride,
} from "./analysis";

const originalFetch = globalThis.fetch;

function mockFetchJson(data: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    text: async () => JSON.stringify(data),
    json: async () => data,
  } as unknown as Response);
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe("analysis api client", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("fetches the analysis manifest", async () => {
    const fetchMock = mockFetchJson({ workspaceId: "user-1" });

    await getAnalysisManifest("user-1");

    expect(fetchMock).toHaveBeenCalledWith("/api/analysis/user-1", expect.anything());
  });

  it("fetches episode progress without a cursor", async () => {
    const fetchMock = mockFetchJson({ events: [], transitions: [] });

    await getEpisodeProgress("user-1");

    expect(fetchMock).toHaveBeenCalledWith("/api/analysis/user-1/progress", expect.anything());
  });

  it("fetches episode progress after a cursor", async () => {
    const fetchMock = mockFetchJson({ events: [], transitions: [] });

    await getEpisodeProgress("user-1", "00000000000000000123-abc.json");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/analysis/user-1/progress?after=00000000000000000123-abc.json",
      expect.anything(),
    );
  });

  it("posts stage overrides to the replay route", async () => {
    const fetchMock = mockFetchJson({ ok: true, workspaceId: "user-1" });

    await replayStageOverride({
      workspaceId: "user-1",
      stageId: "stage-1a",
      stageData: { latent_model: {} },
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/replay",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          workspaceId: "user-1",
          stageId: "stage-1a",
          stageData: { latent_model: {} },
        }),
      }),
    );
  });

  it("posts refinements to the apply route", async () => {
    const fetchMock = mockFetchJson({ ok: true, updatedFields: [] });

    await applyRefinement({
      workspaceId: "user-1",
      stageId: "stage-6",
      stagePatch: { final_summary: "Updated." },
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/refine/apply",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
