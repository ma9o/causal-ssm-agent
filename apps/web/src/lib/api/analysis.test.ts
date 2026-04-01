import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  applyRefinement,
  getAnalysisManifest,
  getStage4ReplayState,
  replayStageOverride,
} from "./analysis";

describe("analysis api helpers", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllGlobals();
  });

  it("fetches the analysis manifest without a root flow id", async () => {
    const payload = {
      workspaceId: "user-1",
      rootFlowRunIds: [],
      latestRootFlowRunId: null,
      stages: {},
      createdAt: "2026-01-01T00:00:00Z",
    };
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(payload),
    } as Response);

    const result = await getAnalysisManifest("user-1");

    expect(result).toEqual(payload);
    expect(fetch).toHaveBeenCalledWith(
      "/api/analysis/user-1",
      expect.objectContaining({
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
      }),
    );
  });

  it("fetches the Stage 4 replay state for a specific root flow run", async () => {
    const payload = { sections: {}, blockStatuses: {} };
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(payload),
    } as Response);

    await getStage4ReplayState("user-1", "root-123");

    expect(fetch).toHaveBeenCalledWith(
      "/api/analysis/user-1/stage4-state?rootFlowRunId=root-123",
      expect.any(Object),
    );
  });

  it("posts a replay stage override to the replay api", async () => {
    const payload = { ok: true, resumeFrom: "stage-2", rootFlowRunId: "replay-1" };
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(payload),
    } as Response);

    const result = await replayStageOverride({
      workspaceId: "user-1",
      stageId: "stage-1b",
      stageData: { causal_spec: { measurement: { indicators: [] } } },
      rootFlowRunId: "root-123",
    });

    expect(result).toEqual(payload);
    expect(fetch).toHaveBeenCalledWith(
      "/api/replay",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          workspaceId: "user-1",
          stageId: "stage-1b",
          stageData: { causal_spec: { measurement: { indicators: [] } } },
          rootFlowRunId: "root-123",
        }),
      }),
    );
  });

  it("posts a materialized refinement to the refine apply api", async () => {
    const payload = {
      ok: true,
      updatedFields: ["latent_model"],
      resumeFrom: "stage-1b",
      rootFlowRunId: "replay-2",
    };
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(payload),
    } as Response);

    const result = await applyRefinement({
      workspaceId: "user-1",
      stageId: "stage-1a",
      stagePatch: { latent_model: { constructs: [], edges: [] } },
      messages: [],
      rootFlowRunId: "root-456",
    });

    expect(result).toEqual(payload);
    expect(fetch).toHaveBeenCalledWith(
      "/api/refine/apply",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          workspaceId: "user-1",
          stageId: "stage-1a",
          stagePatch: { latent_model: { constructs: [], edges: [] } },
          messages: [],
          rootFlowRunId: "root-456",
        }),
      }),
    );
  });
});
