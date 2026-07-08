import { afterEach, describe, expect, it, vi } from "vitest";
import type { MoveOutcome } from "@/lib/server/episode-runs";

vi.mock("@/lib/server/episode-runs", () => ({
  EpisodeRunError: class EpisodeRunError extends Error {
    status: number;

    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  WRITABLE_ARTIFACTS: {
    latent_structure: "latent_structure",
    measurement_structure: "measurement_structure",
    baseline_report: "baseline_report",
  },
  proposeMove: vi.fn(),
  startAutoRun: vi.fn(),
}));

import { EpisodeRunError, proposeMove, startAutoRun } from "@/lib/server/episode-runs";
import { POST } from "./route";

function appliedOutcome(): MoveOutcome {
  return {
    seq: 7,
    status: "applied",
    reason: null,
    error_type: null,
    error_message: null,
    diagnostics: {},
    produced: [],
    retracted: [],
    state: { current: {} },
  };
}

function makeRequest(body: unknown): Request {
  return new Request("http://localhost/api/replay", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("POST /api/replay", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("rejects requests without required fields", async () => {
    const response = await POST(makeRequest({ workspaceId: "user-1" }));

    expect(response.status).toBe(400);
  });

  it("rejects non-writable artifacts", async () => {
    const response = await POST(
      makeRequest({
        workspaceId: "user-1",
        artifactId: "validation_report",
        payload: { is_valid: true },
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "Artifact validation_report is not writable",
    });
    expect(proposeMove).not.toHaveBeenCalled();
  });

  it("writes the artifact with human provenance and starts auto-run", async () => {
    vi.mocked(proposeMove).mockResolvedValue(appliedOutcome());
    vi.mocked(startAutoRun).mockResolvedValue();

    const payload = { latent_structure: { constructs: [] } };
    const response = await POST(
      makeRequest({ workspaceId: "user-1", artifactId: "latent_structure", payload }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true, workspaceId: "user-1" });
    expect(proposeMove).toHaveBeenCalledWith(
      "user-1",
      { kind: "write", artifact_id: "latent_structure", provenance: "human" },
      payload,
    );
    expect(startAutoRun).toHaveBeenCalledWith("user-1");
  });

  it("writes requested writable artifacts directly", async () => {
    vi.mocked(proposeMove).mockResolvedValue(appliedOutcome());
    vi.mocked(startAutoRun).mockResolvedValue();

    await POST(
      makeRequest({
        workspaceId: "user-1",
        artifactId: "measurement_structure",
        payload: { a: 1 },
      }),
    );
    await POST(
      makeRequest({ workspaceId: "user-1", artifactId: "baseline_report", payload: { b: 2 } }),
    );

    expect(proposeMove).toHaveBeenNthCalledWith(
      1,
      "user-1",
      { kind: "write", artifact_id: "measurement_structure", provenance: "human" },
      { a: 1 },
    );
    expect(proposeMove).toHaveBeenNthCalledWith(
      2,
      "user-1",
      { kind: "write", artifact_id: "baseline_report", provenance: "human" },
      { b: 2 },
    );
  });

  it("surfaces rejected write moves as 400", async () => {
    vi.mocked(proposeMove).mockResolvedValue({
      ...appliedOutcome(),
      status: "rejected",
      reason: "write moves must declare provenance 'human' or 'llm'",
    });

    const response = await POST(
      makeRequest({ workspaceId: "user-1", artifactId: "latent_structure", payload: { a: 1 } }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "write moves must declare provenance 'human' or 'llm'",
    });
    expect(startAutoRun).not.toHaveBeenCalled();
  });

  it("surfaces raised write moves as 502", async () => {
    vi.mocked(proposeMove).mockResolvedValue({
      ...appliedOutcome(),
      status: "raised",
      error_type: "SchemaValidationError",
      error_message: "latent_structure payload failed validation",
    });

    const response = await POST(
      makeRequest({ workspaceId: "user-1", artifactId: "latent_structure", payload: { a: 1 } }),
    );

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      error: "latent_structure payload failed validation",
    });
  });

  it("returns 409 when an auto-run is already active", async () => {
    vi.mocked(proposeMove).mockResolvedValue(appliedOutcome());
    vi.mocked(startAutoRun).mockRejectedValue(
      new EpisodeRunError(409, "auto-run already active for user-1"),
    );

    const response = await POST(
      makeRequest({ workspaceId: "user-1", artifactId: "latent_structure", payload: { a: 1 } }),
    );

    expect(response.status).toBe(409);
  });
});
