import { afterEach, describe, expect, it, vi } from "vitest";
import type { EpisodeStatus, TransitionRecord } from "@/lib/server/episode-runs";

vi.mock("@/lib/server/episode-runs", () => ({
  getEpisodeStatus: vi.fn(),
  getEpisodeTimeline: vi.fn(),
}));

vi.mock("@/lib/storage", () => ({
  isStorageNotFoundError: (e: unknown) => e instanceof Error && e.message.startsWith("not found"),
  readData: vi.fn(),
}));

import { getEpisodeStatus, getEpisodeTimeline } from "@/lib/server/episode-runs";
import { readData } from "@/lib/storage";
import { buildAnalysisManifest } from "./_shared";

function emptyStatus(workspaceId: string): EpisodeStatus {
  return {
    workspace_id: workspaceId,
    seq: 0,
    state: { current: {} },
    artifacts: [],
    legal: [],
    auto_running: false,
  };
}

function statusWithQuestion(workspaceId: string, version = 1): EpisodeStatus {
  return {
    ...emptyStatus(workspaceId),
    state: {
      current: {
        question: {
          artifact_id: "question",
          version,
          provenance: "human",
          derived_from: {},
          produced_by: null,
          created_at: "2026-07-01T00:00:00+00:00",
        },
      },
    },
  };
}

function transition(
  overrides: Partial<TransitionRecord> & Pick<TransitionRecord, "seq" | "ts" | "move" | "status">,
): TransitionRecord {
  return {
    reason: null,
    error_type: null,
    error_message: null,
    diagnostics: {},
    produced: [],
    retracted: [],
    state_after: { current: {} },
    ...overrides,
  };
}

describe("buildAnalysisManifest", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("returns null when the episode journal is empty", async () => {
    vi.mocked(getEpisodeStatus).mockResolvedValue(emptyStatus("user-1"));
    vi.mocked(getEpisodeTimeline).mockResolvedValue({
      workspace_id: "user-1",
      transitions: [],
    });

    await expect(buildAnalysisManifest("user-1")).resolves.toBeNull();
  });

  it("builds stage executions from journal run transitions", async () => {
    vi.mocked(getEpisodeStatus).mockResolvedValue(statusWithQuestion("user-1"));
    vi.mocked(readData).mockResolvedValue(JSON.stringify({ text: "Does exercise help sleep?" }));
    vi.mocked(getEpisodeTimeline).mockResolvedValue({
      workspace_id: "user-1",
      transitions: [
        transition({
          seq: 1,
          ts: "2026-07-01T00:00:00+00:00",
          move: { kind: "write", artifact_id: "question", provenance: "human" },
          status: "applied",
        }),
        transition({
          seq: 2,
          ts: "2026-07-01T00:01:00+00:00",
          move: { kind: "run", stage_id: "stage-0" },
          status: "applied",
        }),
        transition({
          seq: 3,
          ts: "2026-07-01T00:02:00+00:00",
          move: { kind: "run", stage_id: "stage-1a" },
          status: "raised",
          error_type: "SchemaValidationError",
          error_message: "latent_structure payload failed validation",
        }),
        transition({
          seq: 4,
          ts: "2026-07-01T00:03:00+00:00",
          move: { kind: "run", stage_id: "stage-1a" },
          status: "rejected",
          reason: "stage-1a requires artifacts that do not exist: raw_data",
        }),
      ],
    });

    const manifest = await buildAnalysisManifest("user-1");

    expect(manifest).not.toBeNull();
    expect(manifest?.createdAt).toBe("2026-07-01T00:00:00+00:00");
    expect(manifest?.question).toBe("Does exercise help sleep?");
    expect(vi.mocked(readData)).toHaveBeenCalledWith("user-1/store/question/v1/question.json");
    expect(manifest?.stages["stage-0"]).toEqual({
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-07-01T00:01:00+00:00",
        endTime: "2026-07-01T00:01:00+00:00",
      },
    });
    // Raised run marks the stage failed; the later rejected attempt never executed.
    expect(manifest?.stages["stage-1a"]?.execution?.stateType).toBe("FAILED");
    expect(manifest?.stages["stage-2"]).toEqual({ execution: null });
  });

  it("prefers the latest run attempt per stage", async () => {
    vi.mocked(getEpisodeStatus).mockResolvedValue(emptyStatus("user-1"));
    vi.mocked(getEpisodeTimeline).mockResolvedValue({
      workspace_id: "user-1",
      transitions: [
        transition({
          seq: 1,
          ts: "2026-07-01T00:00:00+00:00",
          move: { kind: "run", stage_id: "stage-0" },
          status: "raised",
          error_type: "RuntimeError",
        }),
        transition({
          seq: 2,
          ts: "2026-07-01T00:05:00+00:00",
          move: { kind: "run", stage_id: "stage-0" },
          status: "applied",
        }),
      ],
    });

    const manifest = await buildAnalysisManifest("user-1");

    expect(manifest?.stages["stage-0"]?.execution?.stateType).toBe("COMPLETED");
  });
});
