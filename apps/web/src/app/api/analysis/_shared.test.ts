import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EpisodeStatus, TransitionRecord } from "@/lib/server/episode-runs";

vi.mock("@/lib/server/episode-runs", () => ({
  getMachineDescription: vi.fn(),
  getEpisodeStatus: vi.fn(),
  getEpisodeTimeline: vi.fn(),
}));

vi.mock("@/lib/server/artifacts", () => ({
  ArtifactNotFoundError: class ArtifactNotFoundError extends Error {},
  readArtifactJson: vi.fn(),
}));

import {
  getEpisodeStatus,
  getEpisodeTimeline,
  getMachineDescription,
} from "@/lib/server/episode-runs";
import { readArtifactJson } from "@/lib/server/artifacts";
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
    ...overrides,
  };
}

describe("buildAnalysisManifest", () => {
  beforeEach(() => {
    vi.mocked(getMachineDescription).mockResolvedValue({
      topological_artifact_order: [
        "question",
        "raw_data",
        "latent_structure",
        "measurement_structure",
        "causal_design",
        "identification_report",
        "measurements",
        "panel",
        "validation_report",
        "statistical_model_spec",
        "compiled_ssm",
        "posterior",
        "baseline_report",
        "saved_scenarios",
      ],
      topological_transition_order: [
        "raw_data",
        "latent_structure",
        "measurement_structure",
        "measurements",
        "statistical_model_spec",
        "posterior",
        "baseline_report",
      ],
      transitions: [
        { transition_id: "raw_data" },
        { transition_id: "latent_structure" },
        { transition_id: "measurement_structure" },
        { transition_id: "measurements" },
        { transition_id: "statistical_model_spec" },
        { transition_id: "posterior" },
        { transition_id: "baseline_report" },
      ],
    });
  });

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

  it("builds transition executions from journal run transitions", async () => {
    vi.mocked(getEpisodeStatus).mockResolvedValue(statusWithQuestion("user-1"));
    vi.mocked(readArtifactJson).mockResolvedValue({ text: "Does exercise help sleep?" });
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
          move: { kind: "run", artifact_id: "raw_data" },
          status: "applied",
        }),
        transition({
          seq: 3,
          ts: "2026-07-01T00:02:00+00:00",
          move: { kind: "run", artifact_id: "latent_structure" },
          status: "raised",
          error_type: "SchemaValidationError",
          error_message: "latent_structure payload failed validation",
        }),
        transition({
          seq: 4,
          ts: "2026-07-01T00:03:00+00:00",
          move: { kind: "run", artifact_id: "latent_structure" },
          status: "rejected",
          reason: "latent_structure requires artifacts that do not exist: raw_data",
        }),
      ],
    });

    const manifest = await buildAnalysisManifest("user-1");

    expect(manifest).not.toBeNull();
    expect(manifest?.createdAt).toBe("2026-07-01T00:00:00+00:00");
    expect(manifest?.question).toBe("Does exercise help sleep?");
    expect(manifest?.transitionOrder).toEqual([
      "raw_data",
      "latent_structure",
      "measurement_structure",
      "measurements",
      "validation_report",
      "statistical_model_spec",
      "posterior",
      "baseline_report",
    ]);
    expect(vi.mocked(readArtifactJson)).toHaveBeenCalledWith("user-1", "question", "question");
    expect(manifest?.transitionRuns["raw_data"]).toEqual({
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-07-01T00:01:00+00:00",
        endTime: "2026-07-01T00:01:00+00:00",
      },
    });
    // Raised run marks the transition failed; the later rejected attempt never executed.
    expect(manifest?.transitionRuns["latent_structure"]?.execution?.stateType).toBe("FAILED");
    expect(manifest?.transitionRuns["measurements"]).toEqual({ execution: null });
  });

  it("prefers the latest run attempt per artifact", async () => {
    vi.mocked(getEpisodeStatus).mockResolvedValue(emptyStatus("user-1"));
    vi.mocked(getEpisodeTimeline).mockResolvedValue({
      workspace_id: "user-1",
      transitions: [
        transition({
          seq: 1,
          ts: "2026-07-01T00:00:00+00:00",
          move: { kind: "run", artifact_id: "raw_data" },
          status: "raised",
          error_type: "RuntimeError",
        }),
        transition({
          seq: 2,
          ts: "2026-07-01T00:05:00+00:00",
          move: { kind: "run", artifact_id: "raw_data" },
          status: "applied",
        }),
      ],
    });

    const manifest = await buildAnalysisManifest("user-1");

    expect(manifest?.transitionRuns["raw_data"]?.execution?.stateType).toBe("COMPLETED");
  });
});
