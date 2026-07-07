import { describe, expect, it } from "vitest";
import type { EpisodeArtifactStatus } from "@/lib/api/analysis";
import { groupStaleArtifactsByStage, hasStaleArtifacts } from "./artifact-staleness";

function artifact(overrides: Partial<EpisodeArtifactStatus>): EpisodeArtifactStatus {
  return {
    artifact_id: "latent_structure",
    exists: true,
    stale: false,
    version: 1,
    provenance: "computed",
    produced_by: "stage-1a",
    ...overrides,
  };
}

describe("groupStaleArtifactsByStage", () => {
  it("groups stale existing artifacts by producing stage", () => {
    const report = [
      artifact({ artifact_id: "latent_structure", stale: true, produced_by: "stage-1a" }),
      artifact({
        artifact_id: "measurement_structure",
        stale: true,
        produced_by: "stage-1b",
      }),
      artifact({ artifact_id: "panel", stale: false, produced_by: "stage-2" }),
    ];

    expect(groupStaleArtifactsByStage(report)).toEqual({
      "stage-1a": ["latent_structure"],
      "stage-1b": ["measurement_structure"],
    });
  });

  it("ignores absent artifacts even when flagged stale", () => {
    const report = [
      artifact({
        artifact_id: "identification_report",
        exists: false,
        stale: true,
        produced_by: "stage-1b",
      }),
    ];

    expect(groupStaleArtifactsByStage(report)).toEqual({});
  });

  it("ignores root artifacts with no producing stage", () => {
    const report = [
      artifact({ artifact_id: "question", stale: true, produced_by: null, provenance: "human" }),
    ];

    expect(groupStaleArtifactsByStage(report)).toEqual({});
  });
});

describe("hasStaleArtifacts", () => {
  it("is true iff any stage-produced artifact is stale", () => {
    expect(hasStaleArtifacts([artifact({ stale: false })])).toBe(false);
    expect(hasStaleArtifacts([artifact({ stale: true })])).toBe(true);
  });
});
