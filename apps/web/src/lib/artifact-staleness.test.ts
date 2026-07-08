import { describe, expect, it } from "vitest";
import type { EpisodeArtifactStatus } from "@/lib/api/analysis";
import { groupStaleArtifactsByProducer, hasStaleArtifacts } from "./artifact-staleness";

function artifact(overrides: Partial<EpisodeArtifactStatus>): EpisodeArtifactStatus {
  return {
    artifact_id: "latent_structure",
    exists: true,
    stale: false,
    version: 1,
    provenance: "computed",
    produced_by: "run:latent_structure",
    ...overrides,
  };
}

describe("groupStaleArtifactsByProducer", () => {
  it("groups stale existing artifacts by producing artifact", () => {
    const report = [
      artifact({
        artifact_id: "latent_structure",
        stale: true,
        produced_by: "run:latent_structure",
      }),
      artifact({
        artifact_id: "measurement_structure",
        stale: true,
        produced_by: "run:measurement_structure",
      }),
      artifact({ artifact_id: "panel", stale: false, produced_by: "run:measurements" }),
    ];

    expect(groupStaleArtifactsByProducer(report)).toEqual({
      latent_structure: ["latent_structure"],
      measurement_structure: ["measurement_structure"],
    });
  });

  it("ignores absent artifacts even when flagged stale", () => {
    const report = [
      artifact({
        artifact_id: "identification_report",
        exists: false,
        stale: true,
        produced_by: "run:measurement_structure",
      }),
    ];

    expect(groupStaleArtifactsByProducer(report)).toEqual({});
  });

  it("ignores root artifacts with no producer", () => {
    const report = [
      artifact({ artifact_id: "question", stale: true, produced_by: null, provenance: "human" }),
    ];

    expect(groupStaleArtifactsByProducer(report)).toEqual({});
  });
});

describe("hasStaleArtifacts", () => {
  it("is true iff any produced artifact is stale", () => {
    expect(hasStaleArtifacts([artifact({ stale: false })])).toBe(false);
    expect(hasStaleArtifacts([artifact({ stale: true })])).toBe(true);
  });
});
