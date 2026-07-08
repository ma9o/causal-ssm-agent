import { TRANSITIONS, type ArtifactViewId } from "@nof1-causal-lab/api-types";
import type { EpisodeArtifactStatus } from "@/lib/api/analysis";

/** Stale artifact ids grouped by the artifact transition or derivation that produced them. */
export type StaleArtifactsByProducer = Partial<Record<ArtifactViewId, string[]>>;

function isArtifactViewId(value: unknown): value is ArtifactViewId {
  return typeof value === "string" && TRANSITIONS.some((transition) => transition.id === value);
}

function producerArtifactId(producedBy: string | null): ArtifactViewId | null {
  if (!producedBy) {
    return null;
  }
  const [kind, artifactId] = producedBy.split(":", 2);
  if ((kind !== "run" && kind !== "derive") || !isArtifactViewId(artifactId)) {
    return null;
  }
  return artifactId;
}

/**
 * Group the machine's freshness report by producing artifact for display.
 *
 * A producer is stale iff any artifact it produced exists and is stale. Root
 * artifacts (null `produced_by`, e.g. the question) belong to no producer and
 * are never stale by construction, so they are skipped. Staleness itself is
 * computed backend-side; this is pure presentation grouping.
 */
export function groupStaleArtifactsByProducer(
  artifacts: readonly EpisodeArtifactStatus[],
): StaleArtifactsByProducer {
  const byProducer: StaleArtifactsByProducer = {};
  for (const artifact of artifacts) {
    const producerId = producerArtifactId(artifact.produced_by);
    if (!artifact.exists || !artifact.stale || producerId === null) {
      continue;
    }
    (byProducer[producerId] ??= []).push(artifact.artifact_id);
  }
  return byProducer;
}

export function hasStaleArtifacts(artifacts: readonly EpisodeArtifactStatus[]): boolean {
  return Object.keys(groupStaleArtifactsByProducer(artifacts)).length > 0;
}
