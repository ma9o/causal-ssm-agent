import { STAGES, type StageId } from "@nof1-causal-lab/api-types";
import type { EpisodeArtifactStatus } from "@/lib/api/analysis";

/** Stale artifact ids grouped by the stage that produced them. */
export type StaleArtifactsByStage = Partial<Record<StageId, string[]>>;

function isStageId(value: unknown): value is StageId {
  return typeof value === "string" && STAGES.some((stage) => stage.id === value);
}

/**
 * Group the machine's freshness report by producing stage for display.
 *
 * A stage is stale iff any artifact it produced exists and is stale. Root
 * artifacts (null `produced_by`, e.g. the question) belong to no stage and
 * are never stale by construction, so they are skipped. Staleness itself is
 * computed backend-side; this is pure presentation grouping.
 */
export function groupStaleArtifactsByStage(
  artifacts: readonly EpisodeArtifactStatus[],
): StaleArtifactsByStage {
  const byStage: StaleArtifactsByStage = {};
  for (const artifact of artifacts) {
    if (!artifact.exists || !artifact.stale || !isStageId(artifact.produced_by)) {
      continue;
    }
    (byStage[artifact.produced_by] ??= []).push(artifact.artifact_id);
  }
  return byStage;
}

export function hasStaleArtifacts(artifacts: readonly EpisodeArtifactStatus[]): boolean {
  return Object.keys(groupStaleArtifactsByStage(artifacts)).length > 0;
}
