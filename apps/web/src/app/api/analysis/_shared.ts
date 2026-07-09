import type {
  AnalysisManifest,
  AnalysisTransitionRun,
  AnalysisTransitionRuns,
} from "@/lib/api/analysis";
import {
  getMachineDescription,
  getEpisodeStatus,
  getEpisodeTimeline,
  type EpisodeStatus,
  type MachineDescription,
  type TransitionRecord,
} from "@/lib/server/episode-runs";
import { ArtifactNotFoundError, readArtifactJson } from "@/lib/server/artifacts";
import { TRANSITIONS, type ArtifactViewId } from "@nof1-causal-lab/api-types";

function emptyTransitionRun(): AnalysisTransitionRun {
  return { execution: null };
}

function isArtifactViewId(value: unknown): value is ArtifactViewId {
  return typeof value === "string" && TRANSITIONS.some((transition) => transition.id === value);
}

function artifactViewOrder(machine: MachineDescription): ArtifactViewId[] {
  const ordered = machine.topological_artifact_order.filter(isArtifactViewId);
  const missing = TRANSITIONS.filter((transition) => !ordered.includes(transition.id)).map(
    (transition) => transition.id,
  );
  if (missing.length > 0) {
    throw new Error(
      `Machine description omits artifact views from topological order: ${missing.join(", ")}`,
    );
  }
  return ordered;
}

async function readEpisodeQuestion(
  workspaceId: string,
  status: EpisodeStatus,
): Promise<string | undefined> {
  if (status.state.current.question == null) {
    return undefined;
  }

  try {
    const parsed = await readArtifactJson<{ text?: unknown }>(workspaceId, "question", "question");
    return typeof parsed.text === "string" && parsed.text.trim() ? parsed.text.trim() : undefined;
  } catch (e: unknown) {
    if (e instanceof ArtifactNotFoundError) {
      return undefined;
    }
    throw e;
  }
}

/**
 * Per-artifact execution summaries from the episode journal: the latest
 * completed run attempt per artifact wins (applied -> completed, raised ->
 * failed; rejected attempts never executed).
 */
function summarizeTimelineTransitionRuns(transitions: TransitionRecord[]): AnalysisTransitionRuns {
  const transitionRuns = Object.fromEntries(
    TRANSITIONS.map((transition) => [transition.id, emptyTransitionRun()]),
  ) as AnalysisTransitionRuns;

  for (const record of transitions) {
    if (record.move.kind !== "run") {
      continue;
    }
    const artifactId = record.move.artifact_id;
    if (!isArtifactViewId(artifactId)) {
      continue;
    }
    if (record.status === "rejected") {
      continue;
    }

    transitionRuns[artifactId] = {
      execution: {
        stateType: record.status === "applied" ? "COMPLETED" : "FAILED",
        startTime: record.ts,
        endTime: record.ts,
      },
    };
  }

  return transitionRuns;
}

/**
 * The manifest comes straight from the episode journal — a published
 * (read-only) workspace carries its journal along, so there is no separate
 * curated-demo path.
 */
export async function buildAnalysisManifest(
  workspaceId: string,
): Promise<Omit<AnalysisManifest, "readOnly"> | null> {
  const [status, timeline, machine] = await Promise.all([
    getEpisodeStatus(workspaceId),
    getEpisodeTimeline(workspaceId),
    getMachineDescription(),
  ]);
  if (timeline.transitions.length === 0) {
    return null;
  }

  const question = await readEpisodeQuestion(workspaceId, status);

  return {
    workspaceId,
    createdAt: timeline.transitions[0].ts,
    question,
    transitionOrder: artifactViewOrder(machine),
    transitionRuns: summarizeTimelineTransitionRuns(timeline.transitions),
  };
}
