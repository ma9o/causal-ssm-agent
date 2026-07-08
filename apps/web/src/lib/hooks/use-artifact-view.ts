"use client";

import type { ArtifactViewId } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";
import { getArtifactView } from "../api/endpoints";
import { isMockMode } from "../api/mock-provider";

const ARTIFACT_VIEW_QUERY_VERSION = 3;

async function fetchArtifactView<T>(workspaceId: string, artifactId: ArtifactViewId): Promise<T> {
  let payload: unknown;

  if (isMockMode()) {
    const res = await fetch(`/api/artifacts/${workspaceId}/${artifactId}/view`);
    if (!res.ok) throw new Error(`Mock data not found for ${artifactId}`);
    payload = await res.json();
  } else {
    payload = await getArtifactView<unknown>(workspaceId, artifactId);
  }

  return payload as T;
}

export function getArtifactViewQueryKey(workspaceId: string | null, artifactId: ArtifactViewId) {
  return [
    "pipeline",
    workspaceId,
    "artifact",
    artifactId,
    `v${ARTIFACT_VIEW_QUERY_VERSION}`,
  ] as const;
}

export function useArtifactView<T>(
  workspaceId: string | null,
  artifactId: ArtifactViewId,
  enabled: boolean,
) {
  return useQuery<T>({
    queryKey: getArtifactViewQueryKey(workspaceId, artifactId),
    queryFn: () => fetchArtifactView<T>(workspaceId as string, artifactId),
    enabled: !!workspaceId && enabled,
    staleTime: Number.POSITIVE_INFINITY,
  });
}
