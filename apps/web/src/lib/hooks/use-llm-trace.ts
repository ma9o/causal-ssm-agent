"use client";

import { useQuery } from "@tanstack/react-query";
import { getLLMTrace } from "../api/endpoints";

const LLM_TRACE_QUERY_VERSION = 2;

export function getLLMTraceQueryKey(workspaceId: string | null, artifactId: string | null) {
  return ["pipeline", workspaceId, "llm-trace", artifactId, `v${LLM_TRACE_QUERY_VERSION}`] as const;
}

/** Merged LLM trace of the applied transition that produced an artifact's current version. */
export function useLLMTrace(
  workspaceId: string | null,
  artifactId: string | null,
  enabled: boolean,
) {
  return useQuery({
    queryKey: getLLMTraceQueryKey(workspaceId, artifactId),
    queryFn: () => getLLMTrace(workspaceId as string, artifactId as string),
    enabled: !!workspaceId && !!artifactId && enabled,
    staleTime: Number.POSITIVE_INFINITY,
    // A 404 means the producing transition promoted no traces — not transient.
    retry: false,
  });
}
