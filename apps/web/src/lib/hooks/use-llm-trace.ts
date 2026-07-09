"use client";

import { useQuery } from "@tanstack/react-query";
import { getLLMTrace } from "../api/endpoints";

const LLM_TRACE_QUERY_VERSION = 1;

export function getLLMTraceQueryKey(workspaceId: string | null, ref: string | null) {
  return ["pipeline", workspaceId, "llm-trace", ref, `v${LLM_TRACE_QUERY_VERSION}`] as const;
}

export function useLLMTrace(workspaceId: string | null, ref: string | null, enabled: boolean) {
  return useQuery({
    queryKey: getLLMTraceQueryKey(workspaceId, ref),
    queryFn: () => getLLMTrace(workspaceId as string, ref as string),
    enabled: !!workspaceId && !!ref && enabled,
    staleTime: Number.POSITIVE_INFINITY,
  });
}
