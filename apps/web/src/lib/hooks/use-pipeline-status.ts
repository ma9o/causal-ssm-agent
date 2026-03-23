"use client";

import { useQuery } from "@tanstack/react-query";
import type { PipelineProgress } from "./use-run-events";

export function usePipelineStatus(workspaceId: string | null): PipelineProgress | undefined {
  const { data } = useQuery<PipelineProgress>({
    queryKey: ["pipeline", workspaceId, "status"],
    queryFn: () => undefined as never,
    enabled: false,
  });
  return data;
}
