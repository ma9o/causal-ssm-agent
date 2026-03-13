"use client";

import { useQuery } from "@tanstack/react-query";
import type { PipelineProgress } from "./use-run-events";

export function usePipelineStatus(userId: string | null): PipelineProgress | undefined {
  const { data } = useQuery<PipelineProgress>({
    queryKey: ["pipeline", userId, "status"],
    queryFn: () => undefined as never,
    enabled: false,
  });
  return data;
}
