"use client";

import {
  getStage4AdmissionStateQueryKey,
  type Stage4AdmissionReplayState,
} from "@/lib/stage4-admission-runtime";
import { useQuery } from "@tanstack/react-query";

export type {
  Stage4AdmissionCheckResult,
  Stage4AdmissionConstructState,
  Stage4AdmissionConstructStatus,
  Stage4AdmissionParameter,
  Stage4AdmissionPlan,
  Stage4AdmissionPlanConstruct,
  Stage4AdmissionPlanEdge,
  Stage4AdmissionReport,
  Stage4AdmissionReplayState,
} from "@/lib/stage4-admission-runtime";

export function useStage4Admission(workspaceId: string) {
  const { data } = useQuery<Stage4AdmissionReplayState>({
    queryKey: getStage4AdmissionStateQueryKey(workspaceId),
    queryFn: () => undefined as never,
    enabled: false,
  });

  return data ?? null;
}
