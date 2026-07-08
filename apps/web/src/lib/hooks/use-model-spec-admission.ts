"use client";

import {
  getModelSpecAdmissionStateQueryKey,
  type ModelSpecAdmissionReplayState,
} from "@/lib/model-spec-admission-runtime";
import { useQuery } from "@tanstack/react-query";

export type {
  ModelSpecAdmissionCheckResult,
  ModelSpecAdmissionConstructState,
  ModelSpecAdmissionConstructStatus,
  ModelSpecAdmissionParameter,
  ModelSpecAdmissionPlan,
  ModelSpecAdmissionPlanConstruct,
  ModelSpecAdmissionPlanEdge,
  ModelSpecAdmissionReport,
  ModelSpecAdmissionReplayState,
} from "@/lib/model-spec-admission-runtime";

export function useModelSpecAdmission(workspaceId: string) {
  const { data } = useQuery<ModelSpecAdmissionReplayState>({
    queryKey: getModelSpecAdmissionStateQueryKey(workspaceId),
    queryFn: () => undefined as never,
    enabled: false,
  });

  return data ?? null;
}
