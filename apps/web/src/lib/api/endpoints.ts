import type { StageId } from "@causal-ssm/api-types";
import { apiFetch } from "./client";

export async function uploadFile(file: File, userId: string): Promise<{ path: string }> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("userId", userId);
  const res = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function getStageResult<T>(runId: string, stage: StageId): Promise<T> {
  return apiFetch<T>(`/api/results/${runId}/${stage}`);
}
