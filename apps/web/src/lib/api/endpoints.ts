import type { StageId } from "@nof1-causal-lab/api-types";
import { apiFetch } from "./client";

export async function uploadFile(
  file: File,
  workspaceId: string,
): Promise<{ path: string }> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("workspaceId", workspaceId);
  const res = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function getStageResult<T>(workspaceId: string, stage: StageId): Promise<T> {
  return apiFetch<T>(`/api/results/${workspaceId}/${stage}`);
}
