import type { ArtifactViewId, LLMTrace, UploadResponse } from "@nof1-causal-lab/api-types";
import { apiFetch } from "./client";

export async function uploadFile(file: File, workspaceId: string): Promise<UploadResponse> {
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

export async function getArtifactView<T>(
  workspaceId: string,
  artifactId: ArtifactViewId,
): Promise<T> {
  return apiFetch<T>(`/api/artifacts/${workspaceId}/${artifactId}/view`);
}

export async function getLLMTrace(workspaceId: string, ref: string): Promise<LLMTrace> {
  const search = new URLSearchParams({ ref }).toString();
  return apiFetch<LLMTrace>(`/api/traces/${workspaceId}?${search}`);
}
