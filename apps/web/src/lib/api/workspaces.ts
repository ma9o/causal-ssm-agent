import { apiFetch } from "@/lib/api/client";
import type { WorkspaceList } from "@/lib/server/workspaces";

export function getWorkspacesQueryKey() {
  return ["workspaces"] as const;
}

export async function getWorkspaces(): Promise<WorkspaceList> {
  return apiFetch<WorkspaceList>("/api/workspaces", {
    cache: "no-store",
  });
}
