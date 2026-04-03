import { apiFetch } from "@/lib/api/client";
import type { AccessibleWorkspaceList } from "@/lib/server/workspace-ownership";

export function getAccessibleWorkspacesQueryKey() {
  return ["accessible-workspaces"] as const;
}

export async function getAccessibleWorkspaces(): Promise<AccessibleWorkspaceList> {
  return apiFetch<AccessibleWorkspaceList>("/api/workspaces", {
    cache: "no-store",
  });
}
