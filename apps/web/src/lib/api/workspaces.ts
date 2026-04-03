import { apiFetch } from "@/lib/api/client";
import type { AccessibleWorkspaceList } from "@/lib/server/workspace-ownership";

export function getAccessibleWorkspacesQueryKey(authScope: string) {
  return ["accessible-workspaces", authScope] as const;
}

export async function getAccessibleWorkspaces(): Promise<AccessibleWorkspaceList> {
  return apiFetch<AccessibleWorkspaceList>("/api/workspaces", {
    cache: "no-store",
  });
}
