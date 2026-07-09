import { getToolServerUrl } from "@/lib/runtime-urls";
export type { WorkspaceEntry, WorkspaceList } from "@nof1-causal-lab/api-types";
import type { WorkspaceList } from "@nof1-causal-lab/api-types";

const TOOL_SERVER = getToolServerUrl();

export async function listWorkspaces(): Promise<WorkspaceList> {
  const response = await fetch(`${TOOL_SERVER}/api/workspaces`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Workspace facade error ${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<WorkspaceList>;
}
