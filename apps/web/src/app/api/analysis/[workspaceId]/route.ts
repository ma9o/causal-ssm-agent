import { NextResponse } from "next/server";
import { SHARED_WORKSPACE_CACHE_CONTROL } from "@/lib/shared-workspace-cache";
import { isSharedWorkspaceId } from "@/lib/shared-workspaces";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { buildAnalysisManifest } from "../_shared";

export async function GET(request: Request, { params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = await params;
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: normalizedWorkspaceId } = workspaceAccess;

  try {
    const url = new URL(request.url);
    const bootstrapRootFlowRunIds = url.searchParams.getAll("rootFlowRunId").filter(Boolean);
    const manifest = await buildAnalysisManifest(normalizedWorkspaceId, bootstrapRootFlowRunIds);
    if (manifest) {
      const response = NextResponse.json(manifest);
      if (isSharedWorkspaceId(normalizedWorkspaceId) && bootstrapRootFlowRunIds.length === 0) {
        response.headers.set("Cache-Control", SHARED_WORKSPACE_CACHE_CONTROL);
      }
      return response;
    }
  } catch {
    // Fall through
  }

  return NextResponse.json({ error: "Analysis manifest not found" }, { status: 404 });
}
