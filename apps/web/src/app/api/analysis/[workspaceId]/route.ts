import { NextResponse } from "next/server";
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
      return NextResponse.json(manifest);
    }
  } catch {
    // Fall through
  }

  return NextResponse.json({ error: "Analysis manifest not found" }, { status: 404 });
}
