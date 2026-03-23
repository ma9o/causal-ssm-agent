import { NextResponse } from "next/server";
import type { SessionResponse } from "@/lib/api/analysis";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { readQuestion, readSession } from "../_shared";

export async function GET(request: Request, { params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = await params;
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: normalizedWorkspaceId } = workspaceAccess;

  try {
    const session = await readSession(normalizedWorkspaceId);
    if (session) {
      const question = await readQuestion(normalizedWorkspaceId);
      const response: SessionResponse = { ...session, question };
      return NextResponse.json(response);
    }
  } catch {
    // Fall through
  }

  return NextResponse.json({ error: "Session not found" }, { status: 404 });
}
