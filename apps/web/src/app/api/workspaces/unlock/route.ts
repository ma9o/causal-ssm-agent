import { NextResponse } from "next/server";
import {
  requireWorkspaceAccess,
  setWorkspaceAccessCookie,
} from "@/lib/workspace-access";

export async function POST(request: Request) {
  const body = await request.json();
  const { workspaceId, accessCode } = body as {
    workspaceId?: string;
    accessCode?: string;
  };

  if (typeof accessCode !== "string" || accessCode.trim().length === 0) {
    return NextResponse.json({ error: "accessCode is required" }, { status: 400 });
  }

  const workspaceAccess = await requireWorkspaceAccess(
    request,
    workspaceId,
    { accessCode: accessCode.trim(), allowCreate: false },
  );
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: normalizedWorkspaceId, setCookieCode } = workspaceAccess;

  const response = NextResponse.json({ ok: true });
  if (setCookieCode) {
    setWorkspaceAccessCookie(response, normalizedWorkspaceId, setCookieCode);
  }
  return response;
}
