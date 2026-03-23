import { NextResponse } from "next/server";
import { writeData, ensureDir } from "@/lib/storage";
import {
  requireWorkspaceAccess,
  setWorkspaceAccessCookie,
} from "@/lib/workspace-access";
import {
  appendSessionRootFlowRunId,
  normalizeSession,
  readSession,
  writeSession,
} from "./_shared";

const MAX_WORKSPACE_ID_LENGTH = 200;
const MAX_QUESTION_LENGTH = 2000;

export async function POST(request: Request) {
  const body = await request.json();
  const { workspaceId, accessCode, question, rootFlowRunId } = body as {
    workspaceId?: string;
    accessCode?: string;
    question?: string;
    rootFlowRunId?: string;
  };

  if (!workspaceId || !question || !accessCode) {
    return NextResponse.json(
      { error: "workspaceId, accessCode, and question are required" },
      { status: 400 },
    );
  }

  if (workspaceId.length > MAX_WORKSPACE_ID_LENGTH) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId, {
    accessCode: accessCode.trim(),
    allowCreate: true,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: normalizedWorkspaceId, setCookieCode } = workspaceAccess;
  if (question.length > MAX_QUESTION_LENGTH) {
    return NextResponse.json({ error: "Question too long" }, { status: 400 });
  }

  // Materialize question to data/{workspaceId}/query.txt
  await ensureDir(normalizedWorkspaceId);
  await writeData(`${normalizedWorkspaceId}/query.txt`, question);

  const existingSession = await readSession(normalizedWorkspaceId) ?? undefined;
  const session = rootFlowRunId
    ? appendSessionRootFlowRunId(existingSession, rootFlowRunId)
    : normalizeSession(existingSession);

  await writeSession(normalizedWorkspaceId, session);
  const response = NextResponse.json({ ok: true });
  if (setCookieCode) {
    setWorkspaceAccessCookie(response, normalizedWorkspaceId, setCookieCode);
  }
  return response;
}
