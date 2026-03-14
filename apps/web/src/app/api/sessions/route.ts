import { mkdir, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { NextResponse } from "next/server";
import {
  DATA_DIR,
  appendSessionRootFlowRunId,
  normalizeSession,
  readSessions,
  writeSessions,
} from "./_shared";

const MAX_USER_ID_LENGTH = 200;
const MAX_QUESTION_LENGTH = 2000;

function parseUserId(userId: string): string | null {
  const trimmed = userId.trim();
  if (!trimmed || trimmed.length > MAX_USER_ID_LENGTH) {
    return null;
  }

  const safeUserId = basename(trimmed);
  if (safeUserId !== trimmed || safeUserId === "." || safeUserId === "..") {
    return null;
  }

  return safeUserId;
}

export async function POST(request: Request) {
  const body = await request.json();
  const { userId, question, rootFlowRunId } = body as {
    userId?: string;
    question?: string;
    rootFlowRunId?: string;
  };

  if (!userId || !question) {
    return NextResponse.json({ error: "userId and question are required" }, { status: 400 });
  }

  const normalizedUserId = parseUserId(userId);
  if (!normalizedUserId) {
    return NextResponse.json({ error: "Invalid userId format" }, { status: 400 });
  }
  if (question.length > MAX_QUESTION_LENGTH) {
    return NextResponse.json({ error: "Question too long" }, { status: 400 });
  }

  // Materialize question to data/{userId}/query.txt
  const userDir = join(DATA_DIR, normalizedUserId);
  await mkdir(userDir, { recursive: true });
  await writeFile(join(userDir, "query.txt"), question);

  // Store session metadata (without question — it lives on disk)
  const sessions = await readSessions();
  const existingSession = sessions[normalizedUserId];

  sessions[normalizedUserId] = rootFlowRunId
    ? appendSessionRootFlowRunId(existingSession, rootFlowRunId)
    : normalizeSession(existingSession);

  await writeSessions(sessions);
  return NextResponse.json({ ok: true });
}
