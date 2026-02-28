import { writeFile } from "node:fs/promises";
import { NextResponse } from "next/server";
import { SESSIONS_PATH, readSessions } from "./_shared";

const CODE_RE = /^[A-Z0-9]{6}$/;
const MAX_QUESTION_LENGTH = 2000;
const MAX_RUN_ID_LENGTH = 200;

export async function POST(request: Request) {
  const body = await request.json();
  const { code, runId, question } = body as {
    code?: string;
    runId?: string;
    question?: string;
  };

  if (!code || !runId || !question) {
    return NextResponse.json({ error: "code, runId, and question are required" }, { status: 400 });
  }

  const normalizedCode = code.toUpperCase();
  if (!CODE_RE.test(normalizedCode)) {
    return NextResponse.json({ error: "Invalid session code format" }, { status: 400 });
  }
  if (runId.length > MAX_RUN_ID_LENGTH) {
    return NextResponse.json({ error: "runId too long" }, { status: 400 });
  }
  if (question.length > MAX_QUESTION_LENGTH) {
    return NextResponse.json({ error: "Question too long" }, { status: 400 });
  }

  const sessions = await readSessions();
  sessions[normalizedCode] = {
    runId,
    question,
    createdAt: new Date().toISOString(),
  };

  await writeFile(SESSIONS_PATH, JSON.stringify(sessions, null, 2));
  return NextResponse.json({ ok: true });
}
