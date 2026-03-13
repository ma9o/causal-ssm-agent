import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";
import { DATA_DIR, readSessions, writeSessions } from "./_shared";

const CODE_RE = /^[A-Z0-9]{6}$/;
const MAX_QUESTION_LENGTH = 2000;

export async function POST(request: Request) {
  const body = await request.json();
  const { code, question, flowRunId } = body as {
    code?: string;
    question?: string;
    flowRunId?: string;
  };

  if (!code || !question) {
    return NextResponse.json({ error: "code and question are required" }, { status: 400 });
  }

  const normalizedCode = code.toUpperCase();
  if (!CODE_RE.test(normalizedCode)) {
    return NextResponse.json({ error: "Invalid session code format" }, { status: 400 });
  }
  if (question.length > MAX_QUESTION_LENGTH) {
    return NextResponse.json({ error: "Question too long" }, { status: 400 });
  }

  // Materialize question to data/{code}/query.txt
  const codeDir = join(DATA_DIR, normalizedCode);
  await mkdir(codeDir, { recursive: true });
  await writeFile(join(codeDir, "query.txt"), question);

  // Store session metadata (without question — it lives on disk)
  const sessions = await readSessions();
  sessions[normalizedCode] = {
    createdAt: new Date().toISOString(),
    ...(flowRunId ? { flowRunId } : {}),
  };

  await writeSessions(sessions);
  return NextResponse.json({ ok: true });
}
