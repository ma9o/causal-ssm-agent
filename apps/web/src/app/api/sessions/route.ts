import { writeFile } from "node:fs/promises";
import { NextResponse } from "next/server";
import { SESSIONS_PATH, readSessions } from "./_shared";

export async function POST(request: Request) {
  const body = await request.json();
  const { code, runId, question } = body as {
    code?: string;
    runId?: string;
    question?: string;
  };

  if (!code || !runId || !question) {
    return NextResponse.json(
      { error: "code, runId, and question are required" },
      { status: 400 },
    );
  }

  const normalizedCode = code.toUpperCase();
  const sessions = await readSessions();
  sessions[normalizedCode] = {
    runId,
    question,
    createdAt: new Date().toISOString(),
  };

  await writeFile(SESSIONS_PATH, JSON.stringify(sessions, null, 2));
  return NextResponse.json({ ok: true });
}
