import { basename } from "node:path";
import { NextResponse } from "next/server";
import { type SessionWithQuestion, readQuestion, readSessions } from "../_shared";

export async function GET(_request: Request, { params }: { params: Promise<{ userId: string }> }) {
  const { userId } = await params;
  const normalizedUserId = basename(userId);

  if (
    !normalizedUserId ||
    normalizedUserId !== userId ||
    normalizedUserId === "." ||
    normalizedUserId === ".."
  ) {
    return NextResponse.json({ error: "Invalid userId format" }, { status: 400 });
  }

  try {
    const sessions = await readSessions();
    const session = sessions[normalizedUserId];
    if (session) {
      const question = await readQuestion(normalizedUserId);
      const response: SessionWithQuestion = { ...session, question };
      return NextResponse.json(response);
    }
  } catch {
    // Fall through
  }

  return NextResponse.json({ error: "Session not found" }, { status: 404 });
}
