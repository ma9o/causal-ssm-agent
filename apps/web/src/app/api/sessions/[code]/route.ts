import { NextResponse } from "next/server";
import { type SessionWithQuestion, readQuestion, readSessions } from "../_shared";

export async function GET(_request: Request, { params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const normalizedCode = code.toUpperCase();

  try {
    const sessions = await readSessions();
    const session = sessions[normalizedCode];
    if (session) {
      const question = await readQuestion(normalizedCode);
      const response: SessionWithQuestion = { ...session, question };
      return NextResponse.json(response);
    }
  } catch {
    // Fall through
  }

  return NextResponse.json({ error: "Session not found" }, { status: 404 });
}
