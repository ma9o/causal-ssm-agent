import { NextResponse } from "next/server";
import { readSessions } from "../_shared";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ code: string }> },
) {
  const { code } = await params;
  const normalizedCode = code.toUpperCase();

  try {
    const sessions = await readSessions();
    const session = sessions[normalizedCode];

    if (!session) {
      return NextResponse.json({ error: "Session not found" }, { status: 404 });
    }

    return NextResponse.json(session);
  } catch {
    return NextResponse.json({ error: "Session not found" }, { status: 404 });
  }
}
