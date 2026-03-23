import { basename } from "node:path";
import { NextResponse } from "next/server";
import type { SessionResponse } from "@/lib/api/analysis";
import { readQuestion, readSession } from "../_shared";

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
    const session = await readSession(normalizedUserId);
    if (session) {
      const question = await readQuestion(normalizedUserId);
      const response: SessionResponse = { ...session, question };
      return NextResponse.json(response);
    }
  } catch {
    // Fall through
  }

  return NextResponse.json({ error: "Session not found" }, { status: 404 });
}
