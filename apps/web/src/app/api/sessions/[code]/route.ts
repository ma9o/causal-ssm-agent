import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { NextResponse } from "next/server";
import { readSessions } from "../_shared";

const FIXTURES_DIR = resolve(process.cwd(), "test", "fixtures");

export async function GET(_request: Request, { params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const normalizedCode = code.toUpperCase();

  // 1. Try real session store
  try {
    const sessions = await readSessions();
    const session = sessions[normalizedCode];
    if (session) {
      return NextResponse.json(session);
    }
  } catch {
    // Fall through
  }

  // 2. Try fixture (code doubles as fixture directory name)
  try {
    const safeCode = basename(code.toLowerCase());
    const fixturePath = resolve(join(FIXTURES_DIR, safeCode, "session.json"));
    if (!fixturePath.startsWith(FIXTURES_DIR)) {
      return NextResponse.json({ error: "Invalid session code" }, { status: 400 });
    }
    const data = await readFile(fixturePath, "utf-8");
    return NextResponse.json(JSON.parse(data));
  } catch {
    // No fixture either
  }

  return NextResponse.json({ error: "Session not found" }, { status: 404 });
}
