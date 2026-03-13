import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { NextResponse } from "next/server";

const DATA_DIR = resolve(process.cwd(), "..", "..", "data");

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ code: string; stage: string }> },
) {
  const { code, stage } = await params;

  // Sanitize path components to prevent directory traversal
  const safeCode = basename(code);
  const safeStage = basename(stage);

  const filePath = resolve(join(DATA_DIR, safeCode, "run", `${safeStage}.json`));
  if (!filePath.startsWith(DATA_DIR)) {
    return NextResponse.json({ error: "Invalid path" }, { status: 400 });
  }

  try {
    const raw = await readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw);
    // Prefect wraps persisted results as { metadata, result: "<JSON string>" }.
    // Unwrap to return the actual stage data.
    if (parsed.metadata && typeof parsed.result === "string") {
      return NextResponse.json(JSON.parse(parsed.result));
    }
    return NextResponse.json(parsed);
  } catch {
    return NextResponse.json({ error: `No data for ${stage}` }, { status: 404 });
  }
}
