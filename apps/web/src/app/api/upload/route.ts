import { mkdir, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { NextResponse } from "next/server";

const DATA_DIR = resolve(process.cwd(), "..", "..", "data");

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("file") as File | null;
  const code = formData.get("code") as string | null;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }
  if (!code) {
    return NextResponse.json({ error: "No code provided" }, { status: 400 });
  }

  // Sanitize path components to prevent directory traversal
  const safeCode = basename(code);
  const safeFileName = basename(file.name);

  const dir = join(DATA_DIR, safeCode, "input");
  await mkdir(dir, { recursive: true });

  const buffer = Buffer.from(await file.arrayBuffer());
  const filePath = join(dir, safeFileName);

  // Final safety check: ensure resolved path stays within DATA_DIR
  if (!resolve(filePath).startsWith(DATA_DIR)) {
    return NextResponse.json({ error: "Invalid file path" }, { status: 400 });
  }

  await writeFile(filePath, buffer);

  return NextResponse.json({ path: filePath });
}
