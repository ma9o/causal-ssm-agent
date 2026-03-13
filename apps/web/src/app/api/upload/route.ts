import { mkdir, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { NextResponse } from "next/server";

const DATA_DIR = resolve(process.cwd(), "..", "..", "data");

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("file") as File | null;
  const userId = formData.get("userId") as string | null;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }
  if (!userId) {
    return NextResponse.json({ error: "No userId provided" }, { status: 400 });
  }

  // Sanitize path components to prevent directory traversal
  const trimmedUserId = userId.trim();
  const safeUserId = basename(trimmedUserId);
  const safeFileName = basename(file.name);

  if (
    !safeUserId ||
    safeUserId !== trimmedUserId ||
    safeUserId === "." ||
    safeUserId === ".."
  ) {
    return NextResponse.json({ error: "Invalid userId format" }, { status: 400 });
  }

  const dir = join(DATA_DIR, safeUserId, "input");
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
