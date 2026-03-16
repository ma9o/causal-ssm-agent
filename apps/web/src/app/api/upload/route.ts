import { basename } from "node:path";
import { NextResponse } from "next/server";
import { writeBinary, ensureDir } from "@/lib/storage";

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

  const relativePath = `${safeUserId}/input/${safeFileName}`;
  await ensureDir(`${safeUserId}/input`);

  const buffer = Buffer.from(await file.arrayBuffer());
  await writeBinary(relativePath, buffer);

  return NextResponse.json({ path: relativePath });
}
