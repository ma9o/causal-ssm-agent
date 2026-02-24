import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

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

  const dir = join(process.cwd(), "..", "data-pipeline", "data", "raw", userId);
  await mkdir(dir, { recursive: true });

  const buffer = Buffer.from(await file.arrayBuffer());
  const filePath = join(dir, file.name);
  await writeFile(filePath, buffer);

  return NextResponse.json({ path: filePath });
}
