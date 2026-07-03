import { NextResponse } from "next/server";
import { ensureDir, writeBinary } from "@/lib/storage";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("file") as File | null;
  const workspaceId = formData.get("workspaceId") as string | null;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }
  if (!workspaceId) {
    return NextResponse.json({ error: "No workspaceId provided" }, { status: 400 });
  }
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  const rawFileName = typeof file.name === "string" ? file.name : "";
  const safeFileName = rawFileName.split("/").at(-1)?.split("\\").at(-1) ?? "";
  if (!safeFileName) {
    return NextResponse.json({ error: "Invalid file name" }, { status: 400 });
  }

  const relativePath = `${safeWorkspaceId}/input/${safeFileName}`;
  await ensureDir(`${safeWorkspaceId}/input`);

  const buffer = Buffer.from(await file.arrayBuffer());
  await writeBinary(relativePath, buffer);

  return NextResponse.json({ path: relativePath });
}
