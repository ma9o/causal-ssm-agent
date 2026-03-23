import { NextResponse } from "next/server";
import { writeBinary, ensureDir } from "@/lib/storage";
import {
  requireWorkspaceAccess,
  setWorkspaceAccessCookie,
} from "@/lib/workspace-access";

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("file") as File | null;
  const workspaceId = formData.get("workspaceId") as string | null;
  const accessCode = formData.get("accessCode") as string | null;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }
  if (!workspaceId) {
    return NextResponse.json({ error: "No workspaceId provided" }, { status: 400 });
  }
  if (!accessCode) {
    return NextResponse.json({ error: "No accessCode provided" }, { status: 400 });
  }

  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId, {
    accessCode: accessCode.trim(),
    allowCreate: true,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: normalizedWorkspaceId, setCookieCode } = workspaceAccess;

  const safeFileName = file.name.split("/").at(-1)?.split("\\").at(-1) ?? "";
  if (!safeFileName) {
    return NextResponse.json({ error: "Invalid file name" }, { status: 400 });
  }

  const relativePath = `${normalizedWorkspaceId}/input/${safeFileName}`;
  await ensureDir(`${normalizedWorkspaceId}/input`);

  const buffer = Buffer.from(await file.arrayBuffer());
  await writeBinary(relativePath, buffer);

  const response = NextResponse.json({ path: relativePath });
  if (setCookieCode) {
    setWorkspaceAccessCookie(response, normalizedWorkspaceId, setCookieCode);
  }
  return response;
}
