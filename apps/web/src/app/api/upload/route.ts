import { NextResponse } from "next/server";
import { deleteData, ensureDir, writeBinary } from "@/lib/storage";
import { finalizeWorkspaceCreate, requireWorkspaceAccess } from "@/lib/workspace-access";

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

  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId, {
    allowCreate: true,
    requireMutable: true,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const {
    workspaceId: normalizedWorkspaceId,
    creationPending,
  } = workspaceAccess;

  const rawFileName = typeof file.name === "string" ? file.name : "";
  const safeFileName = rawFileName.split("/").at(-1)?.split("\\").at(-1) ?? "";
  if (!safeFileName) {
    return NextResponse.json({ error: "Invalid file name" }, { status: 400 });
  }

  const relativePath = `${normalizedWorkspaceId}/input/${safeFileName}`;
  await ensureDir(`${normalizedWorkspaceId}/input`);

  const buffer = Buffer.from(await file.arrayBuffer());
  await writeBinary(relativePath, buffer);
  try {
    if (creationPending) {
      await finalizeWorkspaceCreate(normalizedWorkspaceId);
    }
  } catch (e) {
    try {
      await deleteData(relativePath);
    } catch (cleanupError) {
      console.error(`Failed to roll back upload for '${normalizedWorkspaceId}':`, cleanupError);
    }
    console.error(`Failed to finalize workspace '${normalizedWorkspaceId}':`, e);
    return NextResponse.json(
      { error: "Failed to finalize workspace creation" },
      { status: 500 },
    );
  }

  return NextResponse.json({ path: relativePath });
}
