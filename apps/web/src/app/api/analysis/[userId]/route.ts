import { basename } from "node:path";
import { NextResponse } from "next/server";
import { buildAnalysisManifest } from "../_shared";

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
    const url = new URL(_request.url);
    const bootstrapRootFlowRunIds = url.searchParams.getAll("rootFlowRunId").filter(Boolean);
    const manifest = await buildAnalysisManifest(normalizedUserId, bootstrapRootFlowRunIds);
    if (manifest) {
      return NextResponse.json(manifest);
    }
  } catch {
    // Fall through
  }

  return NextResponse.json({ error: "Analysis manifest not found" }, { status: 404 });
}
