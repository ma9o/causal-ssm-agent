import { NextResponse } from "next/server";
import { EpisodeRunError, getEpisodeTrace } from "@/lib/server/episode-runs";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string }> },
) {
  const { workspaceId } = await params;
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  const ref = new URL(request.url).searchParams.get("ref")?.trim();
  if (!ref) {
    return NextResponse.json({ error: "Missing trace ref" }, { status: 400 });
  }

  try {
    return NextResponse.json(await getEpisodeTrace(safeWorkspaceId, ref));
  } catch (error) {
    const status = error instanceof EpisodeRunError && error.status === 404 ? 404 : 502;
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status },
    );
  }
}
