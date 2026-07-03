import { NextResponse } from "next/server";
import { getFacadeCapabilities } from "@/lib/server/episode-runs";

export const dynamic = "force-dynamic";

/**
 * GET /api/capabilities — proxy over the facade's capability report.
 * moves_enabled=false means the backing facade is read-only (hosted
 * viewer): the UI hides run/edit/recompute/simulate affordances.
 */
export async function GET() {
  try {
    return NextResponse.json(await getFacadeCapabilities());
  } catch {
    return NextResponse.json({ error: "Failed to load capabilities" }, { status: 502 });
  }
}
