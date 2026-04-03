import { NextResponse } from "next/server";
import { clearOpenRouterSession } from "@/lib/server/openrouter-session";
import { clearAuthorizedWorkspaceIds } from "@/lib/server/workspace-session";

export async function POST() {
  await clearAuthorizedWorkspaceIds();
  await clearOpenRouterSession();
  return NextResponse.json({ ok: true });
}
