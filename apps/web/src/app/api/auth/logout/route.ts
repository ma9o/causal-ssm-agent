import { NextResponse } from "next/server";
import { clearOpenRouterSession } from "@/lib/server/openrouter-session";

export async function POST() {
  await clearOpenRouterSession();
  return NextResponse.json({ ok: true });
}
