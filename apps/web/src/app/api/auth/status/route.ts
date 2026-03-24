import { NextResponse } from "next/server";
import { getOpenRouterStatus } from "@/lib/server/openrouter-access";

export async function GET() {
  return NextResponse.json(await getOpenRouterStatus());
}
