import { NextResponse } from "next/server";
import { listWorkspaces } from "@/lib/server/workspaces";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(await listWorkspaces());
}
