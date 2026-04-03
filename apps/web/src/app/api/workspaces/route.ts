import { NextResponse } from "next/server";
import { listAccessibleWorkspaces } from "@/lib/server/workspace-ownership";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(await listAccessibleWorkspaces());
}
