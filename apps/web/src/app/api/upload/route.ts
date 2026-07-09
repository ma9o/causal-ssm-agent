import { NextResponse } from "next/server";
import { getToolServerUrl } from "@/lib/runtime-urls";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

const TOOL_SERVER = getToolServerUrl();

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
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  const rawFileName = typeof file.name === "string" ? file.name : "";
  const safeFileName = rawFileName.split("/").at(-1)?.split("\\").at(-1) ?? "";
  if (!safeFileName) {
    return NextResponse.json({ error: "Invalid file name" }, { status: 400 });
  }

  formData.set("workspaceId", safeWorkspaceId);
  const upstream = await fetch(`${TOOL_SERVER}/api/upload`, {
    method: "POST",
    body: formData,
  });

  const body = await upstream.text();
  return new Response(body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
    },
  });
}
