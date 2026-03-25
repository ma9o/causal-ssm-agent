import { NextResponse } from "next/server";
import {
  findCausalInferenceDeploymentId,
  findFlowRunIdByIdempotencyKey,
  launchWorkspaceRootFlowRun,
  PrefectRunError,
} from "@/lib/server/prefect-runs";
import {
  requireWorkspaceAccess,
  setWorkspaceAccessCookie,
} from "@/lib/workspace-access";

function buildInitialRunIdempotencyKey(
  workspaceId: string,
  launchId: string,
): string {
  return `launch:${workspaceId}:${launchId}`;
}

export async function POST(request: Request) {
  const { workspaceId, accessCode, launchId, query } = await request.json();

  if (typeof workspaceId !== "string" || !workspaceId.trim()) {
    return NextResponse.json(
      { error: "workspaceId is required" },
      { status: 400 },
    );
  }
  if (typeof accessCode !== "string" || !accessCode.trim()) {
    return NextResponse.json(
      { error: "accessCode is required" },
      { status: 400 },
    );
  }
  if (typeof launchId !== "string" || !launchId.trim()) {
    return NextResponse.json(
      { error: "launchId is required" },
      { status: 400 },
    );
  }
  if (typeof query !== "string" || !query.trim()) {
    return NextResponse.json({ error: "query is required" }, { status: 400 });
  }

  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId, {
    accessCode: accessCode.trim(),
    allowCreate: false,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }

  try {
    const deploymentId = await findCausalInferenceDeploymentId();
    if (!deploymentId) {
      return NextResponse.json(
        { error: "causal-inference deployment not found" },
        { status: 502 },
      );
    }

    const idempotencyKey = buildInitialRunIdempotencyKey(
      workspaceAccess.workspaceId,
      launchId.trim(),
    );
    const existingFlowRunId = await findFlowRunIdByIdempotencyKey(
      deploymentId,
      idempotencyKey,
    );
    if (existingFlowRunId) {
      const response = NextResponse.json({ rootFlowRunId: existingFlowRunId });
      if (workspaceAccess.setCookieCode) {
        setWorkspaceAccessCookie(
          response,
          workspaceAccess.workspaceId,
          workspaceAccess.setCookieCode,
        );
      }
      return response;
    }

    const launch = await launchWorkspaceRootFlowRun({
      deploymentId,
      idempotencyKey,
      parameters: {
        workspace_id: workspaceAccess.workspaceId,
        query: query.trim(),
      },
      workspaceId: workspaceAccess.workspaceId,
    });
    if (launch.status === "busy") {
      const busyResponse = NextResponse.json(
        {
          error: launch.message,
          ...(launch.rootFlowRunId
            ? { rootFlowRunId: launch.rootFlowRunId }
            : {}),
        },
        { status: 409 },
      );
      if (workspaceAccess.setCookieCode) {
        setWorkspaceAccessCookie(
          busyResponse,
          workspaceAccess.workspaceId,
          workspaceAccess.setCookieCode,
        );
      }
      return busyResponse;
    }

    const response = NextResponse.json({ rootFlowRunId: launch.rootFlowRunId });
    if (workspaceAccess.setCookieCode) {
      setWorkspaceAccessCookie(
        response,
        workspaceAccess.workspaceId,
        workspaceAccess.setCookieCode,
      );
    }
    return response;
  } catch (error) {
    if (error instanceof PrefectRunError) {
      return NextResponse.json(
        { error: error.message },
        { status: error.status },
      );
    }

    return NextResponse.json(
      { error: "Failed to trigger pipeline" },
      { status: 502 },
    );
  }
}
