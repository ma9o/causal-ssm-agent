import { NextResponse } from "next/server";
import {
  findCausalInferenceDeploymentId,
  findFlowRunIdByIdempotencyKey,
  launchWorkspaceRootFlowRun,
  PrefectRunError,
} from "@/lib/server/prefect-runs";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

function buildInitialRunIdempotencyKey(
  workspaceId: string,
  launchId: string,
): string {
  return `launch:${workspaceId}:${launchId}`;
}

export async function POST(request: Request) {
  const { workspaceId, launchId, query } = await request.json();

  if (typeof workspaceId !== "string" || !workspaceId.trim()) {
    return NextResponse.json(
      { error: "workspaceId is required" },
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
    requireMutable: true,
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
      return NextResponse.json({ rootFlowRunId: existingFlowRunId });
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
      return NextResponse.json(
        {
          error: launch.message,
          ...(launch.rootFlowRunId
            ? { rootFlowRunId: launch.rootFlowRunId }
            : {}),
        },
        { status: 409 },
      );
    }

    return NextResponse.json({ rootFlowRunId: launch.rootFlowRunId });
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
